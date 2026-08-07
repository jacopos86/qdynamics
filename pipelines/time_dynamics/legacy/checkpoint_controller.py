#!/usr/bin/env python3
"""Legacy exact/oracle HH adaptive realtime checkpoint controller.

This module preserves the old checkpoint-controller runtime for compatibility
and diagnostics. It is not the Paper-II AP-McLachlan route identity.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
import time
from typing import TYPE_CHECKING, Any, Mapping, MutableMapping, Sequence

import numpy as np

from pipelines.scaffold.hh_continuation_scoring import (
    FullScoreConfig,
    MeasurementCacheAudit,
    Phase1CompileCostOracle,
    Phase2NoveltyOracle,
    shortlist_records,
)
from pipelines.scaffold.hh_continuation_stage_control import allowed_positions
from pipelines.time_dynamics.legacy.checkpoint_types import (
    HIGH_MISS_NO_ADMIT_POLICY_CANONICAL,
    HIGH_MISS_NO_ADMIT_POLICY_DEFAULT,
    HIGH_MISS_NO_ADMIT_SOFT_FALLBACK_REASON,
    HIGH_MISS_NO_ADMIT_SOFT_FALLBACK_WARNING,
    BaselineGeometrySummary,
    CandidateProbeSummary,
    CheckpointLedgerEntry,
    DerivedGeometryKey,
    GeometryValueKey,
    OracleValueKey,
    RealtimeCheckpointConfig,
    dataclass_to_payload,
    decision_data_flow_fields,
    full_horizon_completion_fields,
    hash_measurement_state,
    high_miss_no_admit_diagnostic_counts,
    high_miss_no_admit_soft_fallback_counts,
    is_successful_stable_early_stop_reason,
    make_checkpoint_context,
    make_measurement_checkpoint_context,
    normalize_high_miss_no_admit_policy,
    normalize_reference_mode,
    normalize_realtime_controller_mode,
    physical_trajectory_rows,
    strict_qpu_faithful_decision_contract,
    trajectory_repair_counts,
)
from pipelines.time_dynamics.legacy.checkpoint_measurement import (
    BackendScheduledRawGroupPool,
    DerivedGeometryMemo,
    ExactCheckpointValueCache,
    OracleCheckpointValueCache,
    TemporalMeasurementLedger,
    build_controller_oracle_tier_configs,
    controller_oracle_supports_raw_group_sampling,
    estimate_grouped_raw_mclachlan_incremental_block,
    estimate_grouped_raw_mclachlan_geometry,
    estimate_observable_specs,
    planning_group_keys_for_term,
    planning_stats_for_term,
    validate_controller_oracle_base_config,
    validate_controller_tiers_mean_only,
)
from pipelines.time_dynamics.adapters.observables import (
    auto_primary_density_mode,
    measured_snapshot_from_estimates,
    observable_measurement_bundle_for_problem,
    observable_snapshot_for_state,
    primary_density_value_from_snapshot,
    summary_fields_from_row,
)
from pipelines.time_dynamics.legacy import checkpoint_motion as _realtime_motion
from pipelines.time_dynamics.legacy import checkpoint_progress as _realtime_progress
from pipelines.time_dynamics.legacy.checkpoint_prune_loss import (
    COMPAT_SCHUR_NORMALIZED_V1,
    DENOM_MAX_NORM_B_EPS_COMPAT_V1,
    LEGACY_PROXY_V1,
    MATRIX_COMPAT_SCHUR_K,
    MATRIX_LEGACY_PROXY,
    compute_prune_loss_payload,
    selected_prune_loss_payload,
)
from pipelines.scaffold.hh_vqe_from_adapt_family import ReplayScaffoldContext
from src.quantum.ansatz_parameterization import (
    AnsatzParameterLayout,
    GeneratorParameterBlock,
    RotationTermSpec,
    build_parameter_layout,
    runtime_insert_position,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.compiled_polynomial import (
    apply_compiled_polynomial,
    compile_polynomial_action,
)
from src.quantum.drives_time_potential import (
    build_gaussian_sinusoid_density_drive,
    default_spatial_weights,
    reference_method_name,
)
from pipelines.time_dynamics.adapters.drive_terms import (
    RealtimeDriveModel,
    resolve_realtime_drive_model,
)
from pipelines.time_dynamics.adapters.hamiltonian import (
    DRIVEN_HAMILTONIAN_FLOW_FAMILIES as _DRIVEN_HAMILTONIAN_FLOW_FAMILIES,
    HAMILTONIAN_FLOW_FAMILIES as _HAMILTONIAN_FLOW_FAMILIES,
    SPINFUL_LATTICE_FAMILIES as _STATIC_SPINFUL_LATTICE_HAMILTONIAN_FLOW_FAMILIES,
    SPINFUL_LATTICE_FAMILIES as _DRIVEN_SPINFUL_LATTICE_HAMILTONIAN_FLOW_FAMILIES,
    adapter_for_family_key,
)
from pipelines.time_dynamics.fixed_manifold.observables import ObservableSpec
from src.quantum.hubbard_latex_python_pairs import build_hubbard_holstein_drive
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, PauliPolynomial, hamiltonian_matrix


def build_runtime_layout_circuit(*args, **kwargs):
    from pipelines.exact_bench.noise_oracle_runtime import build_runtime_layout_circuit as _impl

    return _impl(*args, **kwargs)


def pauli_poly_to_sparse_pauli_op(*args, **kwargs):
    from pipelines.exact_bench.noise_oracle_runtime import pauli_poly_to_sparse_pauli_op as _impl

    return _impl(*args, **kwargs)

if TYPE_CHECKING:
    from pipelines.hardcoded.hh_fixed_manifold_measured import (
        FixedManifoldMeasuredConfig,
    )


@dataclass(frozen=True)
class RuntimeTermCarrier:
    label: str
    source_label: str
    polynomial: PauliPolynomial
    runtime_specs: tuple[RotationTermSpec, ...]
    repetition_index: int | None = None
    source_logical_index: int | None = None
    candidate_pool_index: int | None = None


@dataclass(frozen=True)
class ControllerRunArtifacts:
    trajectory: list[dict[str, Any]]
    ledger: list[dict[str, Any]]
    summary: dict[str, Any]
    reference: dict[str, Any]


@dataclass(frozen=True)
class ControllerDriveConfig:
    enabled: bool
    n_sites: int
    ordering: str
    drive_A: float
    drive_omega: float
    drive_tbar: float
    drive_phi: float
    drive_pattern: str
    drive_custom_weights: tuple[float, ...] | None = None
    drive_include_identity: bool = False
    drive_time_sampling: str = "midpoint"
    drive_t0: float = 0.0
    exact_steps_multiplier: int = 1


def _resolve_drive_config_for_register(
    drive_config: ControllerDriveConfig | None,
    *,
    num_qubits: int,
    family_key: str | None = None,
) -> ControllerDriveConfig | None:
    if drive_config is None or not bool(drive_config.enabled):
        return None
    requested_sites = int(drive_config.n_sites)
    if requested_sites <= 0:
        return None
    family_norm = "" if family_key is None else str(family_key).strip().lower()
    adapter = adapter_for_family_key(family_norm or "hh")
    if adapter.observable_kind in {"spinless_lattice", "boson_chain"}:
        if requested_sites <= int(num_qubits):
            return drive_config
        return replace(drive_config, n_sites=int(num_qubits))
    max_sites = int(num_qubits) // 2
    if 2 * requested_sites <= int(num_qubits):
        return drive_config
    if int(num_qubits) == 1 and requested_sites == 1:
        return drive_config
    if max_sites <= 0:
        return None
    # Some toy replay fixtures intentionally use a reduced register while still
    # exercising the driven-time bookkeeping path. Clamp the drive footprint to
    # the available spin-orbital register instead of failing during startup.
    return replace(drive_config, n_sites=int(max_sites))


STRICT_QPU_FAITHFUL_DECISION_PATH_KIND = "strict_qpu_faithful_observable_v1"
QPU_FAITHFUL_CONTROLLER_MODES = {"oracle_v1", "observable_v1"}
AUTO_EULER_RK4_POLICY_SCHEMA = "auto_euler_rk4_policy_v2"


def _auto_euler_blocker_diagnostics(
    *,
    geometry_gate_pass: bool,
    euler_error_pass: bool,
    condition_pass: bool,
    rho_miss_pass: bool,
    euler_time_gate_pass: bool,
    observable_gate_pass: bool,
) -> dict[str, Any]:
    blockers: list[str] = []
    if not bool(geometry_gate_pass):
        blockers.append("geometry")
    if not bool(euler_error_pass):
        blockers.append("embedded_euler_error")
    if not bool(condition_pass):
        blockers.append("condition")
    if not bool(rho_miss_pass):
        blockers.append("rho_miss")
    if not bool(euler_time_gate_pass):
        blockers.append("early_time_prior")
    if not bool(observable_gate_pass):
        blockers.append("observable_span")
    return {
        "integrator_auto_policy_schema": AUTO_EULER_RK4_POLICY_SCHEMA,
        "integrator_geometry_gate_pass": bool(geometry_gate_pass),
        "integrator_euler_error_pass": bool(euler_error_pass),
        "integrator_auto_admit_euler": not blockers,
        "integrator_euler_blockers": list(blockers),
    }


def _controller_family_key(
    *,
    resolved_problem: Any | None,
    replay_context: ReplayScaffoldContext,
) -> str:
    if resolved_problem is not None and getattr(resolved_problem, "family_key", None) not in {None, ""}:
        return str(getattr(resolved_problem, "family_key")).strip().lower()
    return "hh"


def _validate_driven_controller_mode_for_adapter(
    *,
    family_key: str,
    cfg: RealtimeCheckpointConfig,
    strict_qpu_faithful: bool,
) -> None:
    adapter = adapter_for_family_key(str(family_key))
    capabilities = adapter.capabilities
    if not bool(capabilities.supports_driven_realtime):
        raise ValueError(
            f"Driven neutral realtime currently has no drive-term seam for family {family_key!r}."
        )
    if bool(strict_qpu_faithful):
        if str(cfg.mode) not in QPU_FAITHFUL_CONTROLLER_MODES:
            raise ValueError(
                f"Driven {family_key} strict QPU-faithful realtime requires "
                "--checkpoint-controller-mode observable_v1 or oracle_v1."
            )
        if str(cfg.reference_mode) != "off":
            raise ValueError(
                f"Driven {family_key} strict QPU-faithful realtime requires controller exact inputs off."
            )
        return
    if bool(capabilities.supports_drive_mode_off):
        if str(cfg.mode) not in {"off", "exact_v1"}:
            raise ValueError(
                f"Driven {family_key} neutral realtime currently supports --checkpoint-controller-mode off or exact_v1."
            )
        if str(cfg.mode) == "exact_v1" and str(cfg.reference_mode) != "benchmark_exact":
            raise ValueError(
                f"Driven {family_key} exact_v1 currently requires --checkpoint-controller-reference-mode benchmark_exact."
            )
        if str(cfg.reference_mode) not in {"off", "benchmark_exact"}:
            raise ValueError(
                f"Driven {family_key} neutral realtime currently requires --checkpoint-controller-reference-mode off or benchmark_exact."
            )
        return
    if str(cfg.mode) != "exact_v1":
        raise ValueError(
            f"Driven {family_key} neutral realtime requires --checkpoint-controller-mode exact_v1."
        )
    if str(cfg.reference_mode) != "benchmark_exact":
        raise ValueError(
            f"Driven {family_key} neutral realtime requires --checkpoint-controller-reference-mode benchmark_exact."
        )


def _pauli_poly_from_real_coeff_map(
    coeff_map: Mapping[str, complex],
    *,
    nq: int,
    drop_abs_tol: float,
    hermiticity_tol: float,
    context: str,
) -> PauliPolynomial:
    poly = PauliPolynomial("JW")
    for label, coeff in sorted(
        ((str(key), complex(value)) for key, value in dict(coeff_map).items()),
        key=lambda item: item[0],
    ):
        if abs(coeff) <= float(drop_abs_tol):
            continue
        if abs(coeff.imag) > float(hermiticity_tol):
            raise ValueError(f"{context} produced non-Hermitian coefficient for {label}: {coeff}.")
        poly.add_term(PauliTerm(int(nq), ps=str(label), pc=float(coeff.real)))
    return poly


def _is_computational_basis_statevector(
    psi: np.ndarray | Sequence[complex],
    *,
    tol: float = 1.0e-10,
) -> bool:
    arr = np.asarray(psi, dtype=complex).reshape(-1)
    if arr.size <= 0 or not np.all(np.isfinite(arr)):
        return False
    norm_sq = float(np.vdot(arr, arr).real)
    if abs(norm_sq - 1.0) > float(tol):
        return False
    idx = int(np.argmax(np.abs(arr)))
    max_amp = complex(arr[idx])
    rest = np.array(arr, copy=True)
    rest[idx] = 0.0
    return bool(
        abs(abs(max_amp) - 1.0) <= float(tol)
        and float(np.vdot(rest, rest).real) <= float(tol) ** 2
    )


def _statevector_validation_errors(
    psi: np.ndarray | Sequence[complex],
    *,
    label: str,
    tol: float = 1.0e-10,
) -> list[str]:
    arr = np.asarray(psi, dtype=complex).reshape(-1)
    errors: list[str] = []
    if arr.size <= 0:
        return [f"{label} is empty"]
    if int(arr.size) & (int(arr.size) - 1):
        errors.append(f"{label} dimension is not a power of two")
    if not np.all(np.isfinite(arr)):
        errors.append(f"{label} contains non-finite amplitudes")
    norm_sq = float(np.vdot(arr, arr).real)
    if abs(norm_sq - 1.0) > float(tol):
        errors.append(f"{label} is not normalized")
    return errors


def _statevector_sha256(psi: np.ndarray | Sequence[complex]) -> str:
    arr = np.asarray(psi, dtype=np.complex128).reshape(-1)
    payload = np.ascontiguousarray(
        np.stack([arr.real, arr.imag], axis=1).astype("<f8", copy=False)
    )
    return hashlib.sha256(payload.tobytes()).hexdigest()


def _strict_state_prep_metadata_from_replay_context(
    replay_context: Any,
) -> dict[str, Any]:
    for container_name in ("pool_meta", "append_pool_meta", "payload_in"):
        container = getattr(replay_context, container_name, None)
        if not isinstance(container, Mapping):
            continue
        for key in ("strict_state_prep_contract", "state_prep_contract"):
            value = container.get(key, None)
            if isinstance(value, Mapping):
                return dict(value)
    return {}


def _strict_state_text_is_exact_target_like(value: Any) -> bool:
    text = "" if value is None else str(value).strip().lower()
    if text == "":
        return False
    if "exact" in text:
        return True
    if re.search(r"(^|[_:\-./])ed([_:\-./]|$)", text):
        return True
    if text.startswith("ed") and ("ground" in text or "state" in text):
        return True
    forbidden_phrases = (
        "state_at",
        "target_state",
        "target_trajectory",
        "reference_trajectory",
        "benchmark_trajectory",
    )
    return any(phrase in text for phrase in forbidden_phrases)


def _strict_state_source_allowlist(block: Mapping[str, Any]) -> set[str]:
    raw = block.get("source_allowlist", ())
    if raw is None:
        values = ()
    elif isinstance(raw, (str, bytes, bytearray)):
        values = () if str(raw).strip() == "" else (raw,)
    elif isinstance(raw, Sequence):
        values = tuple(raw)
    else:
        values = ()
    return {
        str(item).strip().lower()
        for item in values
        if item not in {None, ""} and str(item).strip() != ""
    }


def _strict_state_prep_source_errors(
    metadata: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    ansatz = metadata.get("ansatz_input_state", {})
    initial = metadata.get("initial_state", {})
    if not isinstance(ansatz, Mapping):
        ansatz = {}
    if not isinstance(initial, Mapping):
        initial = {}

    ansatz_role = str(ansatz.get("role", "")).strip().lower()
    initial_role = str(initial.get("role", "")).strip().lower()
    if ansatz_role not in {"ansatz_input_state", "reference_state", "seed_ansatz_input_state"}:
        errors.append("non-basis state prep requires ansatz_input_state role metadata")
    if initial_role not in {"prepared_ansatz_state", "prepared_state", "reconstructed_prepared_state"}:
        errors.append("non-basis state prep requires prepared initial_state role metadata")

    ansatz_location = str(ansatz.get("source_location", "")).strip().lower()
    if ansatz_location not in {
        "payload.ansatz_input_state",
        "resolved_problem.reference_state",
        "runtime_input.psi_ref",
    }:
        errors.append("non-basis ansatz input state must come from the seed/artifact state-prep boundary")
    initial_location = str(initial.get("source_location", "")).strip().lower()
    if initial_location not in {
        "payload.initial_state",
        "runtime_loader.reconstructed_from_scaffold",
        "runtime_input.psi_initial",
    }:
        errors.append("strict initial state must be the seed prepared ansatz/circuit state")

    for prefix, block in (("ansatz_input_state", ansatz), ("initial_state", initial)):
        source = str(block.get("source", "")).strip().lower()
        source_allowlist = _strict_state_source_allowlist(block)
        if not source_allowlist:
            errors.append(f"strict_qpu_faithful state prep requires {prefix}.source_allowlist")
        elif source not in source_allowlist:
            errors.append(
                f"strict_qpu_faithful state prep source {prefix}.source={block.get('source')!r} "
                "is not allowlisted for the prepared seed boundary"
            )
        for field in ("source", "source_location", "handoff_state_kind"):
            if _strict_state_text_is_exact_target_like(block.get(field, None)):
                errors.append(
                    f"strict_qpu_faithful forbids exact-target/reference state prep metadata "
                    f"{prefix}.{field}={block.get(field)!r}"
                )
    if bool(metadata.get("exact_target_or_reference_trajectory", False)):
        errors.append("strict_qpu_faithful state prep metadata marks exact target/reference trajectory")
    return errors


@dataclass(frozen=True)
class StepHamiltonianArtifacts:
    physical_time: float
    h_poly: Any
    hmat: np.ndarray
    compiled_h: Any
    oracle_observable: Any | None
    drive_term_count: int


@dataclass(frozen=True)
class RepairAttemptState:
    attempt_index: int
    max_attempts: int | None
    escalation_kind: str | None


@dataclass(frozen=True)
class StateObservableSnapshot:
    n_up_site: np.ndarray
    n_dn_site: np.ndarray
    n_site: np.ndarray
    doublon: float
    staggered: float


def _spin_orbital_bit_index(site: int, spin: int, num_sites: int, ordering: str) -> int:
    ord_norm = str(ordering).strip().lower()
    if ord_norm == "blocked":
        return int(site) if int(spin) == 0 else int(num_sites) + int(site)
    if ord_norm == "interleaved":
        return (2 * int(site)) + int(spin)
    raise ValueError(f"Unsupported ordering {ordering!r}")


def _site_resolved_number_observables(
    psi: np.ndarray,
    *,
    num_sites: int,
    ordering: str,
) -> StateObservableSnapshot:
    probs = np.abs(np.asarray(psi, dtype=complex).reshape(-1)) ** 2
    n_up = np.zeros(int(num_sites), dtype=float)
    n_dn = np.zeros(int(num_sites), dtype=float)
    doublon_total = 0.0
    up_bits = [_spin_orbital_bit_index(site, 0, num_sites, ordering) for site in range(int(num_sites))]
    dn_bits = [_spin_orbital_bit_index(site, 1, num_sites, ordering) for site in range(int(num_sites))]

    for idx, prob in enumerate(probs):
        p = float(prob)
        if p <= 0.0:
            continue
        for site in range(int(num_sites)):
            up = int((idx >> up_bits[site]) & 1)
            dn = int((idx >> dn_bits[site]) & 1)
            n_up[site] += float(up) * p
            n_dn[site] += float(dn) * p
            doublon_total += float(up * dn) * p

    n_site = np.asarray(n_up + n_dn, dtype=float)
    if n_site.size == 0:
        staggered = float("nan")
    else:
        signs = np.array(
            [1.0 if (site % 2 == 0) else -1.0 for site in range(int(n_site.size))],
            dtype=float,
        )
        staggered = float(np.sum(signs * n_site) / float(n_site.size))
    return StateObservableSnapshot(
        n_up_site=np.asarray(n_up, dtype=float),
        n_dn_site=np.asarray(n_dn, dtype=float),
        n_site=n_site,
        doublon=float(doublon_total),
        staggered=float(staggered),
    )


def _drive_aligned_density_label(*, pattern: str) -> str:
    return f"drive_aligned_density(pattern={str(pattern)})"


"""
psi(theta, 0_drive) = psi(theta) while adding a drive-aligned density tangent.
"""
def _augment_replay_context_with_drive_aligned_density(
    replay_context: ReplayScaffoldContext,
    *,
    best_theta: np.ndarray | Sequence[float],
    drive_config: ControllerDriveConfig | None,
    num_qubits: int,
) -> tuple[ReplayScaffoldContext, np.ndarray, bool, str | None]:
    best_theta_arr = np.asarray(best_theta, dtype=float).reshape(-1)
    if drive_config is None or not bool(drive_config.enabled):
        return replay_context, best_theta_arr, False, None
    if abs(float(drive_config.drive_A)) <= 1.0e-12:
        return replay_context, best_theta_arr, False, None
    # Some unit-test toy contexts use a reduced single-qubit register to exercise
    # drive timing logic. Those fixtures do not carry the HH spin-orbital register
    # needed to synthesize the staggered density generator, so skip augmentation.
    if int(num_qubits) < (2 * int(drive_config.n_sites)):
        return replay_context, best_theta_arr, False, None

    label = _drive_aligned_density_label(pattern=str(drive_config.drive_pattern))
    replay_term_labels = {str(term.label) for term in replay_context.replay_terms}
    if label in replay_term_labels:
        return replay_context, best_theta_arr, True, str(label)

    custom_weights = (
        None
        if drive_config.drive_custom_weights is None
        else [float(x) for x in drive_config.drive_custom_weights]
    )
    weights = default_spatial_weights(
        int(drive_config.n_sites),
        mode=str(drive_config.drive_pattern),
        custom=custom_weights,
    )
    drive_poly = build_hubbard_holstein_drive(
        dims=int(drive_config.n_sites),
        v_t=[float(x) for x in np.asarray(weights, dtype=float).tolist()],
        v0=[0.0] * int(drive_config.n_sites),
        repr_mode="JW",
        indexing=str(drive_config.ordering),
        nq_override=int(num_qubits),
    )
    pool_match = next(
        (term for term in replay_context.family_pool if str(term.label) == str(label)),
        None,
    )
    extra_term = (
        pool_match
        if pool_match is not None
        else AnsatzTerm(label=str(label), polynomial=drive_poly)
    )

    old_layout = replay_context.base_layout
    new_replay_terms = tuple(replay_context.replay_terms) + (extra_term,)
    new_layout = build_parameter_layout(
        list(new_replay_terms),
        ignore_identity=bool(old_layout.ignore_identity),
        coefficient_tolerance=float(old_layout.coefficient_tolerance),
        sort_terms=(str(old_layout.term_order).strip().lower() == "sorted"),
    )
    old_blocks = tuple(old_layout.blocks)
    new_blocks = tuple(new_layout.blocks)
    if new_blocks[: len(old_blocks)] != old_blocks:
        raise ValueError("Drive-aligned density augmentation changed replay runtime layout prefix.")

    runtime_delta = int(new_layout.runtime_parameter_count) - int(old_layout.runtime_parameter_count)
    if runtime_delta < 0:
        raise ValueError("Drive-aligned density augmentation reduced runtime parameter count unexpectedly.")
    logical_delta = int(new_layout.logical_parameter_count) - int(old_layout.logical_parameter_count)
    if logical_delta != 1:
        raise ValueError(
            f"Drive-aligned density augmentation expected one logical block; got delta={logical_delta}."
        )

    new_family_pool = (
        tuple(replay_context.family_pool)
        if pool_match is not None
        else tuple(replay_context.family_pool) + (extra_term,)
    )
    new_best_theta = np.concatenate(
        [
            best_theta_arr,
            np.zeros(runtime_delta, dtype=float),
        ]
    )
    new_theta_runtime = np.concatenate(
        [
            np.asarray(replay_context.adapt_theta_runtime, dtype=float).reshape(-1),
            np.zeros(runtime_delta, dtype=float),
        ]
    )
    new_theta_logical = np.concatenate(
        [
            np.asarray(replay_context.adapt_theta_logical, dtype=float).reshape(-1),
            np.zeros(1, dtype=float),
        ]
    )
    pool_meta = dict(replay_context.pool_meta)
    pool_meta["drive_generator_mode"] = "aligned_density"
    pool_meta["drive_aligned_density_active"] = True
    family_info = dict(replay_context.family_info)
    family_info["drive_generator_mode"] = "aligned_density"
    family_info["drive_aligned_density_label"] = str(label)
    replay_context_aug = replace(
        replay_context,
        family_info=family_info,
        family_pool=tuple(new_family_pool),
        pool_meta=pool_meta,
        replay_terms=tuple(new_replay_terms),
        base_layout=new_layout,
        adapt_theta_runtime=np.asarray(new_theta_runtime, dtype=float).reshape(-1),
        adapt_theta_logical=np.asarray(new_theta_logical, dtype=float).reshape(-1),
        adapt_depth=int(len(new_replay_terms)),
        family_terms_count=int(len(new_family_pool)),
    )
    return replay_context_aug, np.asarray(new_best_theta, dtype=float).reshape(-1), True, str(label)


MotionSchedulerTelemetry = _realtime_motion.MotionSchedulerTelemetry

def _carrier_to_term(carrier: RuntimeTermCarrier) -> AnsatzTerm:
    return AnsatzTerm(label=str(carrier.label), polynomial=carrier.polynomial)


def _layout_from_carriers(
    carriers: Sequence[RuntimeTermCarrier],
    *,
    template: AnsatzParameterLayout,
) -> AnsatzParameterLayout:
    runtime_start = 0
    blocks: list[GeneratorParameterBlock] = []
    for logical_index, carrier in enumerate(carriers):
        blocks.append(
            GeneratorParameterBlock(
                candidate_label=str(carrier.label),
                logical_index=int(logical_index),
                runtime_start=int(runtime_start),
                terms=tuple(carrier.runtime_specs),
            )
        )
        runtime_start += int(len(carrier.runtime_specs))
    return AnsatzParameterLayout(
        mode=str(template.mode),
        term_order=str(template.term_order),
        ignore_identity=bool(template.ignore_identity),
        coefficient_tolerance=float(template.coefficient_tolerance),
        blocks=tuple(blocks),
    )


def _build_candidate_carrier(
    term: AnsatzTerm,
    *,
    logical_index: int,
    unique_label: str,
    template_layout: AnsatzParameterLayout,
    candidate_pool_index: int,
) -> RuntimeTermCarrier:
    block_layout = build_parameter_layout(
        [term],
        ignore_identity=bool(template_layout.ignore_identity),
        coefficient_tolerance=float(template_layout.coefficient_tolerance),
        sort_terms=(str(template_layout.term_order).strip().lower() == "sorted"),
    )
    block = block_layout.blocks[0] if block_layout.blocks else GeneratorParameterBlock(
        candidate_label=str(unique_label),
        logical_index=int(logical_index),
        runtime_start=0,
        terms=tuple(),
    )
    return RuntimeTermCarrier(
        label=str(unique_label),
        source_label=str(term.label),
        polynomial=term.polynomial,
        runtime_specs=tuple(block.terms),
        repetition_index=None,
        source_logical_index=None,
        candidate_pool_index=int(candidate_pool_index),
    )


def _build_replay_runtime_terms(
    replay_context: ReplayScaffoldContext,
    *,
    reps: int,
) -> tuple[list[RuntimeTermCarrier], AnsatzParameterLayout]:
    carriers: list[RuntimeTermCarrier] = []
    for rep_idx in range(int(reps)):
        for logical_index, (term, block) in enumerate(
            zip(replay_context.replay_terms, replay_context.base_layout.blocks)
        ):
            carriers.append(
                RuntimeTermCarrier(
                    label=f"{block.candidate_label}__r{rep_idx}",
                    source_label=str(block.candidate_label),
                    polynomial=term.polynomial,
                    runtime_specs=tuple(block.terms),
                    repetition_index=int(rep_idx),
                    source_logical_index=int(logical_index),
                    candidate_pool_index=None,
                )
            )
    layout = _layout_from_carriers(carriers, template=replay_context.base_layout)
    return carriers, layout


def _insert_theta_block(theta: np.ndarray, *, runtime_position: int, width: int) -> np.ndarray:
    arr = np.asarray(theta, dtype=float).reshape(-1)
    return np.concatenate(
        [arr[: int(runtime_position)], np.zeros(int(width), dtype=float), arr[int(runtime_position) :]]
    )


def _delete_theta_block(
    theta: np.ndarray | Sequence[float],
    *,
    runtime_start: int,
    runtime_stop: int,
) -> np.ndarray:
    arr = np.asarray(theta, dtype=float).reshape(-1)
    start = max(0, int(runtime_start))
    stop = max(start, min(int(runtime_stop), int(arr.size)))
    return np.concatenate([arr[:start], arr[stop:]])


def _overlap_l2(lhs: np.ndarray, rhs: np.ndarray | None) -> float | None:
    if rhs is None:
        return None
    lhs_arr = np.asarray(lhs, dtype=float).reshape(-1)
    rhs_arr = np.asarray(rhs, dtype=float).reshape(-1)
    overlap = min(int(lhs_arr.size), int(rhs_arr.size))
    total = 0.0
    if overlap > 0:
        total += float(np.vdot(lhs_arr[:overlap] - rhs_arr[:overlap], lhs_arr[:overlap] - rhs_arr[:overlap]).real)
    if int(lhs_arr.size) > overlap:
        tail = lhs_arr[overlap:]
        total += float(np.vdot(tail, tail).real)
    if int(rhs_arr.size) > overlap:
        tail = rhs_arr[overlap:]
        total += float(np.vdot(tail, tail).real)
    return float(np.sqrt(max(total, 0.0)))


_align_theta_vectors = _realtime_motion.align_theta_vectors
_cosine_similarity = _realtime_motion.cosine_similarity


class RealtimeCheckpointController:
    """Exact/oracle horizon-1 stay/append/prune adaptive checkpoint controller."""

    def __init__(
        self,
        *,
        cfg: RealtimeCheckpointConfig,
        replay_context: ReplayScaffoldContext,
        h_poly: Any,
        hmat: np.ndarray | None,
        psi_initial: np.ndarray,
        best_theta: Sequence[float],
        allow_repeats: bool,
        t_final: float,
        num_times: int,
        drive_config: ControllerDriveConfig | None = None,
        oracle_base_config: Any | None = None,
        wallclock_cap_s: int | None = None,
        progress_path: str | Path | None = None,
        partial_payload_path: str | Path | None = None,
        progress_every_s: float = 5.0,
        exact_reference_cache: MutableMapping[str, object] | None = None,
        resolved_problem: Any | None = None,
        strict_qpu_faithful: bool = False,
        strict_qpu_hh: bool | None = None,
    ) -> None:
        cfg = replace(
            cfg,
            mode=normalize_realtime_controller_mode(getattr(cfg, "mode", "off")),
            reference_mode=normalize_reference_mode(
                getattr(cfg, "reference_mode", "off")
            ),
            high_miss_no_admit_policy=normalize_high_miss_no_admit_policy(
                getattr(cfg, "high_miss_no_admit_policy", None)
            ),
            integrator_policy=str(getattr(cfg, "integrator_policy", "euler")).strip().lower(),
        )
        validate_controller_tiers_mean_only(cfg.tiers)
        self.cfg = cfg
        self.h_poly = h_poly
        self.resolved_problem = resolved_problem
        self._family_key = _controller_family_key(
            resolved_problem=resolved_problem,
            replay_context=replay_context,
        )
        strict_requested = bool(strict_qpu_faithful) or bool(strict_qpu_hh)
        self.strict_qpu_faithful = bool(strict_requested)
        self.strict_qpu_hh = bool(self.strict_qpu_faithful and self._family_key == "hh")
        if self.strict_qpu_faithful and hmat is not None:
            raise ValueError(
                "strict_qpu_faithful forbids dense hmat (legacy strict_qpu_hh forbids dense hmat)"
            )
        if hmat is None:
            if not self.strict_qpu_faithful:
                raise ValueError("hmat is required unless strict_qpu_faithful=True")
            self.hmat = None
        else:
            self.hmat = np.asarray(hmat, dtype=complex)
        self.psi_initial = np.asarray(psi_initial, dtype=complex).reshape(-1)
        self._num_qubits = int(round(np.log2(int(self.psi_initial.size))))
        self.allow_repeats = bool(allow_repeats)
        self.times = np.linspace(0.0, float(t_final), int(num_times), dtype=float)
        self._pauli_action_cache: dict[str, Any] = {}
        self._compiled_poly_cache: dict[str, Any] = {}
        self._drive_config = (
            None
            if drive_config is None or not bool(drive_config.enabled)
            else ControllerDriveConfig(
                enabled=True,
                n_sites=int(drive_config.n_sites),
                ordering=str(drive_config.ordering),
                drive_A=float(drive_config.drive_A),
                drive_omega=float(drive_config.drive_omega),
                drive_tbar=float(drive_config.drive_tbar),
                drive_phi=float(drive_config.drive_phi),
                drive_pattern=str(drive_config.drive_pattern),
                drive_custom_weights=(
                    None
                    if drive_config.drive_custom_weights is None
                    else tuple(float(x) for x in drive_config.drive_custom_weights)
                ),
                drive_include_identity=bool(drive_config.drive_include_identity),
                drive_time_sampling=str(drive_config.drive_time_sampling),
                drive_t0=float(drive_config.drive_t0),
                exact_steps_multiplier=int(drive_config.exact_steps_multiplier),
            )
        )
        self._drive_coeff_provider_exyz = None
        self._drive_model: RealtimeDriveModel | None = None
        self._drive_profile: dict[str, Any] | None = None
        replay_context_local = replay_context
        best_theta_arr = np.asarray(best_theta, dtype=float).reshape(-1)
        self._drive_aligned_density_active = False
        self._drive_aligned_density_label: str | None = None
        self._drive_config = _resolve_drive_config_for_register(
            self._drive_config,
            num_qubits=int(self._num_qubits),
            family_key=str(self._family_key),
        )
        if self._drive_config is not None and self._family_key != "hh":
            adapter = adapter_for_family_key(self._family_key)
            if adapter.drive_operator_kind is None:
                raise ValueError(
                    "Driven neutral realtime currently supports only Hamiltonian families "
                    "whose adapter exposes a drive_operator_kind; "
                    f"got {self._family_key!r}."
                )
            _validate_driven_controller_mode_for_adapter(
                family_key=str(self._family_key),
                cfg=cfg,
                strict_qpu_faithful=bool(self.strict_qpu_faithful),
            )
            if self._family_key == "spin_boson":
                if bool(self._drive_config.drive_include_identity):
                    raise ValueError(
                        "Driven spin_boson neutral realtime currently does not support --drive-include-identity."
                    )
                if (
                    self.resolved_problem is not None
                    and int(getattr(getattr(self.resolved_problem, "request", None), "num_sites", 0)) != 1
                ):
                    raise ValueError(
                        "Driven spin_boson neutral realtime currently requires num_sites == 1."
                    )
            self._drive_model = resolve_realtime_drive_model(
                resolved_problem=self.resolved_problem,
                drive_config=self._drive_config,
            )
            self._drive_profile = dict(self._drive_model.profile_payload)
        drive_augments_replay_context = bool(
            str(cfg.mode) == "exact_v1"
            or str(cfg.mode) == "observable_v1"
            or (str(cfg.mode) == "oracle_v1" and self._family_key == "hh")
        )
        if (
            self._drive_config is not None
            and drive_augments_replay_context
            and (
                self._drive_model is None
                or self._family_key in _DRIVEN_SPINFUL_LATTICE_HAMILTONIAN_FLOW_FAMILIES
            )
        ):
            (
                replay_context_local,
                best_theta_arr,
                self._drive_aligned_density_active,
                self._drive_aligned_density_label,
            ) = _augment_replay_context_with_drive_aligned_density(
                replay_context_local,
                best_theta=best_theta_arr,
                drive_config=self._drive_config,
                num_qubits=int(self._num_qubits),
            )
        self.replay_context = replay_context_local
        self._num_sites = int(
            getattr(
                getattr(self.replay_context, "cfg", None),
                "L",
                (1 if self._drive_config is None else int(self._drive_config.n_sites)),
            )
        )
        self._ordering = str(
            getattr(
                getattr(self.replay_context, "cfg", None),
                "ordering",
                ("blocked" if self._drive_config is None else str(self._drive_config.ordering)),
            )
        )
        self._compiled_h = compile_polynomial_action(
            h_poly,
            tol=1e-12,
            pauli_action_cache=self._pauli_action_cache,
        )
        self._planning_audit = MeasurementCacheAudit(
            nominal_shots_per_group=1,
            plan_version="phase1_qwc_basis_cover_reuse",
            grouping_mode=str(cfg.grouping_mode),
        )
        self._compile_oracle = Phase1CompileCostOracle()
        self._novelty_oracle = Phase2NoveltyOracle()
        self._shortlist_cfg = FullScoreConfig(
            shortlist_fraction=float(cfg.shortlist_fraction),
            shortlist_size=int(cfg.shortlist_size),
        )
        self._append_counter = 0
        self._exact_v1_append_lane_stall_streak = 0
        self._last_exact_v1_postcross_compare_diag: dict[str, Any] | None = None
        self._last_append_no_harm_diagnostics: dict[str, Any] | None = None
        self._trajectory: list[dict[str, Any]] = []
        self._ledger: list[dict[str, Any]] = []
        self._compile_audit_prune_events: list[dict[str, Any]] = []
        self._last_scout_records: list[dict[str, Any]] = []
        self._previous_theta_dot: np.ndarray | None = None
        self._theta_dot_history: list[np.ndarray] = []
        self._high_miss_history: list[bool] = []
        self._high_miss_relative_history: list[bool] = []
        self._previous_append_position: int | None = None
        self._block_birth_checkpoint: dict[str, int] = {}
        self._block_cooldown: dict[str, int] = {}
        self._block_burden: dict[str, float] = {}
        self._block_origin: dict[str, str] = {}
        self._block_motion_history: dict[str, list[float]] = {}
        self._block_fit_history: dict[str, list[float]] = {}
        self._previous_block_theta_snapshot: dict[str, np.ndarray] = {}
        self._run_wallclock_start: float | None = None
        self._wallclock_cap_s = (None if wallclock_cap_s is None else int(wallclock_cap_s))
        self._progress_path = (
            None
            if progress_path in {None, ""}
            else Path(progress_path).resolve()
        )
        self._partial_payload_path = (
            None
            if partial_payload_path in {None, ""}
            else Path(partial_payload_path).resolve()
        )
        self._progress_every_s = max(0.0, float(progress_every_s))
        self._last_progress_emit_wallclock: float | None = None
        self._oracle_base_config = None
        self._oracle_tier_configs: dict[str, Any] = {}
        self._oracle_qop = None
        self._oracle_instances: dict[str, Any] = {}
        self._strict_state_prep_contract: dict[str, Any] = {}
        self._degraded_checkpoint_count = 0
        self._last_candidate_pool_diagnostics: dict[str, Any] = {}
        self._prune_blocker_reason_counts: dict[str, int] = {}
        self._prune_persistence_history: dict[str, list[bool]] = {}
        self._temporal_ledger = TemporalMeasurementLedger()
        analytic_noise_std = float(getattr(cfg, "analytic_noise_std", 0.0))
        if (not np.isfinite(analytic_noise_std)) or analytic_noise_std < 0.0:
            raise ValueError(
                f"analytic_noise_std must be finite and nonnegative; got {analytic_noise_std!r}."
            )
        analytic_noise_seed = getattr(cfg, "analytic_noise_seed", None)
        self._analytic_noise_std = float(analytic_noise_std)
        self._analytic_noise_seed = (
            None if analytic_noise_seed is None else int(analytic_noise_seed)
        )
        self._analytic_noise_model = str(
            getattr(cfg, "analytic_noise_model", "iid_gaussian_legacy")
        ).strip().lower()
        self._analytic_noise_nominal_shots = int(
            getattr(cfg, "analytic_noise_nominal_shots", 2048)
        )
        self._analytic_noise_nominal_repeats = int(
            getattr(cfg, "analytic_noise_nominal_repeats", 1)
        )
        self._analytic_noise_shot_scale = float(
            getattr(cfg, "analytic_noise_shot_scale", 1.0)
        )
        self._analytic_noise_two_qubit_depth_scale = float(
            getattr(cfg, "analytic_noise_two_qubit_depth_scale", 0.0)
        )
        self._analytic_noise_groups_new_scale = float(
            getattr(cfg, "analytic_noise_groups_new_scale", 0.0)
        )
        self._analytic_noise_time_corr = float(
            getattr(cfg, "analytic_noise_time_corr", 0.0)
        )
        self._analytic_noise_bias_energy = float(
            getattr(cfg, "analytic_noise_bias_energy", 0.0)
        )
        self._analytic_noise_bias_doublon = float(
            getattr(cfg, "analytic_noise_bias_doublon", 0.0)
        )
        self._analytic_noise_bias_staggered = float(
            getattr(cfg, "analytic_noise_bias_staggered", 0.0)
        )
        self._analytic_noise_metric_scale = float(
            getattr(cfg, "analytic_noise_metric_scale", 1.0)
        )
        self._analytic_noise_force_psd = bool(
            getattr(cfg, "analytic_noise_force_psd", True)
        )
        self._analytic_noise_rng = np.random.default_rng(self._analytic_noise_seed)
        self._analytic_noise_prev_scalar: float | None = None
        self._analytic_noise_prev_vector: np.ndarray | None = None
        self._analytic_noise_prev_symmetric: np.ndarray | None = None

        mode = str(cfg.mode)
        if mode not in {"off", "exact_v1", "oracle_v1", "observable_v1"}:
            raise ValueError(f"Unsupported realtime checkpoint controller mode {mode!r}.")
        high_miss_policy = normalize_high_miss_no_admit_policy(
            getattr(cfg, "high_miss_no_admit_policy", None)
        )
        if high_miss_policy not in HIGH_MISS_NO_ADMIT_POLICY_CANONICAL:
            raise ValueError(
                "high_miss_no_admit_policy must be one of bounded_stay_advance, repair_stop, or repair_retry."
            )
        retry_max_attempts = int(getattr(cfg, "repair_retry_max_attempts", 2))
        if retry_max_attempts < 0 or retry_max_attempts > 2:
            raise ValueError("repair_retry_max_attempts must satisfy 0 <= value <= 2.")
        retry_mode = str(getattr(cfg, "repair_retry_escalation_mode", "append_budget_then_stabilize_v1")).strip().lower()
        if retry_mode != "append_budget_then_stabilize_v1":
            raise ValueError("repair_retry_escalation_mode must be append_budget_then_stabilize_v1.")
        retry_admission_policy = str(getattr(cfg, "repair_retry_admission_policy", "strict")).strip().lower()
        if retry_admission_policy not in {"strict", "rescue_best_confirmed_append_v1"}:
            raise ValueError(
                "repair_retry_admission_policy must be strict or rescue_best_confirmed_append_v1."
            )
        rescue_min_gain_ratio = float(getattr(cfg, "repair_retry_rescue_min_gain_ratio", 0.0))
        if (not np.isfinite(rescue_min_gain_ratio)) or rescue_min_gain_ratio < 0.0:
            raise ValueError("repair_retry_rescue_min_gain_ratio must be finite and nonnegative.")
        rescue_attempt = str(getattr(cfg, "repair_retry_rescue_attempt", "terminal_attempt_only")).strip().lower()
        if rescue_attempt != "terminal_attempt_only":
            raise ValueError("repair_retry_rescue_attempt must be terminal_attempt_only.")
        self._repair_attempt_state = RepairAttemptState(
            attempt_index=0,
            max_attempts=(retry_max_attempts if high_miss_policy == "repair_retry" else None),
            escalation_kind="base",
        )
        self._repair_effective_cfg = cfg
        miss_abs_threshold = float(getattr(cfg, "miss_abs_threshold", 0.0))
        if (not np.isfinite(miss_abs_threshold)) or miss_abs_threshold < 0.0:
            raise ValueError("miss_abs_threshold must be finite and nonnegative.")
        miss_persistence_window = int(getattr(cfg, "miss_persistence_window", 1))
        miss_persistence_count = int(getattr(cfg, "miss_persistence_count", 1))
        if miss_persistence_window < 1:
            raise ValueError("miss_persistence_window must be >= 1.")
        if miss_persistence_count < 1:
            raise ValueError("miss_persistence_count must be >= 1.")
        if miss_persistence_count > miss_persistence_window:
            raise ValueError("miss_persistence_count must be <= miss_persistence_window.")
        integrator_policy = str(getattr(cfg, "integrator_policy", "euler"))
        if integrator_policy not in {"euler", "rk4", "auto_euler_rk4"}:
            raise ValueError("integrator_policy must be one of euler, rk4, or auto_euler_rk4.")
        for field_name in (
            "integrator_columnarity_threshold",
            "integrator_curvature_threshold",
            "integrator_euler_fs_error_threshold",
            "integrator_condition_max",
            "integrator_euler_min_time_fraction",
        ):
            raw_value = float(getattr(cfg, field_name))
            if (not np.isfinite(raw_value)) or raw_value < 0.0:
                raise ValueError(f"{field_name} must be finite and nonnegative.")
        if float(getattr(cfg, "integrator_euler_min_time_fraction")) > 1.0:
            raise ValueError("integrator_euler_min_time_fraction must be <= 1.")
        integrator_observable_window = int(getattr(cfg, "integrator_euler_observable_window", 16))
        if integrator_observable_window <= 0:
            raise ValueError("integrator_euler_observable_window must be >= 1.")
        for field_name in (
            "integrator_euler_site_span_max",
            "integrator_euler_primary_density_span_max",
            "integrator_euler_energy_span_max",
        ):
            raw_value = getattr(cfg, field_name, None)
            if raw_value is None:
                continue
            raw_float = float(raw_value)
            if (not np.isfinite(raw_float)) or raw_float < 0.0:
                raise ValueError(f"{field_name} must be finite and nonnegative when set.")
        if self._analytic_noise_model not in {"iid_gaussian_legacy", "hybrid_qpu_proxy_v1"}:
            raise ValueError(
                f"Unsupported analytic noise model {self._analytic_noise_model!r}."
            )
        if self._analytic_noise_nominal_shots < 1:
            raise ValueError("analytic_noise_nominal_shots must be >= 1.")
        if self._analytic_noise_nominal_repeats < 1:
            raise ValueError("analytic_noise_nominal_repeats must be >= 1.")
        for field_name, raw_value in (
            ("analytic_noise_shot_scale", self._analytic_noise_shot_scale),
            ("analytic_noise_two_qubit_depth_scale", self._analytic_noise_two_qubit_depth_scale),
            ("analytic_noise_groups_new_scale", self._analytic_noise_groups_new_scale),
            ("analytic_noise_metric_scale", self._analytic_noise_metric_scale),
        ):
            if (not np.isfinite(raw_value)) or raw_value < 0.0:
                raise ValueError(f"{field_name} must be finite and nonnegative.")
        if (not np.isfinite(self._analytic_noise_time_corr)) or not (0.0 <= self._analytic_noise_time_corr < 1.0):
            raise ValueError("analytic_noise_time_corr must lie in [0, 1).")
        for field_name, raw_value in (
            ("analytic_noise_bias_energy", self._analytic_noise_bias_energy),
            ("analytic_noise_bias_doublon", self._analytic_noise_bias_doublon),
            ("analytic_noise_bias_staggered", self._analytic_noise_bias_staggered),
        ):
            if not np.isfinite(raw_value):
                raise ValueError(f"{field_name} must be finite.")
        forecast_guardrail_mode = str(getattr(cfg, "exact_forecast_guardrail_mode", "off"))
        if forecast_guardrail_mode not in {
            "off",
            "dual_metric_v1",
            "d_shape_barrier_v1",
            "fidelity_first_barrier_v1",
        }:
            raise ValueError(
                f"Unsupported exact forecast guardrail mode {forecast_guardrail_mode!r}."
            )
        baseline_proposal_mode = str(
            getattr(cfg, "exact_forecast_baseline_proposal_mode", "norm_locked_blend_v1")
        )
        if baseline_proposal_mode not in {"norm_locked_blend_v1", "anticipatory_drive_basis_v1"}:
            raise ValueError(
                f"Unsupported exact forecast baseline proposal mode {baseline_proposal_mode!r}."
            )
        confirm_score_mode = str(getattr(cfg, "confirm_score_mode", "exact_gain_ratio"))
        if confirm_score_mode not in {"exact_gain_ratio", "compressed_whitened_v1"}:
            raise ValueError(f"Unsupported confirm score mode {confirm_score_mode!r}.")
        prune_mode = str(getattr(cfg, "prune_mode", "off"))
        if prune_mode not in {"off", "exact_local_v1", "schur_projected_shadow_v1"}:
            raise ValueError(f"Unsupported prune mode {prune_mode!r}.")
        prune_appended_origin_target_policy = str(
            getattr(cfg, "prune_appended_origin_target_policy", "append_only")
        )
        if prune_appended_origin_target_policy not in {"append_only", "prefer_append", "bias_only"}:
            raise ValueError(
                "Unsupported prune appended-origin target policy "
                f"{prune_appended_origin_target_policy!r}."
            )
        for field_name in (
            "exact_forecast_fidelity_loss_tol",
            "exact_forecast_abs_energy_error_increase_tol",
            "exact_forecast_total_occupation_error_increase_tol",
            "exact_forecast_energy_slope_weight",
            "exact_forecast_energy_curvature_weight",
            "exact_forecast_tangent_secant_trust_radius",
            "exact_forecast_tangent_secant_signed_energy_lead_limit",
            "confirm_compress_fraction",
            "prune_miss_threshold",
            "prune_stagnation_alpha",
            "prune_stale_score_threshold",
            "prune_loss_threshold",
            "prune_safe_miss_increase_tol",
            "prune_no_harm_score_increase_tol",
            "prune_no_harm_step_residual_ratio_increase_tol",
            "prune_schur_monotonicity_tol",
            "prune_loss_norm_epsilon",
            "prune_differential_miss_tol",
            "prune_projection_trust_radius",
            "prune_projection_state_weight",
            "prune_projection_observable_weight",
            "prune_projection_regularization",
            "prune_ray_distance_tol",
            "prune_shadow_score_tol",
            "prune_shadow_score_increase_tol",
            "prune_shadow_scale_floor",
            "prune_state_jump_l2_tol",
            "prune_state_jump_l2_hard_cap",
            "prune_theta_block_tol",
            "prune_active_block_theta_dot_rel_tol",
            "prune_active_block_theta_dot_abs_tol",
            "prune_active_block_theta_dot_abs_hard_tol",
            "prune_appended_origin_bias_scale",
            "prune_appended_origin_bias_max_factor",
        ):
            raw_value = float(getattr(cfg, field_name))
            if (not np.isfinite(raw_value)) or raw_value < 0.0:
                raise ValueError(
                    f"{field_name} must be finite and nonnegative; got {raw_value!r}."
                )
        forecast_horizon_steps = int(getattr(cfg, "exact_forecast_tracking_horizon_steps", 1))
        if forecast_horizon_steps < 1:
            raise ValueError(
                "exact_forecast_tracking_horizon_steps must be >= 1."
            )
        baseline_step_refine_rounds = int(
            getattr(cfg, "exact_forecast_baseline_step_refine_rounds", 0)
        )
        if baseline_step_refine_rounds < 0:
            raise ValueError(
                "exact_forecast_baseline_step_refine_rounds must be >= 0."
            )
        baseline_blend_weights = tuple(
            float(x) for x in getattr(cfg, "exact_forecast_baseline_blend_weights", ())
        )
        for weight in baseline_blend_weights:
            if (not np.isfinite(weight)) or weight < -1.0 or weight > 1.0:
                raise ValueError(
                    "exact_forecast_baseline_blend_weights must be finite and lie in [-1, 1]."
                )
        baseline_gain_scales = tuple(
            float(x) for x in getattr(cfg, "exact_forecast_baseline_gain_scales", ())
        )
        for scale in baseline_gain_scales:
            if (not np.isfinite(scale)) or scale <= 0.0:
                raise ValueError(
                    "exact_forecast_baseline_gain_scales must be finite and positive."
                )
        forecast_horizon_weights = tuple(
            float(x) for x in getattr(cfg, "exact_forecast_tracking_horizon_weights", ())
        )
        for weight in forecast_horizon_weights:
            if (not np.isfinite(weight)) or weight <= 0.0:
                raise ValueError(
                    f"exact_forecast_tracking_horizon_weights must be finite and positive; got {weight!r}."
                )
        primary_density_mode = str(
            getattr(cfg, "exact_forecast_primary_density_target_mode", "auto")
        ).strip().lower()
        if primary_density_mode not in {"auto", "pair_difference", "staggered"}:
            raise ValueError(
                "exact_forecast_primary_density_target_mode must be one of auto, pair_difference, staggered."
            )
        repeat_reopen_mode = str(
            getattr(cfg, "exact_v1_repeat_reopen_mode", "off")
        ).strip().lower()
        if repeat_reopen_mode not in {"off", "sign_reversal_window"}:
            raise ValueError(
                "exact_v1_repeat_reopen_mode must be one of off, sign_reversal_window."
            )
        for field_name in (
            "exact_v1_density_first_target_gain_floor",
            "exact_v1_below_floor_probe_target_gain_floor",
            "exact_v1_sign_lag_window_activation",
            "exact_v1_postcross_wrong_sign_activation",
            "exact_v1_d_shape_turn_window_abs_activation",
        ):
            raw_value = float(getattr(cfg, field_name, 0.0))
            if (not np.isfinite(raw_value)) or raw_value < 0.0:
                raise ValueError(f"{field_name} must be finite and nonnegative.")
        d_shape_preturn_probe_threshold = int(
            getattr(cfg, "exact_v1_d_shape_outside_turn_below_floor_probe_stall_threshold", 0)
        )
        if d_shape_preturn_probe_threshold < 0:
            raise ValueError(
                "exact_v1_d_shape_outside_turn_below_floor_probe_stall_threshold must be nonnegative."
            )
        measurement_active_window_size = int(
            getattr(cfg, "measurement_active_window_size", 0)
        )
        if measurement_active_window_size < 0:
            raise ValueError("measurement_active_window_size must be nonnegative.")
        sign_lag_window_floor = getattr(
            cfg,
            "exact_v1_sign_lag_window_target_gain_floor",
            None,
        )
        if sign_lag_window_floor is not None:
            raw_value = float(sign_lag_window_floor)
            if (not np.isfinite(raw_value)) or raw_value < 0.0:
                raise ValueError(
                    "exact_v1_sign_lag_window_target_gain_floor must be finite and nonnegative when set."
                )
        postcross_wrong_sign_floor = getattr(
            cfg,
            "exact_v1_postcross_wrong_sign_target_gain_floor",
            None,
        )
        if postcross_wrong_sign_floor is not None:
            raw_value = float(postcross_wrong_sign_floor)
            if (not np.isfinite(raw_value)) or raw_value < 0.0:
                raise ValueError(
                    "exact_v1_postcross_wrong_sign_target_gain_floor must be finite and nonnegative when set."
                )
        primary_density_tracking_weight = getattr(
            cfg,
            "exact_forecast_tracking_primary_density_error_weight",
            None,
        )
        if primary_density_tracking_weight is None:
            primary_density_tracking_weight = getattr(
                cfg,
                "exact_forecast_tracking_staggered_error_weight",
                1.0,
            )
        tracking_term_weights = (
            float(getattr(cfg, "exact_forecast_tracking_fidelity_defect_weight", 1.0)),
            float(primary_density_tracking_weight),
            float(getattr(cfg, "exact_forecast_tracking_staggered_error_weight", 1.0)),
            float(getattr(cfg, "exact_forecast_tracking_doublon_error_weight", 1.0)),
            float(getattr(cfg, "exact_forecast_tracking_site_occupations_error_weight", 1.0)),
            float(getattr(cfg, "exact_forecast_tracking_energy_total_error_weight", 1.0)),
            float(getattr(cfg, "exact_forecast_density_slope_weight", 1.0)),
            float(getattr(cfg, "exact_forecast_density_curvature_weight", 0.0)),
            float(getattr(cfg, "exact_forecast_density_excursion_under_weight", 0.0)),
            float(getattr(cfg, "exact_forecast_density_excursion_over_weight", 0.0)),
            float(getattr(cfg, "exact_forecast_density_sign_lag_weight", 0.0)),
            float(getattr(cfg, "exact_forecast_density_postcross_wrong_sign_weight", 0.0)),
            float(getattr(cfg, "exact_forecast_drive_harmonic_weight", 0.0)),
        )
        for weight in tracking_term_weights:
            if (not np.isfinite(weight)) or weight < 0.0:
                raise ValueError(
                    "exact_forecast_tracking_*_weight values must be finite and nonnegative."
                )
        for field_name in (
            "exact_forecast_primary_density_scale_floor",
            "exact_forecast_density_slope_scale_floor",
            "exact_forecast_doublon_scale_floor",
            "exact_forecast_site_occupations_scale_floor",
            "exact_forecast_energy_total_scale_floor",
        ):
            raw_value = float(getattr(cfg, field_name, 1.0e-6))
            if (not np.isfinite(raw_value)) or raw_value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive.")
        excursion_under_weight = float(
            getattr(cfg, "exact_forecast_energy_excursion_under_weight", 0.0)
        )
        if (not np.isfinite(excursion_under_weight)) or excursion_under_weight < 0.0:
            raise ValueError(
                "exact_forecast_energy_excursion_under_weight must be finite and nonnegative."
            )
        excursion_over_weight = float(
            getattr(cfg, "exact_forecast_energy_excursion_over_weight", 0.0)
        )
        if (not np.isfinite(excursion_over_weight)) or excursion_over_weight < 0.0:
            raise ValueError(
                "exact_forecast_energy_excursion_over_weight must be finite and nonnegative."
            )
        excursion_rel_tolerance = float(
            getattr(cfg, "exact_forecast_energy_excursion_rel_tolerance", 0.0)
        )
        if (not np.isfinite(excursion_rel_tolerance)) or excursion_rel_tolerance < 0.0:
            raise ValueError(
                "exact_forecast_energy_excursion_rel_tolerance must be finite and nonnegative."
            )
        if forecast_horizon_weights and len(forecast_horizon_weights) != forecast_horizon_steps:
            raise ValueError(
                "exact_forecast_tracking_horizon_weights must be empty or match exact_forecast_tracking_horizon_steps."
            )
        progress_window = int(getattr(cfg, "progress_observable_window", 16))
        if int(progress_window) <= 0:
            raise ValueError("progress_observable_window must be >= 1.")
        progress_min_checkpoint = int(getattr(cfg, "progress_early_stop_min_checkpoint", 0))
        if int(progress_min_checkpoint) < 0:
            raise ValueError("progress_early_stop_min_checkpoint must be >= 0.")
        for field_name in (
            "progress_early_stop_site_error_mean_max",
            "progress_early_stop_primary_density_error_mean_max",
            "progress_early_stop_energy_error_mean_max",
            "progress_early_stop_site_span_max",
            "progress_early_stop_primary_density_span_max",
            "progress_early_stop_energy_span_max",
        ):
            raw_value = getattr(cfg, field_name, None)
            if raw_value is None:
                continue
            raw_float = float(raw_value)
            if (not np.isfinite(raw_float)) or raw_float < 0.0:
                raise ValueError(f"{field_name} must be finite and nonnegative when set.")
        if float(cfg.confirm_compress_fraction) > 1.0:
            raise ValueError("confirm_compress_fraction must be <= 1.0.")
        if not (0.0 <= float(cfg.prune_stagnation_alpha) <= 1.0):
            raise ValueError("prune_stagnation_alpha must lie in [0, 1].")
        if float(cfg.prune_stale_score_threshold) > 1.0:
            raise ValueError("prune_stale_score_threshold must be <= 1.0.")
        for field_name in (
            "confirm_compress_min_modes",
            "confirm_compress_max_modes",
            "prune_schur_ladder_local_radius",
            "prune_projection_rounds",
            "prune_projection_max_active_runtime",
            "prune_shadow_horizon_steps",
            "prune_persistence_window",
            "prune_persistence_required",
            "prune_protection_steps",
            "prune_stagnation_window",
            "prune_max_candidates",
            "prune_cooldown_steps",
            "prune_appended_origin_grace_steps",
            "prune_initial_scaffold_grace_steps",
        ):
            raw_value = int(getattr(cfg, field_name))
            if raw_value < 0:
                raise ValueError(f"{field_name} must be nonnegative; got {raw_value!r}.")
        if int(cfg.prune_shadow_horizon_steps) < 1:
            raise ValueError("prune_shadow_horizon_steps must be >= 1.")
        if int(cfg.prune_projection_max_active_runtime) < 1:
            raise ValueError("prune_projection_max_active_runtime must be >= 1.")
        if int(cfg.prune_persistence_window) < 1:
            raise ValueError("prune_persistence_window must be >= 1.")
        if int(cfg.prune_persistence_required) < 1 or int(cfg.prune_persistence_required) > int(cfg.prune_persistence_window):
            raise ValueError("prune_persistence_required must lie in [1, prune_persistence_window].")
        if int(cfg.confirm_compress_max_modes) > 0 and int(cfg.confirm_compress_min_modes) > int(cfg.confirm_compress_max_modes):
            raise ValueError("confirm_compress_min_modes must be <= confirm_compress_max_modes.")
        if str(prune_mode) == "exact_local_v1" and float(cfg.prune_miss_threshold) > float(cfg.miss_threshold):
            raise ValueError("prune_miss_threshold must be <= miss_threshold when prune_mode is active.")
        if mode in {"oracle_v1", "off"} and oracle_base_config is not None:
            validate_controller_oracle_base_config(oracle_base_config)
            from pipelines.exact_bench.noise_oracle_runtime import pauli_poly_to_sparse_pauli_op

            self._oracle_base_config = oracle_base_config
            self._oracle_tier_configs = build_controller_oracle_tier_configs(
                oracle_base_config,
                cfg.tiers,
            )
            self._oracle_qop = pauli_poly_to_sparse_pauli_op(h_poly)
        elif mode == "oracle_v1":
            raise ValueError("checkpoint controller oracle_v1 requires oracle_base_config.")

        self._validate_strict_qpu_faithful_config()

        raw_append_pool_meta = getattr(self.replay_context, "append_pool_meta", None)
        append_pool_meta = dict(
            self.replay_context.pool_meta if raw_append_pool_meta is None else raw_append_pool_meta
        )
        if bool(getattr(cfg, "append_enabled", True)) and not bool(
            append_pool_meta.get("candidate_pool_complete", True)
        ):
            raise ValueError(
                "checkpoint controller requires a complete append candidate family pool when append is enabled; "
                f"incomplete append pool source={append_pool_meta.get('append_pool_source', 'unknown')} "
                f"reason={append_pool_meta.get('incomplete_reason', 'unknown')}."
            )

        self.current_terms, self.current_layout = _build_replay_runtime_terms(
            self.replay_context,
            reps=int(self.replay_context.cfg.reps),
        )
        self.current_theta = np.asarray(best_theta_arr, dtype=float).reshape(-1)
        self.current_executor = self._build_executor(self.current_terms, self.current_layout)
        if int(self.current_theta.size) != int(self.current_layout.runtime_parameter_count):
            raise ValueError(
                f"Replay best_theta length mismatch: {self.current_theta.size} vs expected {self.current_layout.runtime_parameter_count}."
            )
        self._candidate_pool_terms()
        state_prep_kind = str(
            getattr(self, "_strict_state_prep_contract", {}).get("state_prep_kind", "")
        )
        if bool(self.strict_qpu_hh) and state_prep_kind == "computational_basis_ansatz_input":
            contract = dict(getattr(self, "_strict_state_prep_contract", {}))
            contract["prepared_state_reconstruction_skipped"] = True
            contract["prepared_state_reconstruction_skip_reason"] = "strict_qpu_hh_constructor_no_statevector_prepare"
            self._strict_state_prep_contract = contract
        else:
            self._validate_prepared_state_reconstruction()

        for carrier in self.current_terms:
            self._planning_audit.commit(planning_group_keys_for_term(_carrier_to_term(carrier)))
        self._initialize_prune_state()
        self._initialize_drive_runtime()

    def _initialize_drive_runtime(self) -> None:
        if self._drive_config is None:
            self._drive_coeff_provider_exyz = None
            self._drive_profile = None
            return
        if self._drive_model is not None:
            self._drive_coeff_provider_exyz = None
            self._drive_profile = dict(self._drive_model.profile_payload)
            return
        drive = build_gaussian_sinusoid_density_drive(
            n_sites=int(self._drive_config.n_sites),
            nq_total=int(self._num_qubits),
            indexing=str(self._drive_config.ordering),
            A=float(self._drive_config.drive_A),
            omega=float(self._drive_config.drive_omega),
            tbar=float(self._drive_config.drive_tbar),
            phi=float(self._drive_config.drive_phi),
            pattern_mode=str(self._drive_config.drive_pattern),
            custom_weights=(
                None
                if self._drive_config.drive_custom_weights is None
                else [float(x) for x in self._drive_config.drive_custom_weights]
            ),
            include_identity=bool(self._drive_config.drive_include_identity),
            coeff_tol=0.0,
        )
        self._drive_coeff_provider_exyz = drive.coeff_map_exyz
        self._drive_profile = {
            "A": float(self._drive_config.drive_A),
            "omega": float(self._drive_config.drive_omega),
            "tbar": float(self._drive_config.drive_tbar),
            "phi": float(self._drive_config.drive_phi),
            "pattern": str(self._drive_config.drive_pattern),
            "custom_weights": (
                None
                if self._drive_config.drive_custom_weights is None
                else [float(x) for x in self._drive_config.drive_custom_weights]
            ),
            "include_identity": bool(self._drive_config.drive_include_identity),
            "time_sampling": str(self._drive_config.drive_time_sampling),
            "t0": float(self._drive_config.drive_t0),
        }

    def _reference_mode(self) -> str:
        return str(getattr(self.cfg, "reference_mode", "off"))

    def _reference_enabled(self) -> bool:
        return False

    @staticmethod
    def _strict_config_mode(value: Any, *, default: str) -> str:
        if isinstance(value, Mapping):
            raw = value.get("mode", value.get("name", default))
        else:
            raw = default if value in {None, ""} else value
        text = str(raw).strip().lower()
        return str(default) if text == "" else text

    def _strict_state_prep_config_contract(self) -> tuple[dict[str, Any], list[str]]:
        psi_ref = np.asarray(self.replay_context.psi_ref, dtype=complex).reshape(-1)
        psi_initial = np.asarray(self.psi_initial, dtype=complex).reshape(-1)
        errors: list[str] = []
        errors.extend(
            f"strict_qpu_faithful state prep invalid: {reason}"
            for reason in _statevector_validation_errors(psi_ref, label="ansatz input state")
        )
        errors.extend(
            f"strict_qpu_faithful state prep invalid: {reason}"
            for reason in _statevector_validation_errors(psi_initial, label="prepared initial state")
        )
        if psi_ref.size != psi_initial.size:
            errors.append(
                "strict_qpu_faithful state prep invalid: ansatz input and prepared initial "
                f"state dimensions differ ({psi_ref.size} vs {psi_initial.size})"
            )

        reference_is_basis = _is_computational_basis_statevector(psi_ref)
        metadata = _strict_state_prep_metadata_from_replay_context(self.replay_context)
        contract: dict[str, Any] = {
            "version": "strict_state_prep_v1",
            "state_prep_role": "prepared_seed_state_only",
            "reference_state_is_computational_basis": bool(reference_is_basis),
            "non_basis_ansatz_input_allowed": False,
            "exact_target_or_reference_trajectory": False,
            "feeds_controller_decisions": "prepared_ansatz_observables_only",
        }
        if metadata:
            contract.update(dict(metadata))
            contract["reference_state_is_computational_basis"] = bool(reference_is_basis)
            contract["feeds_controller_decisions"] = "prepared_ansatz_observables_only"

        if reference_is_basis:
            contract["state_prep_kind"] = "computational_basis_ansatz_input"
            contract["non_basis_ansatz_input_allowed"] = False
            return contract, errors

        if not metadata:
            errors.append(
                "strict_qpu_faithful non-basis state prep requires seed/artifact "
                "strict_state_prep_contract metadata"
            )
        else:
            source_errors = _strict_state_prep_source_errors(metadata)
            errors.extend(str(item) for item in source_errors)
            ansatz = metadata.get("ansatz_input_state", {})
            initial = metadata.get("initial_state", {})
            if not isinstance(ansatz, Mapping):
                ansatz = {}
            if not isinstance(initial, Mapping):
                initial = {}
            expected_ref_digest = ansatz.get("state_sha256", None)
            expected_initial_digest = initial.get("state_sha256", None)
            actual_ref_digest = _statevector_sha256(psi_ref)
            actual_initial_digest = _statevector_sha256(psi_initial)
            if expected_ref_digest in {None, ""}:
                errors.append("strict_qpu_faithful non-basis state prep requires ansatz_input_state.state_sha256")
            elif str(expected_ref_digest) != str(actual_ref_digest):
                errors.append("strict_qpu_faithful ansatz_input_state state digest mismatch")
            if expected_initial_digest in {None, ""}:
                errors.append("strict_qpu_faithful non-basis state prep requires initial_state.state_sha256")
            elif str(expected_initial_digest) != str(actual_initial_digest):
                errors.append("strict_qpu_faithful initial_state state digest mismatch")
            ref_spec = getattr(getattr(self, "resolved_problem", None), "reference_state", None)
            build_trusted_ref = getattr(ref_spec, "build_state", None)
            if not callable(build_trusted_ref):
                errors.append(
                    "strict_qpu_faithful non-basis state prep requires "
                    "resolved_problem.reference_state.build_state"
                )
            else:
                try:
                    trusted_ref = np.asarray(build_trusted_ref(), dtype=complex).reshape(-1)
                    if trusted_ref.size != psi_ref.size:
                        errors.append(
                            "strict_qpu_faithful non-basis ansatz input dimension does not "
                            "match resolved_problem.reference_state"
                        )
                    else:
                        trusted_error = float(np.linalg.norm(trusted_ref - psi_ref))
                        contract["trusted_reference_state_match_error"] = float(trusted_error)
                        if trusted_error > float(getattr(self.cfg, "reconstruction_tol", 1.0e-8)):
                            errors.append(
                                "strict_qpu_faithful non-basis ansatz input must match "
                                "resolved_problem.reference_state"
                            )
                except Exception as exc:
                    errors.append(
                        "strict_qpu_faithful non-basis state prep could not validate "
                        f"resolved_problem.reference_state: {type(exc).__name__}: {exc}"
                    )
        if not errors:
            contract["state_prep_kind"] = "non_basis_seed_ansatz_input"
            contract["non_basis_ansatz_input_allowed"] = True
        return contract, errors

    def _validate_prepared_state_reconstruction(self) -> float:
        psi_reconstructed = self.current_executor.prepare_state(
            self.current_theta,
            np.asarray(self.replay_context.psi_ref, dtype=complex).reshape(-1),
        )
        reconstruction_error = float(np.linalg.norm(psi_reconstructed - self.psi_initial))
        tol = float(getattr(self.cfg, "reconstruction_tol", 1.0e-8))
        if reconstruction_error > tol:
            if bool(self.strict_qpu_faithful):
                raise ValueError(
                    "strict_qpu_faithful state-prep contract requires psi_initial to "
                    "match the prepared ansatz/circuit state from the seed: "
                    f"||psi_reconstructed - psi_initial||={reconstruction_error:.3e} > {tol:.3e}."
                )
            raise ValueError(
                f"Replay reconstruction mismatch: ||psi_reconstructed - psi_initial||={reconstruction_error:.3e} > {tol:.3e}."
            )
        if bool(self.strict_qpu_faithful):
            contract = dict(getattr(self, "_strict_state_prep_contract", {}))
            contract["prepared_state_reconstruction_error"] = float(reconstruction_error)
            contract["prepared_state_reconstruction_tol"] = float(tol)
            contract["prepared_state_reconstruction_passed"] = True
            self._strict_state_prep_contract = contract
        return float(reconstruction_error)

    def _validate_strict_qpu_faithful_config(self) -> None:
        if not bool(getattr(self, "strict_qpu_faithful", False)):
            return
        cfg = self.cfg
        errors: list[str] = []

        def _require(condition: bool, reason: str) -> None:
            if not bool(condition):
                errors.append(str(reason))

        mode = str(cfg.mode)
        _require(
            mode in QPU_FAITHFUL_CONTROLLER_MODES,
            "strict_qpu_faithful requires mode=observable_v1 or oracle_v1",
        )
        _require(
            str(cfg.reference_mode) == "off",
            "strict_qpu_faithful requires controller exact inputs off (reference_mode=off)",
        )
        state_prep_contract, state_prep_errors = self._strict_state_prep_config_contract()
        self._strict_state_prep_contract = dict(state_prep_contract)
        errors.extend(str(item) for item in state_prep_errors)
        if mode == "oracle_v1":
            _require(
                self._oracle_base_config is not None,
                "strict_qpu_faithful oracle_v1 requires oracle_base_config",
            )
        if self._oracle_base_config is not None:
            oracle_cfg = self._oracle_base_config
            _require(
                str(getattr(oracle_cfg, "noise_mode", "")).strip().lower()
                in {"ideal", "shots"},
                "strict_qpu_faithful requires oracle noise_mode=ideal or shots",
            )
            _require(
                not bool(getattr(oracle_cfg, "use_fake_backend", False)),
                "strict_qpu_faithful forbids fake backends",
            )
            _require(
                not bool(getattr(oracle_cfg, "allow_aer_fallback", False)),
                "strict_qpu_faithful forbids Aer fallback",
            )
            _require(
                self._strict_config_mode(
                    getattr(oracle_cfg, "mitigation", "none"),
                    default="none",
                )
                in {"none", "off"},
                "strict_qpu_faithful forbids mitigation",
            )
            _require(
                self._strict_config_mode(
                    getattr(oracle_cfg, "symmetry_mitigation", "off"),
                    default="off",
                )
                == "off",
                "strict_qpu_faithful forbids symmetry mitigation",
            )
            _require(
                str(getattr(oracle_cfg, "execution_surface", "expectation_v1")).strip().lower()
                == "expectation_v1",
                "strict_qpu_faithful forbids raw/runtime execution surfaces",
            )
            _require(
                str(getattr(oracle_cfg, "raw_transport", "auto")).strip().lower()
                in {"auto", "none", "off"},
                "strict_qpu_faithful forbids raw transport",
            )
            _require(
                not bool(getattr(oracle_cfg, "raw_store_memory", False)),
                "strict_qpu_faithful forbids raw memory storage",
            )
            _require(
                getattr(oracle_cfg, "raw_artifact_path", None) in {None, ""},
                "strict_qpu_faithful forbids raw artifact paths",
            )
        _require(
            str(getattr(cfg, "integrator_policy", "euler"))
            in {"euler", "rk4", "auto_euler_rk4"},
            "strict_qpu_faithful requires integrator_policy in {euler,rk4,auto_euler_rk4}",
        )
        if mode == "oracle_v1":
            _require(
                str(getattr(cfg, "prune_mode", "off")) == "off",
                "strict_qpu_faithful oracle_v1 forbids prune_mode",
            )
        _require(
            normalize_high_miss_no_admit_policy(
                getattr(cfg, "high_miss_no_admit_policy", None)
            )
            == HIGH_MISS_NO_ADMIT_POLICY_DEFAULT,
            "strict_qpu_faithful forbids repair_stop/repair_retry high-miss policies",
        )
        if mode == "oracle_v1":
            _require(
                not bool(getattr(cfg, "append_no_harm_guard_enabled", True)),
                "strict_qpu_faithful oracle_v1 forbids append-no-harm guard",
            )
        _require(
            str(getattr(cfg, "oracle_selection_policy", "measured_gain_commit_veto"))
            == "measured_gain_commit_veto",
            "strict_qpu_faithful requires oracle_selection_policy=measured_gain_commit_veto",
        )
        _require(
            int(getattr(cfg, "exact_forecast_baseline_step_refine_rounds", 0)) == 0,
            "strict_qpu_faithful forbids exact forecast baseline refinement",
        )
        _require(
            not tuple(getattr(cfg, "exact_forecast_baseline_blend_weights", ())),
            "strict_qpu_faithful forbids exact forecast baseline blend weights",
        )
        _require(
            not tuple(getattr(cfg, "exact_forecast_baseline_gain_scales", ())),
            "strict_qpu_faithful forbids exact forecast baseline gain scales",
        )
        _require(
            not bool(getattr(cfg, "exact_forecast_include_tangent_secant_proposal", False)),
            "strict_qpu_faithful forbids exact tangent/secant proposals",
        )
        _require(
            str(getattr(cfg, "exact_forecast_guardrail_mode", "off")) == "off",
            "strict_qpu_faithful forbids exact forecast guardrails",
        )
        _require(
            str(getattr(cfg, "exact_v1_repeat_reopen_mode", "off")) == "off",
            "strict_qpu_faithful forbids exact repeat reopen",
        )
        for field_name in (
            "exact_v1_postcross_compare_diag",
            "exact_v1_below_floor_energy_safe_turn_escape",
            "exact_v1_below_floor_energy_safe_d_shape_escape",
            "exact_v1_d_shape_pre_turn_shadow_bridge",
            "exact_v1_single_surface_commit_law",
        ):
            _require(
                not bool(getattr(cfg, field_name, False)),
                f"strict_qpu_faithful forbids {field_name}",
            )
        _require(
            int(getattr(cfg, "progress_early_stop_min_checkpoint", 0)) == 0,
            "strict_qpu_faithful forbids progress early-stop configuration",
        )
        for field_name in (
            "progress_early_stop_site_error_mean_max",
            "progress_early_stop_primary_density_error_mean_max",
            "progress_early_stop_energy_error_mean_max",
            "progress_early_stop_site_span_max",
            "progress_early_stop_primary_density_span_max",
            "progress_early_stop_energy_span_max",
        ):
            _require(
                getattr(cfg, field_name, None) is None,
                f"strict_qpu_faithful forbids progress early-stop threshold {field_name}",
            )
        if errors:
            raise ValueError(
                "strict_qpu_faithful incompatible controller config: " + "; ".join(errors)
            )

    def _active_cfg(self) -> RealtimeCheckpointConfig:
        return getattr(self, "_repair_effective_cfg", self.cfg)

    def _cfg_float(self, field_name: str) -> float:
        return float(getattr(self._active_cfg(), str(field_name)))

    def _cfg_int(self, field_name: str) -> int:
        return int(getattr(self._active_cfg(), str(field_name)))

    def _logical_insertion_position_cap(self) -> int:
        return max(1, int(getattr(self.current_layout, "logical_parameter_count", 0)) + 1)

    def _repair_retry_effective_config(self, attempt_index: int) -> tuple[RealtimeCheckpointConfig, str | None]:
        attempt = int(attempt_index)
        base = self.cfg
        if attempt <= 0 or str(getattr(base, "high_miss_no_admit_policy", "")) != "repair_retry":
            return base, "base"
        mode = str(getattr(base, "repair_retry_escalation_mode", "append_budget_then_stabilize_v1")).strip().lower()
        if mode != "append_budget_then_stabilize_v1":
            raise ValueError("Unsupported repair retry escalation mode.")
        cap = int(self._logical_insertion_position_cap())
        factor = 2 if attempt == 1 else 4
        shortlist_floor = 0.30 if attempt == 1 else 0.50
        fields: dict[str, Any] = {
            "shortlist_size": max(1, int(getattr(base, "shortlist_size", 4)) * int(factor)),
            "shortlist_fraction": max(float(getattr(base, "shortlist_fraction", 0.15)), float(shortlist_floor)),
            "max_probe_positions": min(cap, max(1, int(getattr(base, "max_probe_positions", 4)) * int(factor))),
        }
        kind = "expand_append_budget"
        if attempt >= 2:
            reg = float(getattr(base, "regularization_lambda", 0.0))
            cand_reg = float(getattr(base, "candidate_regularization_lambda", 0.0))
            pinv = float(getattr(base, "pinv_rcond", 0.0))
            if not np.isfinite(reg) or reg < 0.0:
                reg = 0.0
            if not np.isfinite(cand_reg) or cand_reg < 0.0:
                cand_reg = 0.0
            if not np.isfinite(pinv) or pinv < 0.0:
                pinv = 0.0
            fields.update(
                {
                    "regularization_lambda": float(reg * 10.0),
                    "candidate_regularization_lambda": float(cand_reg * 10.0),
                    "pinv_rcond": float(min(float(pinv * 10.0), 1.0e-6)),
                }
            )
            kind = "expand_append_budget_stabilize_solve"
        return replace(base, **fields), kind

    def _set_repair_attempt_state(self, attempt_index: int) -> RepairAttemptState:
        cfg, kind = self._repair_retry_effective_config(int(attempt_index))
        max_attempts = (
            int(getattr(self.cfg, "repair_retry_max_attempts", 2))
            if str(getattr(self.cfg, "high_miss_no_admit_policy", "")) == "repair_retry"
            else None
        )
        state = RepairAttemptState(
            attempt_index=int(attempt_index),
            max_attempts=max_attempts,
            escalation_kind=kind,
        )
        self._repair_attempt_state = state
        self._repair_effective_cfg = cfg
        self._shortlist_cfg = FullScoreConfig(
            shortlist_fraction=float(getattr(cfg, "shortlist_fraction", 0.15)),
            shortlist_size=int(getattr(cfg, "shortlist_size", 4)),
        )
        return state

    def _repair_noadvance_state_snapshot(self) -> dict[str, Any]:
        return {
            "high_miss_history": list(self._high_miss_history),
            "high_miss_relative_history": list(self._high_miss_relative_history),
            "block_birth_checkpoint": dict(self._block_birth_checkpoint),
            "block_cooldown": dict(self._block_cooldown),
            "block_origin": dict(self._block_origin),
            "block_motion_history": {
                str(label): list(values)
                for label, values in self._block_motion_history.items()
            },
            "block_fit_history": {
                str(label): list(values)
                for label, values in self._block_fit_history.items()
            },
        }

    def _restore_repair_noadvance_state(self, snapshot: Mapping[str, Any]) -> None:
        self._high_miss_history = [bool(x) for x in snapshot.get("high_miss_history", [])]
        self._high_miss_relative_history = [
            bool(x) for x in snapshot.get("high_miss_relative_history", [])
        ]
        self._block_birth_checkpoint = {
            str(label): int(value)
            for label, value in dict(snapshot.get("block_birth_checkpoint", {})).items()
        }
        self._block_cooldown = {
            str(label): int(value)
            for label, value in dict(snapshot.get("block_cooldown", {})).items()
        }
        self._block_origin = {
            str(label): str(value)
            for label, value in dict(snapshot.get("block_origin", {})).items()
        }
        self._block_motion_history = {
            str(label): [float(x) for x in values]
            for label, values in dict(snapshot.get("block_motion_history", {})).items()
        }
        self._block_fit_history = {
            str(label): [float(x) for x in values]
            for label, values in dict(snapshot.get("block_fit_history", {})).items()
        }

    def _progress_observable_window(self) -> int:
        return max(1, int(getattr(self.cfg, "progress_observable_window", 16)))

    def _progress_primary_density_error_value(self, row: Mapping[str, Any]) -> float | None:
        raw_value = row.get("abs_primary_density_error", row.get("abs_staggered_error", None))
        if raw_value is None:
            return None
        value = float(raw_value)
        return None if not np.isfinite(value) else float(value)

    def _progress_observable_metrics(self) -> dict[str, Any]:
        if not self._trajectory:
            return {
                "latest_fidelity_exact": None,
                "latest_abs_energy_total_error": None,
                "latest_site_occupations_abs_error_max": None,
                "latest_abs_primary_density_error": None,
                "progress_observable_window": int(self._progress_observable_window()),
                "rolling_fidelity_exact_mean": None,
                "rolling_abs_energy_total_error_mean": None,
                "rolling_site_occupations_abs_error_max_mean": None,
                "rolling_abs_primary_density_error_mean": None,
                "rolling_energy_total_span": None,
                "rolling_site_occupations_span_max": None,
                "rolling_primary_density_span": None,
            }
        physical_rows = physical_trajectory_rows(self._trajectory, fallback_to_raw=False)
        if not physical_rows:
            return {
                "latest_fidelity_exact": None,
                "latest_abs_energy_total_error": None,
                "latest_site_occupations_abs_error_max": None,
                "latest_abs_primary_density_error": None,
                "progress_observable_window": int(self._progress_observable_window()),
                "rolling_fidelity_exact_mean": None,
                "rolling_abs_energy_total_error_mean": None,
                "rolling_site_occupations_abs_error_max_mean": None,
                "rolling_abs_primary_density_error_mean": None,
                "rolling_energy_total_span": None,
                "rolling_site_occupations_span_max": None,
                "rolling_primary_density_span": None,
            }
        latest = physical_rows[-1]
        window = int(min(self._progress_observable_window(), len(physical_rows)))
        rows = physical_rows[-window:]

        def _rolling_mean(values: list[float]) -> float | None:
            if not values:
                return None
            return float(sum(values) / len(values))

        def _finite_row_value(row: Mapping[str, Any], *keys: str) -> float | None:
            for key in keys:
                raw_value = row.get(key, None)
                if raw_value is None:
                    continue
                try:
                    value = float(raw_value)
                except Exception:
                    continue
                if np.isfinite(value):
                    return float(value)
            return None

        def _span(values: list[float]) -> float | None:
            if not values:
                return None
            return float(max(values) - min(values))

        def _rolling_scalar_span(*keys: str) -> float | None:
            values: list[float] = []
            for row in rows:
                value = _finite_row_value(row, *keys)
                if value is not None:
                    values.append(float(value))
            return _span(values)

        def _rolling_site_occupations_span_max() -> float | None:
            vectors: list[list[float]] = []
            for row in rows:
                raw_values = row.get("site_occupations", None)
                if isinstance(raw_values, (str, bytes)) or not isinstance(raw_values, Sequence):
                    continue
                values: list[float] = []
                valid = True
                for raw_value in raw_values:
                    try:
                        value = float(raw_value)
                    except Exception:
                        valid = False
                        break
                    if not np.isfinite(value):
                        valid = False
                        break
                    values.append(float(value))
                if valid and values:
                    vectors.append(values)
            if not vectors:
                return None
            width = min(len(values) for values in vectors)
            if width <= 0:
                return None
            max_span = 0.0
            for site_index in range(width):
                site_values = [float(values[site_index]) for values in vectors]
                max_span = max(max_span, float(max(site_values) - min(site_values)))
            return float(max_span)

        fidelity_vals = [
            float(row["fidelity_exact"])
            for row in rows
            if row.get("fidelity_exact") is not None and np.isfinite(float(row.get("fidelity_exact")))
        ]
        energy_vals = [
            float(row["abs_energy_total_error"])
            for row in rows
            if row.get("abs_energy_total_error") is not None
            and np.isfinite(float(row.get("abs_energy_total_error")))
        ]
        site_vals = [
            float(row["site_occupations_abs_error_max"])
            for row in rows
            if row.get("site_occupations_abs_error_max") is not None
            and np.isfinite(float(row.get("site_occupations_abs_error_max")))
        ]
        density_vals = []
        for row in rows:
            density_value = self._progress_primary_density_error_value(row)
            if density_value is not None:
                density_vals.append(float(density_value))
        latest_density = self._progress_primary_density_error_value(latest)
        latest_fidelity = latest.get("fidelity_exact", None)
        latest_energy = latest.get("abs_energy_total_error", None)
        latest_site = latest.get("site_occupations_abs_error_max", None)
        rolling_energy_total_span = _rolling_scalar_span("energy_total", "energy_total_controller")
        rolling_primary_density_span = _rolling_scalar_span("primary_density", "staggered")
        rolling_site_occupations_span = _rolling_site_occupations_span_max()
        return {
            "latest_fidelity_exact": (
                None
                if latest_fidelity is None or not np.isfinite(float(latest_fidelity))
                else float(latest_fidelity)
            ),
            "latest_abs_energy_total_error": (
                None
                if latest_energy is None or not np.isfinite(float(latest_energy))
                else float(latest_energy)
            ),
            "latest_site_occupations_abs_error_max": (
                None
                if latest_site is None or not np.isfinite(float(latest_site))
                else float(latest_site)
            ),
            "latest_abs_primary_density_error": (
                None if latest_density is None else float(latest_density)
            ),
            "progress_observable_window": int(window),
            "rolling_fidelity_exact_mean": _rolling_mean(fidelity_vals),
            "rolling_abs_energy_total_error_mean": _rolling_mean(energy_vals),
            "rolling_site_occupations_abs_error_max_mean": _rolling_mean(site_vals),
            "rolling_abs_primary_density_error_mean": _rolling_mean(density_vals),
            "rolling_energy_total_span": rolling_energy_total_span,
            "rolling_site_occupations_span_max": rolling_site_occupations_span,
            "rolling_primary_density_span": rolling_primary_density_span,
        }

    def _progress_early_stop_reason(self, *, checkpoint_index: int) -> str | None:
        min_checkpoint_raw = getattr(self.cfg, "progress_early_stop_min_checkpoint", None)
        if min_checkpoint_raw is None:
            return None
        min_checkpoint = int(min_checkpoint_raw)
        if int(checkpoint_index) < int(min_checkpoint):
            return None
        metrics = self._progress_observable_metrics()
        threshold_specs = (
            (
                "progress_early_stop_site_error_mean_max",
                "rolling_site_occupations_abs_error_max_mean",
                "progress_site_error_mean_exceeds_threshold",
            ),
            (
                "progress_early_stop_primary_density_error_mean_max",
                "rolling_abs_primary_density_error_mean",
                "progress_primary_density_error_mean_exceeds_threshold",
            ),
            (
                "progress_early_stop_energy_error_mean_max",
                "rolling_abs_energy_total_error_mean",
                "progress_energy_error_mean_exceeds_threshold",
            ),
        )
        for cfg_key, metric_key, reason in threshold_specs:
            threshold = getattr(self.cfg, cfg_key, None)
            if threshold is None:
                continue
            metric_value = metrics.get(metric_key, None)
            if metric_value is None:
                continue
            if float(metric_value) > float(threshold):
                return f"{reason}:{float(metric_value):.6g}>{float(threshold):.6g}"
        stable_specs = (
            (
                "progress_early_stop_site_span_max",
                "rolling_site_occupations_span_max",
                "site_span",
            ),
            (
                "progress_early_stop_primary_density_span_max",
                "rolling_primary_density_span",
                "primary_density_span",
            ),
            (
                "progress_early_stop_energy_span_max",
                "rolling_energy_total_span",
                "energy_span",
            ),
        )
        stable_checks: list[str] = []
        for cfg_key, metric_key, label in stable_specs:
            threshold = getattr(self.cfg, cfg_key, None)
            if threshold is None:
                continue
            metric_value = metrics.get(metric_key, None)
            if metric_value is None:
                return None
            if float(metric_value) > float(threshold):
                return None
            stable_checks.append(f"{label}<={float(threshold):.6g}")
        if stable_checks:
            stable_checks.append(f"checkpoint>={int(min_checkpoint)}")
            return "progress_observables_stable:" + ",".join(stable_checks)
        return None

    def _progress_payload(self, *, stage: str, **extra: Any) -> dict[str, Any]:
        elapsed = (
            None
            if self._run_wallclock_start is None
            else float(time.perf_counter() - float(self._run_wallclock_start))
        )
        soft_fallback_counts = high_miss_no_admit_soft_fallback_counts(self._ledger)
        high_miss_no_admit_counts = high_miss_no_admit_diagnostic_counts(self._ledger)
        return _realtime_progress.build_progress_payload(
            mode=str(self.cfg.mode),
            append_count=int(self._append_counter),
            prune_count=int(sum(1 for row in self._ledger if str(row.get("action_kind")) == "prune_coordinate")),
            trajectory_points=int(len(self._trajectory)),
            ledger_entries=int(len(self._ledger)),
            repair_count=int(sum(1 for row in self._ledger if str(row.get("action_kind", "")).startswith("repair_"))),
            repair_retry_attempt_count=int(
                sum(
                    1
                    for row in self._ledger
                    if str(row.get("action_kind")) == "repair_miss"
                    and row.get("repair_max_attempts") is not None
                )
            ),
            logical_block_count=int(self.current_layout.logical_parameter_count),
            runtime_parameter_count=int(self.current_layout.runtime_parameter_count),
            total_checkpoints=int(len(self.times)),
            wallclock_elapsed_s=elapsed,
            observable_metrics=self._progress_observable_metrics(),
            extra={**soft_fallback_counts, **high_miss_no_admit_counts, "stage": str(stage), **extra},
        )

    def _write_progress(self, *, stage: str, force: bool = False, **extra: Any) -> None:
        if self._progress_path is None:
            return
        now = time.perf_counter()
        if (
            not force
            and self._last_progress_emit_wallclock is not None
            and float(self._progress_every_s) > 0.0
            and (float(now) - float(self._last_progress_emit_wallclock)) < float(self._progress_every_s)
        ):
            return
        payload = self._progress_payload(stage=str(stage), **extra)
        _realtime_progress.write_json_atomic(self._progress_path, payload)
        self._last_progress_emit_wallclock = float(now)

    def _write_partial_payload(
        self,
        *,
        status: str = "running",
        stage: str,
        summary: Mapping[str, Any] | None = None,
    ) -> None:
        if self._partial_payload_path is None:
            return
        payload = _realtime_progress.build_partial_payload(
            status=str(status),
            stage=str(stage),
            mode=str(self.cfg.mode),
            trajectory=self._trajectory,
            ledger=self._ledger,
            controller_state=self._controller_state_payload(),
            logical_block_count=int(self.current_layout.logical_parameter_count),
            runtime_parameter_count=int(self.current_layout.runtime_parameter_count),
            summary=summary,
        )
        _realtime_progress.write_json_atomic(self._partial_payload_path, payload)

    def _analytic_noise_enabled(self) -> bool:
        if self._analytic_noise_model == "iid_gaussian_legacy":
            return bool(self._analytic_noise_std > 0.0)
        return bool(
            self._analytic_noise_std > 0.0
            or abs(self._analytic_noise_bias_energy) > 0.0
            or abs(self._analytic_noise_bias_doublon) > 0.0
            or abs(self._analytic_noise_bias_staggered) > 0.0
        )

    def _planning_group_burden(self, summary: BaselineGeometrySummary) -> float:
        planning = dict(getattr(summary, "planning_summary", {}) or {})
        for key in (
            "groups_total",
            "group_count",
            "groups_new",
            "measurement_groups_total",
            "entries",
        ):
            raw = planning.get(key, None)
            if raw is None:
                continue
            value = float(raw)
            if np.isfinite(value) and value > 0.0:
                return float(value)
        return float(max(1, int(getattr(summary, "runtime_parameter_count", 1))))

    def _hybrid_noise_scale(self, summary: BaselineGeometrySummary) -> float:
        group_burden = float(max(1.0, self._planning_group_burden(summary)))
        shots_eff = (
            float(self._analytic_noise_nominal_shots)
            * float(self._analytic_noise_nominal_repeats)
            / float(group_burden)
        )
        shots_eff = float(max(shots_eff, 1.0))
        depth_proxy = float(max(1, int(getattr(summary, "runtime_parameter_count", 1))))
        depth_term = depth_proxy / 32.0
        scale = float(self._analytic_noise_std) * float(self._analytic_noise_metric_scale)
        scale *= float(self._analytic_noise_shot_scale) / float(np.sqrt(shots_eff))
        scale *= 1.0 + float(self._analytic_noise_two_qubit_depth_scale) * float(depth_term)
        scale *= 1.0 + float(self._analytic_noise_groups_new_scale) * float(np.log1p(group_burden))
        return float(max(scale, 0.0))

    def _apply_time_correlation(
        self,
        sample: np.ndarray,
        *,
        previous: np.ndarray | None,
    ) -> np.ndarray:
        corr = float(self._analytic_noise_time_corr)
        if previous is None or corr <= 0.0:
            return np.asarray(sample, dtype=float)
        if previous.shape != sample.shape:
            return np.asarray(sample, dtype=float)
        mixed = corr * np.asarray(previous, dtype=float) + np.sqrt(max(0.0, 1.0 - corr * corr)) * np.asarray(sample, dtype=float)
        return np.asarray(mixed, dtype=float)

    def _force_psd_metric(self, value: np.ndarray) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        sym = 0.5 * (arr + arr.T)
        eigvals, eigvecs = np.linalg.eigh(sym)
        floor = float(max(1.0e-10, self._cfg_float("regularization_lambda")))
        eigvals = np.maximum(np.asarray(eigvals, dtype=float), floor)
        rebuilt = eigvecs @ np.diag(eigvals) @ np.asarray(eigvecs, dtype=float).T
        return np.asarray(0.5 * (rebuilt + rebuilt.T), dtype=float)

    def _hybrid_observable_bias_vector(
        self,
        *,
        psi: np.ndarray,
        energy: float,
        f: np.ndarray,
    ) -> np.ndarray:
        f_arr = np.asarray(f, dtype=float).reshape(-1)
        if f_arr.size == 0:
            return f_arr
        snapshot = self._observable_snapshot(np.asarray(psi, dtype=complex).reshape(-1))
        bias_scalar = (
            float(self._analytic_noise_bias_energy) * float(energy)
            + float(self._analytic_noise_bias_doublon) * float(snapshot["doublon"])
            + float(self._analytic_noise_bias_staggered) * float(snapshot["staggered"])
        )
        if abs(float(bias_scalar)) <= 0.0:
            return np.zeros_like(f_arr, dtype=float)
        direction = np.sign(f_arr)
        direction[np.abs(direction) <= 1.0e-12] = 1.0
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1.0e-12:
            return np.zeros_like(f_arr, dtype=float)
        direction = direction / direction_norm
        return np.asarray(float(bias_scalar) * direction, dtype=float)

    def _add_scalar_gaussian_noise(self, value: float) -> float:
        value_f = float(value)
        if not self._analytic_noise_enabled():
            return value_f
        if self._analytic_noise_model != "iid_gaussian_legacy":
            sample = np.asarray([self._analytic_noise_rng.normal(0.0, 1.0)], dtype=float)
            sample = self._apply_time_correlation(sample, previous=None if self._analytic_noise_prev_scalar is None else np.asarray([self._analytic_noise_prev_scalar], dtype=float))
            self._analytic_noise_prev_scalar = float(sample[0])
            return value_f + float(self._analytic_noise_std) * float(self._analytic_noise_metric_scale) * float(sample[0])
        return value_f + float(
            self._analytic_noise_rng.normal(0.0, float(self._analytic_noise_std))
        )

    def _add_vector_gaussian_noise(self, value: np.ndarray) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        if not self._analytic_noise_enabled():
            return arr
        if self._analytic_noise_model != "iid_gaussian_legacy":
            sample = np.asarray(
                self._analytic_noise_rng.normal(0.0, 1.0, size=arr.shape),
                dtype=float,
            )
            sample = self._apply_time_correlation(sample, previous=self._analytic_noise_prev_vector)
            self._analytic_noise_prev_vector = np.asarray(sample, dtype=float)
            return np.asarray(
                arr + float(self._analytic_noise_std) * float(self._analytic_noise_metric_scale) * sample,
                dtype=float,
            )
        return np.asarray(
            arr
            + self._analytic_noise_rng.normal(
                0.0, float(self._analytic_noise_std), size=arr.shape
            ),
            dtype=float,
        )

    def _add_symmetric_gaussian_noise(self, value: np.ndarray) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        if not self._analytic_noise_enabled():
            return arr
        if self._analytic_noise_model != "iid_gaussian_legacy":
            sample = np.asarray(
                self._analytic_noise_rng.normal(0.0, 1.0, size=arr.shape),
                dtype=float,
            )
            sample = np.triu(sample)
            sample = sample + np.triu(sample, 1).T
            sample = self._apply_time_correlation(sample, previous=self._analytic_noise_prev_symmetric)
            self._analytic_noise_prev_symmetric = np.asarray(sample, dtype=float)
            return np.asarray(
                arr + float(self._analytic_noise_std) * float(self._analytic_noise_metric_scale) * sample,
                dtype=float,
            )
        noise = self._analytic_noise_rng.normal(
            0.0, float(self._analytic_noise_std), size=arr.shape
        )
        noise = np.triu(noise)
        noise = noise + np.triu(noise, 1).T
        return np.asarray(arr + noise, dtype=float)

    def _build_executor(
        self,
        carriers: Sequence[RuntimeTermCarrier],
        layout: AnsatzParameterLayout,
    ) -> CompiledAnsatzExecutor:
        return CompiledAnsatzExecutor(
            [_carrier_to_term(carrier) for carrier in carriers],
            coefficient_tolerance=float(layout.coefficient_tolerance),
            ignore_identity=bool(layout.ignore_identity),
            sort_terms=(str(layout.term_order).strip().lower() == "sorted"),
            pauli_action_cache=self._pauli_action_cache,
            parameterization_mode="per_pauli_term",
            parameterization_layout=layout,
        )

    def _observable_snapshot(self, psi: np.ndarray) -> dict[str, Any]:
        return observable_snapshot_for_state(
            np.asarray(psi, dtype=complex).reshape(-1),
            resolved_problem=self.resolved_problem,
            num_sites=int(max(1, self._num_sites)),
            ordering=str(self._ordering),
            compiled_poly_cache=self._compiled_poly_cache,
            pauli_action_cache=self._pauli_action_cache,
        )

    """
    snapshot_k = benchmark pre-action capture at checkpoint k, owned by the exact-audit helper.
    """
    def exact_v1_pre_action_snapshot(self, *, checkpoint_index: int) -> dict[str, Any]:
        from pipelines.time_dynamics.legacy.checkpoint_exact_audit import (
            build_exact_audit_helper_for_controller,
            exact_v1_pre_action_snapshot as _exact_snapshot,
        )

        return _exact_snapshot(
            self,
            build_exact_audit_helper_for_controller(self),
            checkpoint_index=int(checkpoint_index),
        )

    def _exact_forecast_primary_density_target_mode(self) -> str:
        mode = str(getattr(self.cfg, "exact_forecast_primary_density_target_mode", "auto")).strip().lower()
        if mode == "auto":
            return auto_primary_density_mode(
                resolved_problem=self.resolved_problem,
                num_sites=int(max(1, self._num_sites)),
            )
        return mode

    def _primary_density_value_from_snapshot(self, snapshot: Mapping[str, Any]) -> float:
        return primary_density_value_from_snapshot(
            snapshot,
            resolved_problem=self.resolved_problem,
            num_sites=int(max(1, self._num_sites)),
            requested_mode=self._exact_forecast_primary_density_target_mode(),
        )

    def _exact_step_forecast(
        self,
        *,
        time_stop: float,
        executor: CompiledAnsatzExecutor,
        theta_runtime: np.ndarray | Sequence[float],
    ) -> dict[str, Any]:
        from pipelines.time_dynamics.legacy.checkpoint_exact_audit import (
            build_exact_audit_helper_for_controller,
            exact_step_forecast as _exact_step_forecast,
        )

        return _exact_step_forecast(
            self,
            build_exact_audit_helper_for_controller(self),
            time_stop=float(time_stop),
            executor=executor,
            theta_runtime=theta_runtime,
        )

    def _forecast_score_total(
        self,
        forecast: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    ) -> float:
        if isinstance(forecast, Mapping) and forecast.get("local_projective_score_total") is not None:
            return float(forecast["local_projective_score_total"])
        return float(self._forecast_tracking_score(forecast=forecast))

    def _optional_forecast_metric(
        self,
        forecast: Mapping[str, Any],
        *,
        normalized_key: str | None,
        raw_key: str,
        raw_fallback_key: str | None = None,
    ) -> float | None:
        if normalized_key is not None:
            value = self._finite_float_or_none(forecast.get(normalized_key, None))
            if value is not None:
                return float(value)
        value = self._finite_float_or_none(forecast.get(raw_key, None))
        if value is not None:
            return float(value)
        if raw_fallback_key is not None:
            return self._finite_float_or_none(forecast.get(raw_fallback_key, None))
        return None

    def _weighted_optional_term(self, value: float | None, weight: float | None) -> float:
        if value is None or weight is None:
            return 0.0
        value_f = self._finite_float_or_none(value)
        weight_f = self._finite_float_or_none(weight)
        if value_f is None or weight_f is None or abs(float(weight_f)) <= 0.0:
            return 0.0
        return float(weight_f) * float(value_f)

    def _forecast_has_exact_supervision_signals(
        self,
        forecast: Mapping[str, Any] | None,
    ) -> bool:
        if not isinstance(forecast, Mapping):
            return False
        required_scalar_keys = (
            "fidelity_exact_next",
            "abs_energy_total_error_next",
            "abs_primary_density_error_next",
            "site_occupations_abs_error_max_next",
        )
        for key in required_scalar_keys:
            value = forecast.get(key)
            if value is None:
                return False
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                return False
            if not np.isfinite(value_f):
                return False
        return True

    def _local_projective_forecast_rollout(
        self,
        *,
        checkpoint_index: int | None,
        time_stop: float,
        executor: CompiledAnsatzExecutor,
        layout: AnsatzParameterLayout,
        theta_runtime_start: np.ndarray | Sequence[float],
        theta_dot_step: np.ndarray | Sequence[float],
        planning_audit: MeasurementCacheAudit,
        scaffold_labels: Sequence[str],
        immediate_gain_ratio: float | None = None,
        anchor_summary: BaselineGeometrySummary | None = None,
        anchor_predicted_displacement: float | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
        next_index = (
            max(0, int(checkpoint_index) + 1)
            if checkpoint_index is not None
            else int(np.argmin(np.abs(self.times - float(time_stop))))
        )
        if next_index >= int(len(self.times)):
            raise RuntimeError("local forecast requires a future checkpoint on the controller grid")
        horizon_steps = min(
            self._exact_forecast_tracking_horizon_steps(),
            int(len(self.times) - next_index),
        )
        weights = self._exact_forecast_tracking_horizon_weights(steps=horizon_steps)
        theta_now = np.asarray(theta_runtime_start, dtype=float).reshape(-1)
        theta_dot_now = np.asarray(theta_dot_step, dtype=float).reshape(-1)
        rows: list[dict[str, Any]] = []
        for offset in range(int(horizon_steps)):
            row_index = int(next_index + offset)
            row_time = float(self.times[row_index])
            row_time_stop = (
                None
                if int(row_index) + 1 >= int(len(self.times))
                else float(self.times[int(row_index) + 1])
            )
            step_sample_time = self._projection_sample_time(float(row_time), row_time_stop)
            step_hamiltonian = self._step_hamiltonian_artifacts(float(step_sample_time))
            psi_row = executor.prepare_state(theta_now, self.replay_context.psi_ref)
            checkpoint_ctx = make_checkpoint_context(
                checkpoint_index=int(row_index),
                time_start=float(row_time),
                time_stop=(None if row_time_stop is None else float(row_time_stop)),
                scaffold_labels=[str(label) for label in scaffold_labels],
                theta=theta_now,
                psi=psi_row,
                logical_count=int(layout.logical_parameter_count),
                runtime_count=int(layout.runtime_parameter_count),
                resolved_family=str(self.replay_context.family_info.get("resolved", "unknown")),
                grouping_mode=str(self.cfg.grouping_mode),
                structure_locked=False,
            )
            cache = ExactCheckpointValueCache(
                checkpoint_id=str(checkpoint_ctx.checkpoint_id),
                grouping_mode=str(self.cfg.grouping_mode),
            )
            payload = self._compute_baseline_geometry_for_runtime_state(
                checkpoint_ctx=checkpoint_ctx,
                cache=cache,
                executor=executor,
                layout=layout,
                theta_runtime=theta_now,
                planning_audit=planning_audit,
                step_hamiltonian=step_hamiltonian,
            )
            summary = payload["summary"]
            obs = self._observable_snapshot(np.asarray(payload["psi"], dtype=complex).reshape(-1))
            dt_local = 0.0 if row_time_stop is None else float(row_time_stop - row_time)
            predicted_displacement = self._predicted_displacement(
                dt=float(dt_local),
                baseline=payload,
            )
            variance_anchor = max(float(summary.variance), 1.0e-14)
            rows.append(
                {
                    "forecast_offset": int(offset + 1),
                    "checkpoint_index": int(row_index),
                    "time": float(row_time),
                    "time_stop": (None if row_time_stop is None else float(row_time_stop)),
                    "rho_miss": float(summary.rho_miss),
                    "epsilon_proj_sq": float(summary.epsilon_proj_sq),
                    "epsilon_step_sq": float(summary.epsilon_step_sq),
                    "epsilon_step_ratio": float(summary.epsilon_step_sq / variance_anchor),
                    "condition_number": float(summary.condition_number),
                    "theta_dot_l2": float(summary.theta_dot_l2),
                    "predicted_displacement": float(predicted_displacement),
                    "step_objective_value": float(summary.step_objective_value),
                    "step_gain_ratio": float(summary.step_gain_ratio),
                    "energy_total": float(summary.energy),
                    "primary_density": float(self._primary_density_value_from_snapshot(obs)),
                    "site_occupations": [float(x) for x in obs["site_occupations"]],
                }
            )
            if row_time_stop is None:
                break
            theta_now = np.asarray(
                theta_now + float(dt_local) * np.asarray(payload["theta_dot_step"], dtype=float),
                dtype=float,
            ).reshape(-1)
            theta_dot_now = np.asarray(payload["theta_dot_step"], dtype=float).reshape(-1)
            del theta_dot_now
        if not rows:
            raise RuntimeError("local forecast produced no rows")
        anchor = anchor_summary or rows[0]

        def _anchor_field(name: str, default: float = 0.0) -> float:
            if isinstance(anchor, BaselineGeometrySummary):
                return float(getattr(anchor, name, default))
            if isinstance(anchor, Mapping):
                value = anchor.get(name, default)
            else:
                value = getattr(anchor, name, default)
            return float(default if value is None else value)

        gain_anchor = max(abs(_anchor_field("step_gain_ratio", 0.0)), 1.0e-6)
        rho_anchor = max(_anchor_field("rho_miss", 0.0), 1.0e-6)
        step_ratio_default = _anchor_field("epsilon_step_ratio", 0.0)
        if isinstance(anchor, BaselineGeometrySummary):
            step_ratio_default = float(anchor.epsilon_step_sq / max(anchor.variance, 1.0e-14))
        else:
            epsilon_step_sq = _anchor_field("epsilon_step_sq", step_ratio_default)
            variance_value = max(_anchor_field("variance", 0.0), 1.0e-14)
            if epsilon_step_sq > 0.0 and variance_value > 0.0:
                step_ratio_default = float(epsilon_step_sq / variance_value)
        step_anchor = max(step_ratio_default, 1.0e-6)
        cond_anchor = max(float(np.log1p(_anchor_field("condition_number", 1.0))), 1.0)
        theta_anchor = max(_anchor_field("theta_dot_l2", 0.0), 1.0e-6)
        disp_anchor = max(
            float(anchor_predicted_displacement if anchor_predicted_displacement is not None else rows[0]["predicted_displacement"]),
            1.0e-6,
        )
        score_breakdown = {
            "gain_reward": 0.0,
            "rho_miss_penalty": 0.0,
            "step_residual_penalty": 0.0,
            "condition_penalty": 0.0,
            "theta_velocity_penalty": 0.0,
            "displacement_penalty": 0.0,
        }
        weight_total = max(float(sum(weights)), 1.0e-12)
        for weight, row in zip(weights, rows):
            w = float(weight) / weight_total
            score_breakdown["rho_miss_penalty"] += w * float(row["rho_miss"]) / rho_anchor
            score_breakdown["step_residual_penalty"] += w * float(row["epsilon_step_ratio"]) / step_anchor
            score_breakdown["condition_penalty"] += w * float(np.log1p(float(row["condition_number"]))) / cond_anchor
            score_breakdown["theta_velocity_penalty"] += w * float(row["theta_dot_l2"]) / theta_anchor
            score_breakdown["displacement_penalty"] += w * float(row["predicted_displacement"]) / disp_anchor
        gain_value = float(rows[0]["step_gain_ratio"] if immediate_gain_ratio is None else immediate_gain_ratio)
        score_breakdown["gain_reward"] = float(gain_value / gain_anchor)
        score_total = (
            -float(getattr(self.cfg, "forecast_score_gain_weight", 1.0)) * score_breakdown["gain_reward"]
            + float(getattr(self.cfg, "forecast_score_rho_miss_weight", 1.0)) * score_breakdown["rho_miss_penalty"]
            + float(getattr(self.cfg, "forecast_score_step_residual_weight", 0.5)) * score_breakdown["step_residual_penalty"]
            + float(getattr(self.cfg, "forecast_score_condition_weight", 0.1)) * score_breakdown["condition_penalty"]
            + float(getattr(self.cfg, "forecast_score_theta_velocity_weight", 0.1)) * score_breakdown["theta_velocity_penalty"]
            + float(getattr(self.cfg, "forecast_score_displacement_weight", 0.1)) * score_breakdown["displacement_penalty"]
        )
        forecast = {
            "forecast_mode": "local_projective_v1",
            "local_projective_score_total": float(score_total),
            "tracking_score_horizon": float(score_total),
            "tracking_score_step1": float(score_total),
            "score_breakdown": dict(score_breakdown),
            "horizon_steps_scored": int(len(rows)),
            "horizon_weights_used": [float(x) for x in weights[: len(rows)]],
            "tracking_horizon_steps_scored": int(len(rows)),
            "tracking_horizon_weights_used": [float(x) for x in weights[: len(rows)]],
            "rows": [dict(row) for row in rows],
            "rho_miss_next": float(rows[0]["rho_miss"]),
            "step_gain_ratio_next": float(rows[0]["step_gain_ratio"]),
            "condition_number_next": float(rows[0]["condition_number"]),
            "epsilon_step_ratio_next": float(rows[0]["epsilon_step_ratio"]),
            "step_residual_ratio_next": float(rows[0]["epsilon_step_ratio"]),
            "theta_dot_l2_next": float(rows[0]["theta_dot_l2"]),
            "predicted_displacement_next": float(rows[0]["predicted_displacement"]),
        }
        return forecast, rows, float(score_total)

    def _forecast_first_scalar(
        self,
        forecast: Mapping[str, Any] | None,
        *keys: str,
    ) -> float | None:
        if not isinstance(forecast, Mapping):
            return None
        for key in keys:
            value = self._finite_float_or_none(forecast.get(str(key), None))
            if value is not None:
                return float(value)
        rows = forecast.get("rows", None)
        if isinstance(rows, (list, tuple)) and rows and isinstance(rows[0], Mapping):
            first = rows[0]
            for key in keys:
                value = self._finite_float_or_none(first.get(str(key), None))
                if value is not None:
                    return float(value)
        return None

    def _positive_ratio_or_none(
        self,
        numerator: float | None,
        denominator: float | None,
        *,
        floor: float = 1.0e-12,
    ) -> float | None:
        if numerator is None or denominator is None:
            return None
        num = float(numerator)
        den = float(denominator)
        if not np.isfinite(num) or not np.isfinite(den):
            return None
        floor_f = max(float(floor), 1.0e-300)
        if abs(den) <= floor_f:
            return 1.0 if abs(num) <= floor_f else float("inf")
        return float(num / den)

    def _append_no_harm_exact_logging(
        self,
        *,
        stay_forecast: Mapping[str, Any] | None,
        selected_forecast: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        exact_keys = (
            "fidelity_exact_next",
            "abs_energy_total_error_next",
            "abs_primary_density_error_next",
            "abs_primary_density_slope_error_next",
            "abs_staggered_error_next",
            "abs_doublon_error_next",
            "site_occupations_abs_error_max_next",
        )
        out: dict[str, Any] = {
            "logging_only": True,
            "used_for_veto": False,
        }
        found = False
        for key in exact_keys:
            stay_value = self._forecast_first_scalar(stay_forecast, key)
            selected_value = self._forecast_first_scalar(selected_forecast, key)
            if stay_value is not None:
                out[f"stay_{key}"] = float(stay_value)
                found = True
            if selected_value is not None:
                out[f"selected_{key}"] = float(selected_value)
                found = True
            if stay_value is not None and selected_value is not None:
                out[f"delta_{key}_selected_minus_stay"] = float(
                    float(selected_value) - float(stay_value)
                )
        return dict(out) if found else None

    def _append_no_harm_guard_reason(
        self,
        *,
        stay_forecast: Mapping[str, Any] | None,
        selected_forecast: Mapping[str, Any] | None,
        selected: Mapping[str, Any] | None = None,
        motion: MotionSchedulerTelemetry | None = None,
    ) -> tuple[str | None, dict[str, Any]]:
        exact_logging = self._append_no_harm_exact_logging(
            stay_forecast=stay_forecast,
            selected_forecast=selected_forecast,
        )
        stay_condition = self._forecast_first_scalar(stay_forecast, "condition_number_next", "condition_number")
        selected_condition = self._forecast_first_scalar(
            selected_forecast,
            "condition_number_next",
            "condition_number",
        )
        condition_ratio = self._positive_ratio_or_none(selected_condition, stay_condition)
        stay_rho = self._forecast_first_scalar(stay_forecast, "rho_miss_next", "rho_miss")
        selected_rho = self._forecast_first_scalar(selected_forecast, "rho_miss_next", "rho_miss")
        rho_miss_delta = (
            None if stay_rho is None or selected_rho is None else float(stay_rho - selected_rho)
        )
        stay_step_gain = self._forecast_first_scalar(
            stay_forecast,
            "step_gain_ratio_next",
            "step_gain_ratio",
        )
        selected_step_gain = self._forecast_first_scalar(
            selected_forecast,
            "step_gain_ratio_next",
            "step_gain_ratio",
        )
        step_gain_delta = (
            None
            if stay_step_gain is None or selected_step_gain is None
            else float(selected_step_gain - stay_step_gain)
        )
        stay_step_residual = self._forecast_first_scalar(
            stay_forecast,
            "epsilon_step_ratio_next",
            "step_residual_ratio_next",
            "epsilon_step_ratio",
        )
        selected_step_residual = self._forecast_first_scalar(
            selected_forecast,
            "epsilon_step_ratio_next",
            "step_residual_ratio_next",
            "epsilon_step_ratio",
        )
        step_residual_ratio = self._positive_ratio_or_none(
            selected_step_residual,
            stay_step_residual,
        )
        stay_displacement = self._forecast_first_scalar(
            stay_forecast,
            "predicted_displacement_next",
            "predicted_displacement",
        )
        selected_displacement = self._forecast_first_scalar(
            selected_forecast,
            "predicted_displacement_next",
            "predicted_displacement",
        )
        displacement_ratio = self._positive_ratio_or_none(
            selected_displacement,
            stay_displacement,
        )
        selected_gain_ratio = (
            None
            if selected is None
            else self._finite_float_or_none(selected.get("gain_ratio", None))
        )
        selected_candidate_label = (
            None
            if selected is None or selected.get("candidate_label", None) in {None, ""}
            else str(selected.get("candidate_label"))
        )
        motion_regime = None if motion is None else str(motion.regime)
        motion_direction_reversal = bool(False if motion is None else motion.direction_reversal)
        motion_curvature_sign_flip = bool(False if motion is None else motion.curvature_sign_flip)
        motion_kink_score = None if motion is None else float(motion.kink_score)
        motion_bad = bool(
            motion is not None
            and (
                str(motion.regime) == "kink"
                or bool(motion.direction_reversal)
                or bool(motion.curvature_sign_flip)
                or float(motion.kink_score)
                >= float(getattr(self.cfg, "motion_kink_rate_change_ratio_threshold", 0.50))
            )
        )
        condition_ratio_cap = float(getattr(self.cfg, "append_no_harm_condition_ratio_cap", 1.0))
        displacement_ratio_cap = float(
            getattr(self.cfg, "append_no_harm_displacement_ratio_cap", 1.0)
        )
        condition_abs_floor = float(getattr(self.cfg, "append_no_harm_condition_abs_floor", 1.0))
        kink_min_step_gain_delta = float(
            getattr(self.cfg, "append_no_harm_kink_min_step_gain_delta", 1.0e-3)
        )
        kink_condition_cap = float(
            getattr(self.cfg, "append_no_harm_kink_max_condition_ratio", 1.0)
        )
        kink_displacement_cap = float(
            getattr(self.cfg, "append_no_harm_kink_max_displacement_ratio", 1.0)
        )
        rho_only_min_step_gain_delta = float(
            getattr(self.cfg, "append_no_harm_rho_only_min_step_gain_delta", 1.0e-3)
        )
        rho_only_condition_cap = float(
            getattr(self.cfg, "append_no_harm_rho_only_condition_ratio_cap", 1.5)
        )
        rho_only_step_cap = float(
            getattr(self.cfg, "append_no_harm_rho_only_step_residual_ratio_cap", 1.5)
        )
        rho_only_displacement_cap = float(
            getattr(self.cfg, "append_no_harm_rho_only_displacement_ratio_cap", 1.5)
        )
        gain_threshold = max(0.0, float(getattr(self.cfg, "gain_ratio_threshold", 0.0)))
        candidate_gain_support_floor = max(
            float(kink_min_step_gain_delta),
            2.0 * gain_threshold,
        )
        steady_projective_gain_floor = max(float(kink_min_step_gain_delta), gain_threshold)
        step_gain_support = bool(
            step_gain_delta is not None
            and float(step_gain_delta) >= float(kink_min_step_gain_delta)
        )
        candidate_gain_support = bool(
            selected_gain_ratio is not None
            and float(selected_gain_ratio) >= float(candidate_gain_support_floor)
        )
        stability_support = bool(
            (step_gain_support or candidate_gain_support)
            and (
                condition_ratio is None
                or float(condition_ratio) <= float(kink_condition_cap)
            )
            and (
                displacement_ratio is None
                or float(displacement_ratio) <= float(kink_displacement_cap)
            )
        )
        zero_motion_condition_ratio_cap = max(
            float(condition_ratio_cap) + 1.0e-6,
            float(rho_only_condition_cap),
        )
        projective_append_support_mode = str(getattr(self.cfg, "mode", "")) in {
            "exact_v1",
            "observable_v1",
        }
        zero_motion_projective_support = bool(
            str(getattr(self, "_family_key", "")) in _HAMILTONIAN_FLOW_FAMILIES
            and bool(projective_append_support_mode)
            and not bool(motion_bad)
            and stay_displacement is not None
            and abs(float(stay_displacement)) <= 1.0e-12
            and (stay_step_gain is None or abs(float(stay_step_gain)) <= 1.0e-12)
            and rho_miss_delta is not None
            and float(rho_miss_delta) > 0.0
            and step_gain_delta is not None
            and float(step_gain_delta) >= max(
                float(kink_min_step_gain_delta),
                2.0 * max(0.0, float(getattr(self.cfg, "gain_ratio_threshold", 0.0))),
            )
            and step_residual_ratio is not None
            and float(step_residual_ratio) <= 1.0 + 1.0e-12
            and selected_gain_ratio is not None
            and float(selected_gain_ratio) >= float(candidate_gain_support_floor)
            and (
                condition_ratio is None
                or float(condition_ratio) <= float(zero_motion_condition_ratio_cap)
            )
        )
        hamiltonian_flow_projective_support = bool(
            str(getattr(self, "_family_key", "")) in _HAMILTONIAN_FLOW_FAMILIES
            and bool(projective_append_support_mode)
            and selected_candidate_label == "ham_full"
            and rho_miss_delta is not None
            and float(rho_miss_delta) > 0.0
            and selected_rho is not None
            and float(selected_rho) <= 1.0e-8
            and selected_step_gain is not None
            and float(selected_step_gain) >= 1.0 - 1.0e-8
            and step_gain_delta is not None
            and float(step_gain_delta) >= max(
                float(kink_min_step_gain_delta),
                2.0 * max(0.0, float(getattr(self.cfg, "gain_ratio_threshold", 0.0))),
            )
            and step_residual_ratio is not None
            and float(step_residual_ratio) <= 1.0e-6
            and selected_gain_ratio is not None
            and float(selected_gain_ratio) >= float(candidate_gain_support_floor)
        )
        residual_collapse_projective_support = bool(
            str(getattr(self, "_family_key", "")) in _HAMILTONIAN_FLOW_FAMILIES
            and bool(projective_append_support_mode)
            and not bool(motion_bad)
            and rho_miss_delta is not None
            and float(rho_miss_delta) > 0.0
            and step_gain_delta is not None
            and float(step_gain_delta) >= max(
                float(kink_min_step_gain_delta),
                2.0 * gain_threshold,
            )
            and step_residual_ratio is not None
            and float(step_residual_ratio) <= min(0.25, float(rho_only_step_cap))
            and selected_gain_ratio is not None
            and float(selected_gain_ratio) >= float(candidate_gain_support_floor)
            and (
                condition_ratio is None
                or float(condition_ratio) <= float(zero_motion_condition_ratio_cap)
            )
        )
        steady_projective_support = bool(
            str(getattr(self, "_family_key", "")) in _HAMILTONIAN_FLOW_FAMILIES
            and bool(projective_append_support_mode)
            and not bool(motion_bad)
            and rho_miss_delta is not None
            and float(rho_miss_delta) > 0.0
            and step_gain_delta is not None
            and float(step_gain_delta) >= float(steady_projective_gain_floor)
            and step_residual_ratio is not None
            and float(step_residual_ratio) <= min(1.0, float(rho_only_step_cap))
            and selected_gain_ratio is not None
            and float(selected_gain_ratio) >= float(steady_projective_gain_floor)
            and (
                condition_ratio is None
                or float(condition_ratio) <= float(zero_motion_condition_ratio_cap)
            )
            and (
                displacement_ratio is None
                or float(displacement_ratio) <= float(rho_only_displacement_cap)
            )
        )
        projective_complete_support = bool(
            zero_motion_projective_support
            or residual_collapse_projective_support
            or steady_projective_support
        )
        reason: str | None = None
        if bool(getattr(self.cfg, "append_no_harm_guard_enabled", True)):
            if (
                condition_ratio is not None
                and float(condition_ratio) > float(condition_ratio_cap)
                and selected_condition is not None
                and float(selected_condition) >= float(condition_abs_floor)
                and not bool(projective_complete_support)
            ):
                reason = "no_harm_condition_worse"
            elif (
                displacement_ratio is not None
                and float(displacement_ratio) > float(displacement_ratio_cap)
                and not bool(projective_complete_support)
            ):
                reason = "no_harm_displacement_worse"
            elif bool(motion_bad) and not bool(stability_support) and not bool(projective_complete_support):
                reason = "no_harm_motion_kink"
            else:
                rho_improves = bool(rho_miss_delta is not None and float(rho_miss_delta) > 0.0)
                weak_nonrho_support = bool(
                    step_gain_delta is None
                    or float(step_gain_delta) < float(rho_only_min_step_gain_delta)
                )
                bad_condition = bool(
                    condition_ratio is not None
                    and float(condition_ratio) > float(rho_only_condition_cap)
                )
                bad_step_residual = bool(
                    step_residual_ratio is not None
                    and float(step_residual_ratio) > float(rho_only_step_cap)
                )
                bad_displacement = bool(
                    displacement_ratio is not None
                    and float(displacement_ratio) > float(rho_only_displacement_cap)
                )
                if (
                    bool(rho_improves)
                    and bool(weak_nonrho_support)
                    and (bool(bad_condition) or bool(bad_step_residual) or bool(bad_displacement) or bool(motion_bad))
                    and not bool(projective_complete_support)
                ):
                    reason = "no_harm_rho_miss_only"
        diagnostics: dict[str, Any] = {
            "guard_enabled": bool(getattr(self.cfg, "append_no_harm_guard_enabled", True)),
            "veto_reason": reason,
            "stay_condition_number_next": stay_condition,
            "selected_condition_number_next": selected_condition,
            "condition_ratio_selected_vs_stay": condition_ratio,
            "stay_rho_miss_next": stay_rho,
            "selected_rho_miss_next": selected_rho,
            "rho_miss_delta_stay_minus_selected": rho_miss_delta,
            "stay_step_gain_ratio_next": stay_step_gain,
            "selected_step_gain_ratio_next": selected_step_gain,
            "step_gain_delta_selected_minus_stay": step_gain_delta,
            "stay_epsilon_step_ratio_next": stay_step_residual,
            "selected_epsilon_step_ratio_next": selected_step_residual,
            "step_residual_ratio_selected_vs_stay": step_residual_ratio,
            "stay_predicted_displacement_next": stay_displacement,
            "selected_predicted_displacement_next": selected_displacement,
            "displacement_ratio_selected_vs_stay": displacement_ratio,
            "selected_gain_ratio": selected_gain_ratio,
            "motion_regime": motion_regime,
            "motion_direction_reversal": bool(motion_direction_reversal),
            "motion_curvature_sign_flip": bool(motion_curvature_sign_flip),
            "motion_kink_score": motion_kink_score,
            "motion_bad": bool(motion_bad),
            "stability_support": bool(stability_support),
            "projective_append_support_mode": bool(projective_append_support_mode),
            "zero_motion_projective_support": bool(zero_motion_projective_support),
            "hamiltonian_flow_projective_support": bool(hamiltonian_flow_projective_support),
            "residual_collapse_projective_support": bool(residual_collapse_projective_support),
            "steady_projective_support": bool(steady_projective_support),
            "projective_complete_support": bool(projective_complete_support),
            "zero_motion_condition_ratio_cap": float(zero_motion_condition_ratio_cap),
            "condition_ratio_cap": float(condition_ratio_cap),
            "displacement_ratio_cap": float(displacement_ratio_cap),
            "exact_reference_logging": exact_logging,
            "exact_reference_used_for_veto": False,
        }
        return reason, diagnostics

    def _local_forecast_override_reason(
        self,
        *,
        stay_forecast: Mapping[str, Any],
        selected_forecast: Mapping[str, Any],
        selected: Mapping[str, Any] | None = None,
        motion: MotionSchedulerTelemetry | None = None,
    ) -> str | None:
        stay_score = float(self._forecast_score_total(stay_forecast))
        selected_score = float(self._forecast_score_total(selected_forecast))
        if not np.isfinite(stay_score) or not np.isfinite(selected_score):
            return "local_forecast_nonfinite"
        margin = float(getattr(self.cfg, "forecast_accept_margin", 0.0))
        no_harm_reason, no_harm_diagnostics = self._append_no_harm_guard_reason(
            stay_forecast=stay_forecast,
            selected_forecast=selected_forecast,
            selected=selected,
            motion=motion,
        )
        self._last_append_no_harm_diagnostics = dict(no_harm_diagnostics)
        projective_complete_support = bool(
            no_harm_reason is None
            and bool(no_harm_diagnostics.get("projective_complete_support", False))
        )
        score_override_support = bool(
            projective_complete_support
            and str(getattr(self, "_family_key", "")) in {"spin_boson", "molecular_vibronic_h2"}
        )
        if selected_score >= stay_score - margin and not bool(score_override_support):
            return "local_forecast_no_advantage"
        if no_harm_reason is not None:
            return str(no_harm_reason)
        return None

    def _exact_forecast_tracking_horizon_steps(self) -> int:
        return max(1, int(getattr(self.cfg, "exact_forecast_tracking_horizon_steps", 1)))

    def _exact_forecast_tracking_horizon_weights(
        self,
        *,
        steps: int | None = None,
    ) -> tuple[float, ...]:
        configured_steps = self._exact_forecast_tracking_horizon_steps()
        raw = tuple(float(x) for x in getattr(self.cfg, "exact_forecast_tracking_horizon_weights", ()))
        if not raw:
            weights = tuple(1.0 for _ in range(configured_steps))
        else:
            weights = raw
        active_steps = configured_steps if steps is None else max(1, int(steps))
        return tuple(float(x) for x in weights[:active_steps])

    def _exact_forecast_tracking_primary_density_error_weight(self) -> float:
        raw = getattr(self.cfg, "exact_forecast_tracking_primary_density_error_weight", None)
        if raw is None:
            raw = getattr(self.cfg, "exact_forecast_tracking_staggered_error_weight", 1.0)
        return max(0.0, float(raw))

    def _exact_forecast_density_slope_weight(self) -> float:
        return max(0.0, float(getattr(self.cfg, "exact_forecast_density_slope_weight", 1.0)))

    def _exact_forecast_density_curvature_weight(self) -> float:
        return max(
            0.0,
            float(getattr(self.cfg, "exact_forecast_density_curvature_weight", 0.0)),
        )

    def _exact_forecast_density_excursion_under_weight(self) -> float:
        return max(
            0.0,
            float(getattr(self.cfg, "exact_forecast_density_excursion_under_weight", 0.0)),
        )

    def _exact_forecast_density_excursion_over_weight(self) -> float:
        return max(
            0.0,
            float(getattr(self.cfg, "exact_forecast_density_excursion_over_weight", 0.0)),
        )

    def _exact_forecast_density_sign_lag_weight(self) -> float:
        return max(0.0, float(getattr(self.cfg, "exact_forecast_density_sign_lag_weight", 0.0)))

    def _exact_forecast_density_postcross_wrong_sign_weight(self) -> float:
        return max(
            0.0,
            float(getattr(self.cfg, "exact_forecast_density_postcross_wrong_sign_weight", 0.0)),
        )

    def _exact_v1_postcross_compare_diag_enabled(self) -> bool:
        return bool(
            str(self.cfg.mode) == "exact_v1"
            and getattr(self.cfg, "exact_v1_postcross_compare_diag", False)
        )

    def _exact_v1_below_floor_energy_safe_d_shape_escape_enabled(self) -> bool:
        return bool(getattr(self.cfg, "exact_v1_below_floor_energy_safe_d_shape_escape", False))

    def _exact_v1_d_shape_barrier_ranking_active(self) -> bool:
        return bool(
            str(self.cfg.mode) == "exact_v1"
            and str(getattr(self.cfg, "exact_forecast_guardrail_mode", "off")).strip().lower()
            == "d_shape_barrier_v1"
        )

    def _exact_v1_fidelity_first_barrier_ranking_active(self) -> bool:
        return bool(
            str(self.cfg.mode) == "exact_v1"
            and str(getattr(self.cfg, "exact_forecast_guardrail_mode", "off")).strip().lower()
            == "fidelity_first_barrier_v1"
        )

    def _exact_v1_guarded_turn_window_ranking_active(self) -> bool:
        return bool(
            self._exact_v1_d_shape_barrier_ranking_active()
            or self._exact_v1_fidelity_first_barrier_ranking_active()
        )

    def _exact_v1_single_surface_commit_law_enabled(self) -> bool:
        return bool(
            str(getattr(self.cfg, "mode", "")) == "exact_v1"
            and bool(getattr(self.cfg, "exact_v1_single_surface_commit_law", False))
        )

    def _exact_v1_guarded_commit_compare_score(
        self,
        *,
        forecast: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        curvature_anchor: Mapping[str, Any] | None = None,
    ) -> float:
        if self._exact_v1_single_surface_commit_law_enabled():
            return float(
                self._forecast_tracking_score(
                    forecast=forecast,
                    curvature_anchor=curvature_anchor,
                )
            )
        return float(
            self._exact_v1_guarded_turn_window_core_score(
                forecast=forecast,
                curvature_anchor=curvature_anchor,
            )
        )

    def _exact_v1_guarded_commit_surface_mode(self) -> str:
        if not self._exact_v1_guarded_turn_window_ranking_active():
            return "unguarded"
        return (
            "forecast_tracking_total"
            if self._exact_v1_single_surface_commit_law_enabled()
            else "guarded_turn_window_core"
        )

    def _exact_v1_guarded_protected_horizon_admission_reason(self) -> str:
        if self._exact_v1_fidelity_first_barrier_ranking_active():
            return "fidelity_first_barrier_protected_horizon"
        return "d_shape_barrier_protected_horizon"

    def _exact_v1_guarded_pre_turn_shadow_bridge_reason(self) -> str:
        if self._exact_v1_fidelity_first_barrier_ranking_active():
            return "fidelity_first_barrier_pre_turn_shadow_bridge"
        return "d_shape_barrier_pre_turn_shadow_bridge"

    def _exact_v1_fidelity_first_turn_local_target_win_reason(self) -> str:
        return "fidelity_first_turn_local_target_win"

    def _exact_v1_d_shape_shadow_metrics_enabled(self) -> bool:
        return bool(
            self._exact_v1_d_shape_barrier_ranking_active()
            or self._exact_v1_fidelity_first_barrier_ranking_active()
            or self._exact_v1_postcross_compare_diag_enabled()
            or self._exact_v1_below_floor_energy_safe_d_shape_escape_enabled()
            or float(self._exact_forecast_density_curvature_weight()) > 0.0
            or float(self._exact_forecast_density_excursion_under_weight()) > 0.0
            or float(self._exact_forecast_density_excursion_over_weight()) > 0.0
        )

    def _exact_v1_d_shape_turn_window_abs_activation(self) -> float:
        return max(
            0.0,
            float(getattr(self.cfg, "exact_v1_d_shape_turn_window_abs_activation", 0.0)),
        )

    def _exact_v1_site_turn_error_total(self, forecast: Mapping[str, Any]) -> float | None:
        total = 0.0
        seen = False
        for key in (
            "tracking_site_slope_abs_error_mean_by_site",
            "tracking_site_curvature_abs_error_mean_by_site",
            "tracking_site_excursion_under_response_mean_by_site",
            "tracking_site_excursion_over_response_mean_by_site",
        ):
            vals = np.asarray(forecast.get(key, ()), dtype=float).reshape(-1)
            if vals.size:
                total += float(np.sum(vals))
                seen = True
        return None if not seen else float(total)

    def _tracking_vector_values(self, forecast: Mapping[str, Any], key: str) -> list[float]:
        raw = forecast.get(key, ())
        if raw is None:
            return []
        if isinstance(raw, np.ndarray):
            return [float(x) for x in np.asarray(raw, dtype=float).reshape(-1).tolist()]
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            return [float(x) for x in raw]
        try:
            return [float(raw)]
        except (TypeError, ValueError):
            return []

    def _site_turn_compare_summary(self, forecast: Mapping[str, Any]) -> dict[str, Any]:
        slope = self._tracking_vector_values(
            forecast, "tracking_site_slope_abs_error_mean_by_site"
        )
        curvature = self._tracking_vector_values(
            forecast, "tracking_site_curvature_abs_error_mean_by_site"
        )
        excursion_under = self._tracking_vector_values(
            forecast, "tracking_site_excursion_under_response_mean_by_site"
        )
        excursion_over = self._tracking_vector_values(
            forecast, "tracking_site_excursion_over_response_mean_by_site"
        )
        return {
            "slope_abs_error_mean_by_site": [float(x) for x in slope],
            "curvature_abs_error_mean_by_site": [float(x) for x in curvature],
            "excursion_under_response_mean_by_site": [float(x) for x in excursion_under],
            "excursion_over_response_mean_by_site": [float(x) for x in excursion_over],
            "slope_abs_error_max": float(max(slope)) if slope else 0.0,
            "curvature_abs_error_max": float(max(curvature)) if curvature else 0.0,
            "excursion_under_response_max": (
                float(max(excursion_under)) if excursion_under else 0.0
            ),
            "excursion_over_response_max": (
                float(max(excursion_over)) if excursion_over else 0.0
            ),
        }

    def _site_turn_compare_delta(
        self,
        *,
        summary: Mapping[str, Any],
        stay_summary: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        if stay_summary is None:
            return None

        def _delta_vector(key: str) -> list[float]:
            values = [float(x) for x in summary.get(key, [])]
            stay_values = [float(x) for x in stay_summary.get(key, [])]
            length = min(len(values), len(stay_values))
            return [float(values[idx] - stay_values[idx]) for idx in range(length)]

        slope_delta = _delta_vector("slope_abs_error_mean_by_site")
        curvature_delta = _delta_vector("curvature_abs_error_mean_by_site")
        excursion_under_delta = _delta_vector("excursion_under_response_mean_by_site")
        excursion_over_delta = _delta_vector("excursion_over_response_mean_by_site")
        return {
            "slope_abs_error_mean_by_site": slope_delta,
            "curvature_abs_error_mean_by_site": curvature_delta,
            "excursion_under_response_mean_by_site": excursion_under_delta,
            "excursion_over_response_mean_by_site": excursion_over_delta,
            "slope_abs_error_max": (
                None
                if "slope_abs_error_max" not in stay_summary
                else float(summary.get("slope_abs_error_max", 0.0))
                - float(stay_summary.get("slope_abs_error_max", 0.0))
            ),
            "curvature_abs_error_max": (
                None
                if "curvature_abs_error_max" not in stay_summary
                else float(summary.get("curvature_abs_error_max", 0.0))
                - float(stay_summary.get("curvature_abs_error_max", 0.0))
            ),
            "excursion_under_response_max": (
                None
                if "excursion_under_response_max" not in stay_summary
                else float(summary.get("excursion_under_response_max", 0.0))
                - float(stay_summary.get("excursion_under_response_max", 0.0))
            ),
            "excursion_over_response_max": (
                None
                if "excursion_over_response_max" not in stay_summary
                else float(summary.get("excursion_over_response_max", 0.0))
                - float(stay_summary.get("excursion_over_response_max", 0.0))
            ),
        }

    def _d_shape_compare_summary(self, forecast: Mapping[str, Any]) -> dict[str, Any]:
        slope = float(forecast.get("tracking_primary_density_slope_abs_error_mean", 0.0))
        curvature = float(forecast.get("tracking_d_curvature_abs_error_mean", 0.0))
        excursion_under = float(
            forecast.get("tracking_d_excursion_under_response_mean", 0.0)
        )
        excursion_over = float(
            forecast.get("tracking_d_excursion_over_response_mean", 0.0)
        )
        return {
            "slope_abs_error_mean": float(slope),
            "curvature_abs_error_mean": float(curvature),
            "excursion_under_response_mean": float(excursion_under),
            "excursion_over_response_mean": float(excursion_over),
            "shadow_only_total": float(curvature + excursion_under + excursion_over),
            "total_with_slope": float(slope + curvature + excursion_under + excursion_over),
        }

    def _d_shape_compare_delta(
        self,
        *,
        summary: Mapping[str, Any],
        stay_summary: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        if stay_summary is None:
            return None
        return {
            "slope_abs_error_mean": (
                None
                if "slope_abs_error_mean" not in stay_summary
                else float(summary.get("slope_abs_error_mean", 0.0))
                - float(stay_summary.get("slope_abs_error_mean", 0.0))
            ),
            "curvature_abs_error_mean": (
                None
                if "curvature_abs_error_mean" not in stay_summary
                else float(summary.get("curvature_abs_error_mean", 0.0))
                - float(stay_summary.get("curvature_abs_error_mean", 0.0))
            ),
            "excursion_under_response_mean": (
                None
                if "excursion_under_response_mean" not in stay_summary
                else float(summary.get("excursion_under_response_mean", 0.0))
                - float(stay_summary.get("excursion_under_response_mean", 0.0))
            ),
            "excursion_over_response_mean": (
                None
                if "excursion_over_response_mean" not in stay_summary
                else float(summary.get("excursion_over_response_mean", 0.0))
                - float(stay_summary.get("excursion_over_response_mean", 0.0))
            ),
            "shadow_only_total": (
                None
                if "shadow_only_total" not in stay_summary
                else float(summary.get("shadow_only_total", 0.0))
                - float(stay_summary.get("shadow_only_total", 0.0))
            ),
            "total_with_slope": (
                None
                if "total_with_slope" not in stay_summary
                else float(summary.get("total_with_slope", 0.0))
                - float(stay_summary.get("total_with_slope", 0.0))
            ),
        }

    def _total_occupation_compare_summary(self, forecast: Mapping[str, Any]) -> dict[str, Any]:
        abs_error_next = float(forecast.get("tracking_total_occupation_abs_error_next", 0.0))
        abs_error_mean = float(forecast.get("tracking_total_occupation_abs_error_mean", 0.0))
        return {
            "abs_error_next": float(abs_error_next),
            "abs_error_mean": float(abs_error_mean),
        }

    def _total_occupation_compare_delta(
        self,
        *,
        summary: Mapping[str, Any],
        stay_summary: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        if stay_summary is None:
            return None
        return {
            "abs_error_next": (
                None
                if "abs_error_next" not in stay_summary
                else float(summary.get("abs_error_next", 0.0))
                - float(stay_summary.get("abs_error_next", 0.0))
            ),
            "abs_error_mean": (
                None
                if "abs_error_mean" not in stay_summary
                else float(summary.get("abs_error_mean", 0.0))
                - float(stay_summary.get("abs_error_mean", 0.0))
            ),
        }

    def _forecast_postcross_compare_summary(
        self,
        *,
        forecast: Mapping[str, Any],
        score_total: float | None = None,
    ) -> dict[str, Any]:
        postcross_weight = float(self._exact_forecast_density_postcross_wrong_sign_weight())
        postcross_error_mean = float(
            forecast.get("tracking_primary_density_postcross_wrong_sign_error_mean", 0.0)
        )
        postcross_abs_error_mean = float(
            forecast.get("tracking_primary_density_postcross_wrong_sign_abs_error_mean", 0.0)
        )
        postcross_active = bool(
            float(forecast.get("tracking_primary_density_postcross_wrong_sign_active", 0.0)) > 0.0
        )
        total = float(
            self._forecast_tracking_score(forecast=forecast)
            if score_total is None
            else float(score_total)
        )
        postcross_contribution = float(postcross_weight) * float(postcross_error_mean)
        live_d_breakdown = (
            None
            if not self._exact_v1_d_shape_barrier_ranking_active()
            else self._exact_v1_live_d_score_breakdown(forecast=forecast)
        )
        return {
            "tracking_score_total": float(total),
            "tracking_score_ex_postcross": float(total - postcross_contribution),
            "fidelity_exact_next": float(forecast.get("fidelity_exact_next", float("nan"))),
            "abs_energy_total_error_next": float(forecast.get("abs_energy_total_error_next", float("nan"))),
            "site_occupations_abs_error_max_next": float(
                forecast.get("site_occupations_abs_error_max_next", float("nan"))
            ),
            "abs_primary_density_error_next": float(
                forecast.get(
                    "abs_primary_density_error_next",
                    forecast.get("abs_staggered_error_next", float("nan")),
                )
            ),
            "postcross_active": bool(postcross_active),
            "postcross_error_mean": float(postcross_error_mean),
            "postcross_abs_error_mean": float(postcross_abs_error_mean),
            "postcross_contribution": float(postcross_contribution),
            "live_d_core": (None if live_d_breakdown is None else dict(live_d_breakdown["core"])),
            "live_d_barrier": (None if live_d_breakdown is None else dict(live_d_breakdown["barrier"])),
            "live_d_total": (None if live_d_breakdown is None else float(live_d_breakdown["total"])),
            "d_shape": self._d_shape_compare_summary(forecast),
            "site_turn": self._site_turn_compare_summary(forecast),
            "total_occupation": self._total_occupation_compare_summary(forecast),
        }

    def _exact_v1_postcross_candidate_compare_entry(
        self,
        *,
        record: Mapping[str, Any],
        forecast: Mapping[str, Any],
        score_total: float,
        stay_summary: Mapping[str, Any] | None,
        admitted: bool,
        admission_reason: str | None,
        rejection_reason: str | None,
    ) -> dict[str, Any]:
        summary = self._forecast_postcross_compare_summary(
            forecast=forecast,
            score_total=float(score_total),
        )
        stay_total = None if stay_summary is None else stay_summary.get("tracking_score_total")
        stay_postcross = None if stay_summary is None else stay_summary.get("postcross_contribution")
        stay_live_d_total = None if stay_summary is None else stay_summary.get("live_d_total")
        stay_d_shape = None if stay_summary is None else stay_summary.get("d_shape")
        stay_site_turn = None if stay_summary is None else stay_summary.get("site_turn")
        stay_total_occupation = None if stay_summary is None else stay_summary.get("total_occupation")
        return {
            "candidate_label": str(record.get("candidate_label")),
            "candidate_identity": str(record.get("candidate_identity", record.get("candidate_label"))),
            "candidate_pool_index": int(record.get("candidate_pool_index", -1)),
            "position_id": int(record.get("position_id", -1)),
            "admitted": bool(admitted),
            "admission_reason": (None if admission_reason is None else str(admission_reason)),
            "rejection_reason": (None if rejection_reason is None else str(rejection_reason)),
            **summary,
            "tracking_score_delta_vs_stay": (
                None if stay_total is None else float(summary["tracking_score_total"]) - float(stay_total)
            ),
            "postcross_delta_vs_stay": (
                None
                if stay_postcross is None
                else float(summary["postcross_contribution"]) - float(stay_postcross)
            ),
            "live_d_total_delta_vs_stay": (
                None
                if stay_live_d_total is None or summary.get("live_d_total") is None
                else float(summary["live_d_total"]) - float(stay_live_d_total)
            ),
            "d_shape_delta_vs_stay": self._d_shape_compare_delta(
                summary=summary["d_shape"],
                stay_summary=stay_d_shape,
            ),
            "site_turn_delta_vs_stay": self._site_turn_compare_delta(
                summary=summary["site_turn"],
                stay_summary=stay_site_turn,
            ),
            "total_occupation_delta_vs_stay": self._total_occupation_compare_delta(
                summary=summary["total_occupation"],
                stay_summary=stay_total_occupation,
            ),
        }

    def _exact_forecast_drive_harmonic_weight(self) -> float:
        return max(0.0, float(getattr(self.cfg, "exact_forecast_drive_harmonic_weight", 0.0)))

    def _exact_v1_density_first_target_gain_floor(self) -> float:
        return max(0.0, float(getattr(self.cfg, "exact_v1_density_first_target_gain_floor", 2.0e-2)))

    def _exact_v1_below_floor_probe_target_gain_floor(self) -> float:
        return max(
            0.0,
            float(getattr(self.cfg, "exact_v1_below_floor_probe_target_gain_floor", 3.0e-2)),
        )

    def _exact_v1_sign_lag_window_activation(self) -> float:
        return max(0.0, float(getattr(self.cfg, "exact_v1_sign_lag_window_activation", 0.0)))

    def _exact_v1_sign_lag_window_target_gain_floor(self) -> float | None:
        raw_value = getattr(self.cfg, "exact_v1_sign_lag_window_target_gain_floor", None)
        if raw_value is None:
            return None
        return max(0.0, float(raw_value))

    def _exact_v1_postcross_wrong_sign_activation(self) -> float:
        return max(0.0, float(getattr(self.cfg, "exact_v1_postcross_wrong_sign_activation", 0.0)))

    def _exact_v1_postcross_wrong_sign_target_gain_floor(self) -> float | None:
        raw_value = getattr(self.cfg, "exact_v1_postcross_wrong_sign_target_gain_floor", None)
        if raw_value is None:
            return None
        return max(0.0, float(raw_value))

    def _exact_v1_fidelity_first_turn_local_onset_gain_floor(self) -> float:
        sign_lag_floor = self._exact_v1_sign_lag_window_target_gain_floor()
        density_floor = 0.5 * float(self._exact_v1_density_first_target_gain_floor())
        return max(float(density_floor), float(sign_lag_floor or 0.0))

    def _exact_forecast_scale_floor(self, field_name: str, default: float = 1.0e-6) -> float:
        return max(float(default), float(getattr(self.cfg, field_name, default)))

    def _exact_forecast_tracking_error_weights(self) -> tuple[float, float, float, float, float]:
        return (
            max(0.0, float(getattr(self.cfg, "exact_forecast_tracking_fidelity_defect_weight", 1.0))),
            self._exact_forecast_tracking_primary_density_error_weight(),
            max(0.0, float(getattr(self.cfg, "exact_forecast_tracking_doublon_error_weight", 1.0))),
            max(
                0.0,
                float(getattr(self.cfg, "exact_forecast_tracking_site_occupations_error_weight", 1.0)),
            ),
            max(0.0, float(getattr(self.cfg, "exact_forecast_tracking_energy_total_error_weight", 1.0))),
        )

    def _exact_forecast_pair_weights(self, weights: Sequence[float]) -> np.ndarray:
        weight_arr = np.asarray([float(x) for x in weights], dtype=float)
        if weight_arr.size < 2:
            return np.asarray([], dtype=float)
        return 0.5 * (weight_arr[:-1] + weight_arr[1:])

    def _primary_density_sign_bucket(self, value: float, *, eps: float) -> int:
        if float(value) > float(eps):
            return 1
        if float(value) < -float(eps):
            return -1
        return 0

    def _primary_density_sign_lag_terms(
        self,
        *,
        forecasts: Sequence[Mapping[str, Any]],
        weights: Sequence[float],
        anchor: Mapping[str, Any] | None,
        primary_density_scale: float,
    ) -> dict[str, float]:
        if anchor is None or len(forecasts) == 0:
            return {
                "primary_density_sign_lag_abs_error_mean": 0.0,
                "primary_density_sign_lag_error_mean": 0.0,
                "abs_primary_density_sign_lag_next": 0.0,
                "primary_density_sign_lag_next": 0.0,
            }
        if any(
            ("primary_density_controller_next" not in item)
            or ("primary_density_exact_next" not in item)
            for item in forecasts
        ):
            return {
                "primary_density_sign_lag_abs_error_mean": 0.0,
                "primary_density_sign_lag_error_mean": 0.0,
                "abs_primary_density_sign_lag_next": 0.0,
                "primary_density_sign_lag_next": 0.0,
            }
        if ("primary_density_controller_next" not in anchor) or ("primary_density_exact_next" not in anchor):
            return {
                "primary_density_sign_lag_abs_error_mean": 0.0,
                "primary_density_sign_lag_error_mean": 0.0,
                "abs_primary_density_sign_lag_next": 0.0,
                "primary_density_sign_lag_next": 0.0,
            }
        scale = max(float(primary_density_scale), 1.0e-6)
        sign_eps = max(2.0e-2, 0.1 * float(scale))
        anchor_exact = float(anchor["primary_density_exact_next"])
        anchor_exact_sign = self._primary_density_sign_bucket(anchor_exact, eps=float(sign_eps))
        penalties_abs: list[float] = []
        for item in forecasts:
            ctrl_value = float(item["primary_density_controller_next"])
            exact_value = float(item["primary_density_exact_next"])
            ctrl_sign = self._primary_density_sign_bucket(ctrl_value, eps=float(sign_eps))
            exact_sign = self._primary_density_sign_bucket(exact_value, eps=float(sign_eps))
            penalty_abs = 0.0
            sign_mismatch = (
                int(ctrl_sign) != 0
                and int(exact_sign) != 0
                and int(ctrl_sign) != int(exact_sign)
            )
            if sign_mismatch:
                penalty_abs = max(float(penalty_abs), abs(float(ctrl_value) - float(exact_value)))
            delayed_flip = (
                int(anchor_exact_sign) != 0
                and int(exact_sign) != 0
                and int(exact_sign) != int(anchor_exact_sign)
                and int(ctrl_sign) == int(anchor_exact_sign)
            )
            if delayed_flip:
                penalty_abs = max(float(penalty_abs), abs(float(ctrl_value)) + abs(float(exact_value)))
            penalties_abs.append(float(penalty_abs))
        weight_arr = np.asarray([float(x) for x in weights], dtype=float)
        weight_sum = float(np.sum(weight_arr))
        if weight_sum <= 0.0:
            return {
                "primary_density_sign_lag_abs_error_mean": 0.0,
                "primary_density_sign_lag_error_mean": 0.0,
                "abs_primary_density_sign_lag_next": float(penalties_abs[0]) if penalties_abs else 0.0,
                "primary_density_sign_lag_next": float((penalties_abs[0] / float(scale)) if penalties_abs else 0.0),
            }
        mean_abs = float(np.sum(weight_arr * np.asarray(penalties_abs, dtype=float)) / weight_sum)
        step_abs = float(penalties_abs[0]) if penalties_abs else 0.0
        return {
            "primary_density_sign_lag_abs_error_mean": float(mean_abs),
            "primary_density_sign_lag_error_mean": float(mean_abs / float(scale)),
            "abs_primary_density_sign_lag_next": float(step_abs),
            "primary_density_sign_lag_next": float(step_abs / float(scale)),
        }

    def _drive_harmonic_tracking_terms(
        self,
        *,
        forecasts: Sequence[Mapping[str, Any]],
        weights: Sequence[float],
    ) -> dict[str, float]:
        zero_terms = {
            "drive_harmonic_mismatch": 0.0,
            "drive_harmonic_ctrl_real": 0.0,
            "drive_harmonic_ctrl_imag": 0.0,
            "drive_harmonic_exact_real": 0.0,
            "drive_harmonic_exact_imag": 0.0,
        }
        if len(forecasts) == 0:
            return dict(zero_terms)
        if not bool(getattr(self._drive_config, "enabled", False)):
            return dict(zero_terms)
        omega_drive = float(getattr(self._drive_config, "drive_omega", 0.0))
        if (not np.isfinite(omega_drive)) or abs(float(omega_drive)) <= 1.0e-12:
            return dict(zero_terms)
        if any(
            ("primary_density_controller_next" not in item)
            or ("primary_density_exact_next" not in item)
            for item in forecasts
        ):
            return dict(zero_terms)
        time_stops: list[float] = []
        for item in forecasts:
            time_value = item.get("time_stop_next", item.get("time_stop"))
            if time_value is None or (not np.isfinite(float(time_value))):
                return dict(zero_terms)
            time_stops.append(float(time_value))
        weight_arr = np.asarray([float(x) for x in weights], dtype=float)
        weight_sum = float(np.sum(weight_arr))
        if weight_sum <= 0.0:
            return dict(zero_terms)
        normalized_weights = np.asarray(weight_arr / weight_sum, dtype=float)
        phase = np.exp(-1j * float(omega_drive) * np.asarray(time_stops, dtype=float))
        ctrl_values = np.asarray(
            [float(item["primary_density_controller_next"]) for item in forecasts],
            dtype=float,
        )
        exact_values = np.asarray(
            [float(item["primary_density_exact_next"]) for item in forecasts],
            dtype=float,
        )
        z_ctrl = np.sum(normalized_weights * ctrl_values * phase)
        z_exact = np.sum(normalized_weights * exact_values * phase)
        mismatch = float((abs(z_ctrl - z_exact) ** 2) / (1.0e-8 + abs(z_exact) ** 2))
        return {
            "drive_harmonic_mismatch": float(mismatch),
            "drive_harmonic_ctrl_real": float(np.real(z_ctrl)),
            "drive_harmonic_ctrl_imag": float(np.imag(z_ctrl)),
            "drive_harmonic_exact_real": float(np.real(z_exact)),
            "drive_harmonic_exact_imag": float(np.imag(z_exact)),
        }

    def _primary_density_postcross_wrong_sign_terms(
        self,
        *,
        forecasts: Sequence[Mapping[str, Any]],
        weights: Sequence[float],
        anchor: Mapping[str, Any] | None,
        primary_density_scale: float,
    ) -> dict[str, float]:
        zero_terms = {
            "primary_density_postcross_wrong_sign_abs_error_mean": 0.0,
            "primary_density_postcross_wrong_sign_error_mean": 0.0,
            "primary_density_postcross_wrong_sign_active": 0.0,
        }
        if anchor is None or len(forecasts) == 0:
            return dict(zero_terms)
        if any(
            ("primary_density_controller_next" not in item)
            or ("primary_density_exact_next" not in item)
            for item in forecasts
        ):
            return dict(zero_terms)
        if ("primary_density_exact_next" not in anchor) or ("primary_density_controller_next" not in anchor):
            return dict(zero_terms)
        scale = max(float(primary_density_scale), 1.0e-6)
        sign_eps = max(2.0e-2, 0.1 * float(scale))
        anchor_exact_sign = self._primary_density_sign_bucket(
            float(anchor["primary_density_exact_next"]),
            eps=float(sign_eps),
        )
        crossed = False
        debts: list[float] = []
        active_weights: list[float] = []
        for idx, item in enumerate(forecasts):
            exact_value = float(item["primary_density_exact_next"])
            ctrl_value = float(item["primary_density_controller_next"])
            exact_sign = self._primary_density_sign_bucket(exact_value, eps=float(sign_eps))
            if not crossed:
                crossed = bool(
                    int(anchor_exact_sign) != 0
                    and int(exact_sign) != 0
                    and int(exact_sign) != int(anchor_exact_sign)
                )
            if not crossed or int(exact_sign) == 0:
                continue
            debt = max(0.0, -float(exact_sign) * float(ctrl_value))
            debts.append(float(debt))
            active_weights.append(float(weights[idx]))
        if not debts:
            return dict(zero_terms)
        weight_arr = np.asarray(active_weights, dtype=float)
        weight_sum = float(np.sum(weight_arr))
        if weight_sum <= 0.0:
            return dict(zero_terms)
        mean_abs = float(np.sum(weight_arr * np.asarray(debts, dtype=float)) / weight_sum)
        return {
            "primary_density_postcross_wrong_sign_abs_error_mean": float(mean_abs),
            "primary_density_postcross_wrong_sign_error_mean": float(mean_abs / float(scale)),
            "primary_density_postcross_wrong_sign_active": 1.0,
        }

    def _exact_forecast_normalize_terms(
        self,
        *,
        forecasts: Sequence[Mapping[str, Any]],
        weights: Sequence[float],
        anchor: Mapping[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        normalized = [dict(item) for item in forecasts]
        if len(normalized) == 0:
            return [], {
                "primary_density_scale": self._exact_forecast_scale_floor("exact_forecast_primary_density_scale_floor"),
                "doublon_scale": self._exact_forecast_scale_floor("exact_forecast_doublon_scale_floor"),
                "site_occupations_scale": self._exact_forecast_scale_floor("exact_forecast_site_occupations_scale_floor"),
                "energy_total_scale": self._exact_forecast_scale_floor("exact_forecast_energy_total_scale_floor"),
                "primary_density_slope_scale": self._exact_forecast_scale_floor("exact_forecast_density_slope_scale_floor"),
                "primary_density_slope_abs_error_mean": 0.0,
                "primary_density_slope_error_mean": 0.0,
                "abs_primary_density_slope_error_next": 0.0,
                "primary_density_slope_error_next": 0.0,
            }
        weight_arr = np.asarray([float(x) for x in weights], dtype=float)
        weight_sum = float(np.sum(weight_arr))
        if weight_sum <= 0.0:
            return normalized, {
                "primary_density_scale": 1.0,
                "doublon_scale": 1.0,
                "site_occupations_scale": 1.0,
                "energy_total_scale": 1.0,
                "primary_density_slope_scale": 1.0,
                "primary_density_slope_abs_error_mean": 0.0,
                "primary_density_slope_error_mean": 0.0,
                "abs_primary_density_slope_error_next": 0.0,
                "primary_density_slope_error_next": 0.0,
            }
        if anchor is None:
            anchor = normalized[0]
        anchor_primary_density_exact = float(
            anchor.get("primary_density_exact_next", normalized[0]["primary_density_exact_next"])
        )
        anchor_doublon_exact = self._finite_float_or_none(
            anchor.get("doublon_exact_next", normalized[0].get("doublon_exact_next"))
        )
        anchor_site_exact = np.asarray(
            anchor.get("site_occupations_exact_next", normalized[0]["site_occupations_exact_next"]),
            dtype=float,
        ).reshape(-1)
        anchor_energy_exact = float(
            anchor.get("energy_total_exact_next", normalized[0]["energy_total_exact_next"])
        )
        primary_density_exact = np.asarray(
            [float(item["primary_density_exact_next"]) for item in normalized],
            dtype=float,
        )
        doublon_exact_values = [
            self._finite_float_or_none(item.get("doublon_exact_next", None)) for item in normalized
        ]
        doublon_exact = (
            np.asarray([float(x) for x in doublon_exact_values], dtype=float)
            if anchor_doublon_exact is not None and all(value is not None for value in doublon_exact_values)
            else None
        )
        energy_exact = np.asarray(
            [float(item["energy_total_exact_next"]) for item in normalized],
            dtype=float,
        )
        site_exact_err = np.asarray(
            [
                np.max(
                    np.abs(
                        np.asarray(item["site_occupations_exact_next"], dtype=float).reshape(-1)
                        - anchor_site_exact
                    )
                )
                for item in normalized
            ],
            dtype=float,
        )
        primary_density_scale = max(
            self._exact_forecast_scale_floor("exact_forecast_primary_density_scale_floor"),
            float(np.sum(weight_arr * np.abs(primary_density_exact - anchor_primary_density_exact)) / weight_sum),
        )
        doublon_scale = self._exact_forecast_scale_floor(
            "exact_forecast_doublon_scale_floor"
        )
        if doublon_exact is not None and anchor_doublon_exact is not None:
            doublon_scale = max(
                float(doublon_scale),
                float(np.sum(weight_arr * np.abs(doublon_exact - anchor_doublon_exact)) / weight_sum),
            )
        site_occupations_scale = max(
            self._exact_forecast_scale_floor("exact_forecast_site_occupations_scale_floor"),
            float(np.sum(weight_arr * site_exact_err) / weight_sum),
        )
        energy_total_scale = max(
            self._exact_forecast_scale_floor("exact_forecast_energy_total_scale_floor"),
            float(np.sum(weight_arr * np.abs(energy_exact - anchor_energy_exact)) / weight_sum),
        )
        for item in normalized:
            primary_density_error = self._optional_forecast_metric(
                item,
                normalized_key=None,
                raw_key="abs_primary_density_error_next",
                raw_fallback_key="abs_staggered_error_next",
            )
            doublon_error = self._finite_float_or_none(item.get("abs_doublon_error_next", None))
            site_occupations_error = self._finite_float_or_none(
                item.get("site_occupations_abs_error_max_next", None)
            )
            energy_total_error = self._finite_float_or_none(
                item.get("abs_energy_total_error_next", None)
            )
            item["normalized_primary_density_error_next"] = (
                None
                if primary_density_error is None
                else float(primary_density_error) / float(primary_density_scale)
            )
            item["normalized_doublon_error_next"] = (
                None
                if doublon_error is None
                else float(doublon_error) / float(doublon_scale)
            )
            item["normalized_site_occupations_abs_error_max_next"] = (
                None
                if site_occupations_error is None
                else float(site_occupations_error) / float(site_occupations_scale)
            )
            item["normalized_energy_total_error_next"] = (
                None
                if energy_total_error is None
                else float(energy_total_error) / float(energy_total_scale)
            )
        primary_density_ctrl = np.asarray(
            [float(item["primary_density_controller_next"]) for item in normalized],
            dtype=float,
        )
        pair_weights = self._exact_forecast_pair_weights(weights)
        slope_error_mean = 0.0
        slope_abs_error_mean = 0.0
        slope_scale = self._exact_forecast_scale_floor("exact_forecast_density_slope_scale_floor")
        if pair_weights.size > 0:
            pair_weight_sum = float(np.sum(pair_weights))
            exact_diffs = np.diff(primary_density_exact)
            ctrl_diffs = np.diff(primary_density_ctrl)
            slope_mismatch = np.abs(ctrl_diffs - exact_diffs)
            slope_scale = max(
                float(slope_scale),
                float(np.sum(pair_weights * np.abs(exact_diffs)) / pair_weight_sum),
            )
            slope_abs_error_mean = float(np.sum(pair_weights * slope_mismatch) / pair_weight_sum)
            slope_error_mean = float(slope_abs_error_mean / float(slope_scale))
        step_slope_abs_error = 0.0
        step_slope_error = 0.0
        if anchor is not None:
            step_exact = float(normalized[0]["primary_density_exact_next"]) - float(
                anchor.get("primary_density_exact_next", normalized[0]["primary_density_exact_next"])
            )
            step_ctrl = float(normalized[0]["primary_density_controller_next"]) - float(
                anchor.get("primary_density_controller_next", normalized[0]["primary_density_controller_next"])
            )
            step_scale = max(float(slope_scale), abs(float(step_exact)))
            step_slope_abs_error = float(abs(float(step_ctrl) - float(step_exact)))
            step_slope_error = float(step_slope_abs_error / float(step_scale))
            normalized[0]["abs_primary_density_slope_error_next"] = float(step_slope_abs_error)
            normalized[0]["primary_density_slope_error_next"] = float(step_slope_error)
        sign_lag_terms = self._primary_density_sign_lag_terms(
            forecasts=normalized,
            weights=weights,
            anchor=anchor,
            primary_density_scale=float(primary_density_scale),
        )
        normalized[0]["abs_primary_density_sign_lag_next"] = float(
            sign_lag_terms["abs_primary_density_sign_lag_next"]
        )
        normalized[0]["primary_density_sign_lag_next"] = float(
            sign_lag_terms["primary_density_sign_lag_next"]
        )
        postcross_wrong_sign_terms = self._primary_density_postcross_wrong_sign_terms(
            forecasts=normalized,
            weights=weights,
            anchor=anchor,
            primary_density_scale=float(primary_density_scale),
        )
        return normalized, {
            "primary_density_scale": float(primary_density_scale),
            "doublon_scale": float(doublon_scale),
            "site_occupations_scale": float(site_occupations_scale),
            "energy_total_scale": float(energy_total_scale),
            "primary_density_slope_scale": float(slope_scale),
            "primary_density_slope_abs_error_mean": float(slope_abs_error_mean),
            "primary_density_slope_error_mean": float(slope_error_mean),
            "abs_primary_density_slope_error_next": float(step_slope_abs_error),
            "primary_density_slope_error_next": float(step_slope_error),
            "primary_density_sign_lag_abs_error_mean": float(
                sign_lag_terms["primary_density_sign_lag_abs_error_mean"]
            ),
            "primary_density_sign_lag_error_mean": float(
                sign_lag_terms["primary_density_sign_lag_error_mean"]
            ),
            "abs_primary_density_sign_lag_next": float(
                sign_lag_terms["abs_primary_density_sign_lag_next"]
            ),
            "primary_density_sign_lag_next": float(
                sign_lag_terms["primary_density_sign_lag_next"]
            ),
            "primary_density_postcross_wrong_sign_abs_error_mean": float(
                postcross_wrong_sign_terms["primary_density_postcross_wrong_sign_abs_error_mean"]
            ),
            "primary_density_postcross_wrong_sign_error_mean": float(
                postcross_wrong_sign_terms["primary_density_postcross_wrong_sign_error_mean"]
            ),
            "primary_density_postcross_wrong_sign_active": float(
                postcross_wrong_sign_terms["primary_density_postcross_wrong_sign_active"]
            ),
        }

    def _exact_forecast_horizon_length(
        self,
        *,
        time_stop: float,
    ) -> int:
        requested = self._exact_forecast_tracking_horizon_steps()
        if int(self.times.size) <= 0:
            return int(requested)
        idx = int(np.argmin(np.abs(np.asarray(self.times, dtype=float) - float(time_stop))))
        remaining = max(1, int(self.times.size) - int(idx))
        return int(min(int(requested), int(remaining)))

    def _exact_forecast_energy_shape_weights(self) -> tuple[float, float]:
        return (
            max(0.0, float(getattr(self.cfg, "exact_forecast_energy_slope_weight", 0.0))),
            max(0.0, float(getattr(self.cfg, "exact_forecast_energy_curvature_weight", 0.0))),
        )

    def _exact_forecast_energy_excursion_under_weight(self) -> float:
        return max(
            0.0,
            float(getattr(self.cfg, "exact_forecast_energy_excursion_under_weight", 0.0)),
        )

    def _exact_forecast_energy_excursion_over_weight(self) -> float:
        return max(
            0.0,
            float(getattr(self.cfg, "exact_forecast_energy_excursion_over_weight", 0.0)),
        )

    def _exact_forecast_energy_excursion_rel_tolerance(self) -> float:
        return max(
            0.0,
            float(getattr(self.cfg, "exact_forecast_energy_excursion_rel_tolerance", 0.0)),
        )

    def _energy_shape_tracking_terms(
        self,
        *,
        forecasts: Sequence[Mapping[str, Any]],
        weights: Sequence[float],
        curvature_anchor: Mapping[str, Any] | None = None,
    ) -> dict[str, float]:
        if any(
            ("energy_total_controller_next" not in item) or ("energy_total_exact_next" not in item)
            for item in forecasts
        ):
            return {
                "energy_slope_abs_error_mean": 0.0,
                "energy_curvature_abs_error_mean": 0.0,
            }
        energy_ctrl = np.asarray(
            [float(item["energy_total_controller_next"]) for item in forecasts],
            dtype=float,
        )
        energy_exact = np.asarray(
            [float(item["energy_total_exact_next"]) for item in forecasts],
            dtype=float,
        )
        weight_arr = np.asarray([float(x) for x in weights], dtype=float)
        slope_error = 0.0
        curvature_error = 0.0
        slope_ctrl = energy_ctrl
        slope_exact = energy_exact
        slope_weight_arr = weight_arr
        if curvature_anchor is not None and (
            "energy_total_controller_next" in curvature_anchor
            and "energy_total_exact_next" in curvature_anchor
            and weight_arr.size >= 1
        ):
            slope_ctrl = np.concatenate(
                (
                    np.asarray(
                        [float(curvature_anchor["energy_total_controller_next"])],
                        dtype=float,
                    ),
                    energy_ctrl,
                )
            )
            slope_exact = np.concatenate(
                (
                    np.asarray(
                        [float(curvature_anchor["energy_total_exact_next"])],
                        dtype=float,
                    ),
                    energy_exact,
                )
            )
            slope_weight_arr = np.concatenate(
                (
                    np.asarray([float(weight_arr[0])], dtype=float),
                    weight_arr,
                )
            )
        if slope_ctrl.size >= 2:
            slope_mismatch = np.abs(np.diff(slope_ctrl) - np.diff(slope_exact))
            slope_weights = 0.5 * (slope_weight_arr[:-1] + slope_weight_arr[1:])
            slope_weight_sum = float(np.sum(slope_weights))
            if slope_weight_sum > 0.0:
                slope_error = float(np.sum(slope_weights * slope_mismatch) / slope_weight_sum)
        curvature_ctrl = energy_ctrl
        curvature_exact = energy_exact
        curvature_weight_arr = weight_arr
        if curvature_anchor is not None and (
            "energy_total_controller_next" in curvature_anchor
            and "energy_total_exact_next" in curvature_anchor
            and weight_arr.size >= 1
        ):
            curvature_ctrl = np.concatenate(
                (
                    np.asarray(
                        [float(curvature_anchor["energy_total_controller_next"])],
                        dtype=float,
                    ),
                    energy_ctrl,
                )
            )
            curvature_exact = np.concatenate(
                (
                    np.asarray(
                        [float(curvature_anchor["energy_total_exact_next"])],
                        dtype=float,
                    ),
                    energy_exact,
                )
            )
            curvature_weight_arr = np.concatenate(
                (
                    np.asarray([float(weight_arr[0])], dtype=float),
                    weight_arr,
                )
            )
        if curvature_ctrl.size >= 3:
            curvature_mismatch = np.abs(
                np.diff(curvature_ctrl, n=2) - np.diff(curvature_exact, n=2)
            )
            curvature_weights = (
                curvature_weight_arr[:-2]
                + curvature_weight_arr[1:-1]
                + curvature_weight_arr[2:]
            ) / 3.0
            curvature_weight_sum = float(np.sum(curvature_weights))
            if curvature_weight_sum > 0.0:
                curvature_error = float(
                    np.sum(curvature_weights * curvature_mismatch) / curvature_weight_sum
                )
        return {
            "energy_slope_abs_error_mean": float(slope_error),
            "energy_curvature_abs_error_mean": float(curvature_error),
        }

    def _energy_excursion_tracking_terms(
        self,
        *,
        forecasts: Sequence[Mapping[str, Any]],
        weights: Sequence[float],
        anchor: Mapping[str, Any] | None,
    ) -> dict[str, float]:
        if anchor is None:
            return {
                "energy_excursion_under_response_mean": 0.0,
                "energy_excursion_over_response_mean": 0.0,
            }
        if (
            "energy_total_controller_next" not in anchor
            or "energy_total_exact_next" not in anchor
            or any(
                ("energy_total_controller_next" not in item) or ("energy_total_exact_next" not in item)
                for item in forecasts
            )
        ):
            return {
                "energy_excursion_under_response_mean": 0.0,
                "energy_excursion_over_response_mean": 0.0,
            }
        anchor_ctrl = float(anchor["energy_total_controller_next"])
        anchor_exact = float(anchor["energy_total_exact_next"])
        under_penalties: list[float] = []
        over_penalties: list[float] = []
        rel_tolerance = self._exact_forecast_energy_excursion_rel_tolerance()
        for item in forecasts:
            ctrl_exc = float(item["energy_total_controller_next"]) - float(anchor_ctrl)
            exact_exc = float(item["energy_total_exact_next"]) - float(anchor_exact)
            if abs(float(exact_exc)) <= 1.0e-15:
                under_penalties.append(0.0)
                over_penalties.append(0.0)
                continue
            projected_ctrl = float(np.sign(exact_exc)) * float(ctrl_exc)
            target = abs(float(exact_exc))
            band = float(rel_tolerance) * float(target)
            lower = max(0.0, float(target) - float(band))
            upper = float(target) + float(band)
            under_penalties.append(max(0.0, float(lower) - float(projected_ctrl)))
            over_penalties.append(max(0.0, float(projected_ctrl) - float(upper)))
        weight_arr = np.asarray([float(x) for x in weights], dtype=float)
        weight_sum = float(np.sum(weight_arr))
        if weight_sum <= 0.0:
            return {
                "energy_excursion_under_response_mean": 0.0,
                "energy_excursion_over_response_mean": 0.0,
            }
        return {
            "energy_excursion_under_response_mean": float(
                np.sum(weight_arr * np.asarray(under_penalties, dtype=float)) / weight_sum
            ),
            "energy_excursion_over_response_mean": float(
                np.sum(weight_arr * np.asarray(over_penalties, dtype=float)) / weight_sum
            ),
        }

    def _d_shape_tracking_terms(
        self,
        *,
        forecasts: Sequence[Mapping[str, Any]],
        weights: Sequence[float],
        curvature_anchor: Mapping[str, Any] | None = None,
    ) -> dict[str, float]:
        if any(
            ("primary_density_controller_next" not in item)
            or ("primary_density_exact_next" not in item)
            for item in forecasts
        ):
            return {
                "d_curvature_abs_error_mean": 0.0,
            }
        d_ctrl = np.asarray(
            [float(item["primary_density_controller_next"]) for item in forecasts],
            dtype=float,
        )
        d_exact = np.asarray(
            [float(item["primary_density_exact_next"]) for item in forecasts],
            dtype=float,
        )
        weight_arr = np.asarray([float(x) for x in weights], dtype=float)
        curvature_error = 0.0
        curvature_ctrl = d_ctrl
        curvature_exact = d_exact
        curvature_weight_arr = weight_arr
        if curvature_anchor is not None and (
            "primary_density_controller_next" in curvature_anchor
            and "primary_density_exact_next" in curvature_anchor
            and weight_arr.size >= 1
        ):
            curvature_ctrl = np.concatenate(
                (
                    np.asarray(
                        [float(curvature_anchor["primary_density_controller_next"])],
                        dtype=float,
                    ),
                    d_ctrl,
                )
            )
            curvature_exact = np.concatenate(
                (
                    np.asarray(
                        [float(curvature_anchor["primary_density_exact_next"])],
                        dtype=float,
                    ),
                    d_exact,
                )
            )
            curvature_weight_arr = np.concatenate(
                (
                    np.asarray([float(weight_arr[0])], dtype=float),
                    weight_arr,
                )
            )
        if curvature_ctrl.size >= 3:
            curvature_mismatch = np.abs(
                np.diff(curvature_ctrl, n=2) - np.diff(curvature_exact, n=2)
            )
            curvature_weights = (
                curvature_weight_arr[:-2]
                + curvature_weight_arr[1:-1]
                + curvature_weight_arr[2:]
            ) / 3.0
            curvature_weight_sum = float(np.sum(curvature_weights))
            if curvature_weight_sum > 0.0:
                curvature_error = float(
                    np.sum(curvature_weights * curvature_mismatch) / curvature_weight_sum
                )
        return {
            "d_curvature_abs_error_mean": float(curvature_error),
        }

    def _d_excursion_tracking_terms(
        self,
        *,
        forecasts: Sequence[Mapping[str, Any]],
        weights: Sequence[float],
        anchor: Mapping[str, Any] | None,
    ) -> dict[str, float]:
        if anchor is None:
            return {
                "d_excursion_under_response_mean": 0.0,
                "d_excursion_over_response_mean": 0.0,
            }
        if (
            "primary_density_controller_next" not in anchor
            or "primary_density_exact_next" not in anchor
            or any(
                ("primary_density_controller_next" not in item)
                or ("primary_density_exact_next" not in item)
                for item in forecasts
            )
        ):
            return {
                "d_excursion_under_response_mean": 0.0,
                "d_excursion_over_response_mean": 0.0,
            }
        anchor_ctrl = float(anchor["primary_density_controller_next"])
        anchor_exact = float(anchor["primary_density_exact_next"])
        under_penalties: list[float] = []
        over_penalties: list[float] = []
        for item in forecasts:
            ctrl_exc = float(item["primary_density_controller_next"]) - float(anchor_ctrl)
            exact_exc = float(item["primary_density_exact_next"]) - float(anchor_exact)
            if abs(float(exact_exc)) <= 1.0e-15:
                under_penalties.append(0.0)
                over_penalties.append(0.0)
                continue
            projected_ctrl = float(np.sign(exact_exc)) * float(ctrl_exc)
            target = abs(float(exact_exc))
            under_penalties.append(max(0.0, float(target) - float(projected_ctrl)))
            over_penalties.append(max(0.0, float(projected_ctrl) - float(target)))
        weight_arr = np.asarray([float(x) for x in weights], dtype=float)
        weight_sum = float(np.sum(weight_arr))
        if weight_sum <= 0.0:
            return {
                "d_excursion_under_response_mean": 0.0,
                "d_excursion_over_response_mean": 0.0,
            }
        return {
            "d_excursion_under_response_mean": float(
                np.sum(weight_arr * np.asarray(under_penalties, dtype=float)) / weight_sum
            ),
            "d_excursion_over_response_mean": float(
                np.sum(weight_arr * np.asarray(over_penalties, dtype=float)) / weight_sum
            ),
        }

    def _total_occupation_tracking_terms(
        self,
        *,
        forecasts: Sequence[Mapping[str, Any]],
        weights: Sequence[float],
    ) -> dict[str, float]:
        if any(
            ("site_occupations_controller_next" not in item)
            or ("site_occupations_exact_next" not in item)
            for item in forecasts
        ):
            return {
                "total_occupation_abs_error_next": 0.0,
                "total_occupation_abs_error_mean": 0.0,
            }
        weight_arr = np.asarray([float(x) for x in weights], dtype=float)
        weight_sum = float(np.sum(weight_arr))
        if weight_sum <= 0.0:
            return {
                "total_occupation_abs_error_next": 0.0,
                "total_occupation_abs_error_mean": 0.0,
            }
        errors: list[float] = []
        for item in forecasts:
            ctrl = np.asarray(item["site_occupations_controller_next"], dtype=float).reshape(-1)
            exact = np.asarray(item["site_occupations_exact_next"], dtype=float).reshape(-1)
            if ctrl.shape != exact.shape:
                return {
                    "total_occupation_abs_error_next": 0.0,
                    "total_occupation_abs_error_mean": 0.0,
                }
            errors.append(float(abs(float(np.sum(ctrl)) - float(np.sum(exact)))))
        error_arr = np.asarray(errors, dtype=float)
        return {
            "total_occupation_abs_error_next": float(error_arr[0]) if error_arr.size else 0.0,
            "total_occupation_abs_error_mean": float(
                np.sum(weight_arr * error_arr) / weight_sum
            ),
        }

    def _site_shape_tracking_terms(
        self,
        *,
        forecasts: Sequence[Mapping[str, Any]],
        weights: Sequence[float],
        curvature_anchor: Mapping[str, Any] | None = None,
    ) -> dict[str, list[float]]:
        zero_terms = {
            "site_slope_abs_error_mean_by_site": [],
            "site_curvature_abs_error_mean_by_site": [],
        }
        if any(
            ("site_occupations_controller_next" not in item)
            or ("site_occupations_exact_next" not in item)
            for item in forecasts
        ):
            return dict(zero_terms)
        site_ctrl = np.asarray(
            [np.asarray(item["site_occupations_controller_next"], dtype=float).reshape(-1) for item in forecasts],
            dtype=float,
        )
        site_exact = np.asarray(
            [np.asarray(item["site_occupations_exact_next"], dtype=float).reshape(-1) for item in forecasts],
            dtype=float,
        )
        if site_ctrl.ndim != 2 or site_exact.ndim != 2 or site_ctrl.shape != site_exact.shape:
            return dict(zero_terms)
        weight_arr = np.asarray([float(x) for x in weights], dtype=float)
        site_count = int(site_ctrl.shape[1])
        slope_error = np.zeros(site_count, dtype=float)
        curvature_error = np.zeros(site_count, dtype=float)
        slope_ctrl = site_ctrl
        slope_exact = site_exact
        slope_weight_arr = weight_arr
        if curvature_anchor is not None and (
            "site_occupations_controller_next" in curvature_anchor
            and "site_occupations_exact_next" in curvature_anchor
            and weight_arr.size >= 1
        ):
            anchor_ctrl = np.asarray(
                curvature_anchor["site_occupations_controller_next"],
                dtype=float,
            ).reshape(1, -1)
            anchor_exact = np.asarray(
                curvature_anchor["site_occupations_exact_next"],
                dtype=float,
            ).reshape(1, -1)
            if anchor_ctrl.shape[1] == site_count and anchor_exact.shape[1] == site_count:
                slope_ctrl = np.concatenate((anchor_ctrl, site_ctrl), axis=0)
                slope_exact = np.concatenate((anchor_exact, site_exact), axis=0)
                slope_weight_arr = np.concatenate(
                    (np.asarray([float(weight_arr[0])], dtype=float), weight_arr)
                )
        if slope_ctrl.shape[0] >= 2:
            slope_mismatch = np.abs(np.diff(slope_ctrl, axis=0) - np.diff(slope_exact, axis=0))
            slope_weights = 0.5 * (slope_weight_arr[:-1] + slope_weight_arr[1:])
            slope_weight_sum = float(np.sum(slope_weights))
            if slope_weight_sum > 0.0:
                slope_error = np.sum(slope_mismatch * slope_weights[:, None], axis=0) / slope_weight_sum
        curvature_ctrl = site_ctrl
        curvature_exact = site_exact
        curvature_weight_arr = weight_arr
        if curvature_anchor is not None and (
            "site_occupations_controller_next" in curvature_anchor
            and "site_occupations_exact_next" in curvature_anchor
            and weight_arr.size >= 1
        ):
            anchor_ctrl = np.asarray(
                curvature_anchor["site_occupations_controller_next"],
                dtype=float,
            ).reshape(1, -1)
            anchor_exact = np.asarray(
                curvature_anchor["site_occupations_exact_next"],
                dtype=float,
            ).reshape(1, -1)
            if anchor_ctrl.shape[1] == site_count and anchor_exact.shape[1] == site_count:
                curvature_ctrl = np.concatenate((anchor_ctrl, site_ctrl), axis=0)
                curvature_exact = np.concatenate((anchor_exact, site_exact), axis=0)
                curvature_weight_arr = np.concatenate(
                    (np.asarray([float(weight_arr[0])], dtype=float), weight_arr)
                )
        if curvature_ctrl.shape[0] >= 3:
            curvature_mismatch = np.abs(
                np.diff(curvature_ctrl, n=2, axis=0) - np.diff(curvature_exact, n=2, axis=0)
            )
            curvature_weights = (
                curvature_weight_arr[:-2]
                + curvature_weight_arr[1:-1]
                + curvature_weight_arr[2:]
            ) / 3.0
            curvature_weight_sum = float(np.sum(curvature_weights))
            if curvature_weight_sum > 0.0:
                curvature_error = (
                    np.sum(curvature_mismatch * curvature_weights[:, None], axis=0)
                    / curvature_weight_sum
                )
        return {
            "site_slope_abs_error_mean_by_site": [float(x) for x in slope_error.tolist()],
            "site_curvature_abs_error_mean_by_site": [float(x) for x in curvature_error.tolist()],
        }

    def _site_excursion_tracking_terms(
        self,
        *,
        forecasts: Sequence[Mapping[str, Any]],
        weights: Sequence[float],
        anchor: Mapping[str, Any] | None,
    ) -> dict[str, list[float]]:
        zero_terms = {
            "site_excursion_under_response_mean_by_site": [],
            "site_excursion_over_response_mean_by_site": [],
        }
        if anchor is None:
            return dict(zero_terms)
        if (
            "site_occupations_controller_next" not in anchor
            or "site_occupations_exact_next" not in anchor
            or any(
                ("site_occupations_controller_next" not in item)
                or ("site_occupations_exact_next" not in item)
                for item in forecasts
            )
        ):
            return dict(zero_terms)
        anchor_ctrl = np.asarray(anchor["site_occupations_controller_next"], dtype=float).reshape(-1)
        anchor_exact = np.asarray(anchor["site_occupations_exact_next"], dtype=float).reshape(-1)
        if anchor_ctrl.shape != anchor_exact.shape:
            return dict(zero_terms)
        site_count = int(anchor_ctrl.size)
        under_penalties: list[np.ndarray] = []
        over_penalties: list[np.ndarray] = []
        for item in forecasts:
            ctrl = np.asarray(item["site_occupations_controller_next"], dtype=float).reshape(-1)
            exact = np.asarray(item["site_occupations_exact_next"], dtype=float).reshape(-1)
            if ctrl.size != site_count or exact.size != site_count:
                return dict(zero_terms)
            ctrl_exc = ctrl - anchor_ctrl
            exact_exc = exact - anchor_exact
            target = np.abs(exact_exc)
            projected_ctrl = np.sign(exact_exc) * ctrl_exc
            mask = target > 1.0e-15
            under_penalties.append(
                np.where(mask, np.maximum(0.0, target - projected_ctrl), 0.0)
            )
            over_penalties.append(
                np.where(mask, np.maximum(0.0, projected_ctrl - target), 0.0)
            )
        weight_arr = np.asarray([float(x) for x in weights], dtype=float)
        weight_sum = float(np.sum(weight_arr))
        if weight_sum <= 0.0:
            return dict(zero_terms)
        under = np.asarray(under_penalties, dtype=float)
        over = np.asarray(over_penalties, dtype=float)
        under_mean = np.sum(under * weight_arr[:, None], axis=0) / weight_sum
        over_mean = np.sum(over * weight_arr[:, None], axis=0) / weight_sum
        return {
            "site_excursion_under_response_mean_by_site": [float(x) for x in under_mean.tolist()],
            "site_excursion_over_response_mean_by_site": [float(x) for x in over_mean.tolist()],
        }

    def _exact_forecast_rollout(
        self,
        *,
        time_stop: float,
        dt: float,
        executor: CompiledAnsatzExecutor,
        theta_runtime_start: np.ndarray | Sequence[float],
        theta_dot_step: np.ndarray | Sequence[float],
    ) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
        theta_runtime_base = np.asarray(theta_runtime_start, dtype=float).reshape(-1)
        theta_step = float(dt) * np.asarray(theta_dot_step, dtype=float).reshape(-1)
        horizon_steps = self._exact_forecast_horizon_length(time_stop=float(time_stop))
        forecasts: list[dict[str, Any]] = []
        for offset in range(int(horizon_steps)):
            theta_runtime = np.asarray(
                theta_runtime_base + float(offset) * np.asarray(theta_step, dtype=float),
                dtype=float,
            ).reshape(-1)
            forecast = self._exact_step_forecast(
                time_stop=float(time_stop) + float(offset) * float(dt),
                executor=executor,
                theta_runtime=theta_runtime,
            )
            forecasts.append(dict(forecast))
        horizon_weights = self._exact_forecast_tracking_horizon_weights(steps=len(forecasts))
        _slope_weight, curvature_weight = self._exact_forecast_energy_shape_weights()
        density_slope_weight = self._exact_forecast_density_slope_weight()
        density_curvature_weight = self._exact_forecast_density_curvature_weight()
        density_excursion_under_weight = self._exact_forecast_density_excursion_under_weight()
        density_excursion_over_weight = self._exact_forecast_density_excursion_over_weight()
        density_postcross_wrong_sign_weight = (
            self._exact_forecast_density_postcross_wrong_sign_weight()
        )
        excursion_under_weight = self._exact_forecast_energy_excursion_under_weight()
        excursion_over_weight = self._exact_forecast_energy_excursion_over_weight()
        d_shape_metrics_enabled = bool(self._exact_v1_d_shape_shadow_metrics_enabled())
        curvature_anchor: dict[str, Any] | None = None
        if (
            float(density_slope_weight) > 0.0
            or float(density_curvature_weight) > 0.0
            or float(density_excursion_under_weight) > 0.0
            or float(density_excursion_over_weight) > 0.0
            or float(density_postcross_wrong_sign_weight) > 0.0
            or (float(curvature_weight) > 0.0 and len(forecasts) >= 2)
            or float(excursion_under_weight) > 0.0
            or float(excursion_over_weight) > 0.0
            or bool(d_shape_metrics_enabled)
        ):
            theta_runtime_anchor = np.asarray(
                theta_runtime_base - np.asarray(theta_step, dtype=float),
                dtype=float,
            ).reshape(-1)
            curvature_anchor = dict(
                self._exact_step_forecast(
                    time_stop=float(time_stop) - float(dt),
                    executor=executor,
                    theta_runtime=theta_runtime_anchor,
                )
            )
        can_normalize = bool(
            all(
                ("primary_density_controller_next" in item)
                and ("primary_density_exact_next" in item)
                and ("doublon_exact_next" in item)
                and ("site_occupations_exact_next" in item)
                and ("energy_total_exact_next" in item)
                for item in forecasts
            )
            and (
                curvature_anchor is None
                or (
                    "primary_density_controller_next" in curvature_anchor
                    and "primary_density_exact_next" in curvature_anchor
                    and "doublon_exact_next" in curvature_anchor
                    and "site_occupations_exact_next" in curvature_anchor
                    and "energy_total_exact_next" in curvature_anchor
                )
            )
        )
        if can_normalize:
            forecasts, normalized_terms = self._exact_forecast_normalize_terms(
                forecasts=forecasts,
                weights=horizon_weights,
                anchor=curvature_anchor,
            )
        else:
            normalized_terms = {
                "primary_density_scale": 1.0,
                "doublon_scale": 1.0,
                "site_occupations_scale": 1.0,
                "energy_total_scale": 1.0,
                "primary_density_slope_scale": 1.0,
                "primary_density_slope_abs_error_mean": 0.0,
                "primary_density_slope_error_mean": 0.0,
                "abs_primary_density_slope_error_next": 0.0,
                "primary_density_slope_error_next": 0.0,
            }
        score = self._forecast_tracking_score(
            forecast=forecasts,
            curvature_anchor=curvature_anchor,
        )
        shape_terms = self._energy_shape_tracking_terms(
            forecasts=forecasts,
            weights=horizon_weights,
            curvature_anchor=curvature_anchor,
        )
        excursion_terms = self._energy_excursion_tracking_terms(
            forecasts=forecasts,
            weights=horizon_weights,
            anchor=curvature_anchor,
        )
        if bool(d_shape_metrics_enabled):
            d_shape_terms = self._d_shape_tracking_terms(
                forecasts=forecasts,
                weights=horizon_weights,
                curvature_anchor=curvature_anchor,
            )
            d_excursion_terms = self._d_excursion_tracking_terms(
                forecasts=forecasts,
                weights=horizon_weights,
                anchor=curvature_anchor,
            )
            total_occupation_terms = self._total_occupation_tracking_terms(
                forecasts=forecasts,
                weights=horizon_weights,
            )
        else:
            d_shape_terms = {
                "d_curvature_abs_error_mean": 0.0,
            }
            d_excursion_terms = {
                "d_excursion_under_response_mean": 0.0,
                "d_excursion_over_response_mean": 0.0,
            }
            total_occupation_terms = {
                "total_occupation_abs_error_next": 0.0,
                "total_occupation_abs_error_mean": 0.0,
            }
        site_shape_terms = self._site_shape_tracking_terms(
            forecasts=forecasts,
            weights=horizon_weights,
            curvature_anchor=curvature_anchor,
        )
        site_excursion_terms = self._site_excursion_tracking_terms(
            forecasts=forecasts,
            weights=horizon_weights,
            anchor=curvature_anchor,
        )
        slope_weight, curvature_weight = self._exact_forecast_energy_shape_weights()
        excursion_under_weight = self._exact_forecast_energy_excursion_under_weight()
        excursion_over_weight = self._exact_forecast_energy_excursion_over_weight()
        excursion_rel_tolerance = self._exact_forecast_energy_excursion_rel_tolerance()
        first = dict(forecasts[0])
        first["tracking_score_step1"] = float(self._forecast_tracking_score(forecast=first))
        first["tracking_score_horizon"] = float(score)
        first["tracking_horizon_steps_scored"] = int(len(forecasts))
        first["tracking_horizon_weights_used"] = [float(x) for x in horizon_weights]
        first["tracking_primary_density_scale"] = float(normalized_terms["primary_density_scale"])
        first["tracking_doublon_scale"] = float(normalized_terms["doublon_scale"])
        first["tracking_site_occupations_scale"] = float(normalized_terms["site_occupations_scale"])
        first["tracking_energy_total_scale"] = float(normalized_terms["energy_total_scale"])
        first["tracking_primary_density_slope_scale"] = float(normalized_terms["primary_density_slope_scale"])
        first["tracking_primary_density_slope_abs_error_mean"] = float(
            normalized_terms["primary_density_slope_abs_error_mean"]
        )
        first["tracking_primary_density_slope_error_mean"] = float(
            normalized_terms["primary_density_slope_error_mean"]
        )
        first["tracking_primary_density_slope_weight"] = float(density_slope_weight)
        first["tracking_d_curvature_weight"] = float(density_curvature_weight)
        first["tracking_d_excursion_under_weight"] = float(density_excursion_under_weight)
        first["tracking_d_excursion_over_weight"] = float(density_excursion_over_weight)
        first["tracking_primary_density_sign_lag_abs_error_mean"] = float(
            normalized_terms.get("primary_density_sign_lag_abs_error_mean", 0.0)
        )
        first["tracking_primary_density_sign_lag_error_mean"] = float(
            normalized_terms.get("primary_density_sign_lag_error_mean", 0.0)
        )
        first["tracking_primary_density_sign_lag_weight"] = float(
            self._exact_forecast_density_sign_lag_weight()
        )
        first["tracking_primary_density_postcross_wrong_sign_abs_error_mean"] = float(
            normalized_terms.get("primary_density_postcross_wrong_sign_abs_error_mean", 0.0)
        )
        first["tracking_primary_density_postcross_wrong_sign_error_mean"] = float(
            normalized_terms.get("primary_density_postcross_wrong_sign_error_mean", 0.0)
        )
        first["tracking_primary_density_postcross_wrong_sign_active"] = float(
            normalized_terms.get("primary_density_postcross_wrong_sign_active", 0.0)
        )
        first["tracking_primary_density_postcross_wrong_sign_weight"] = float(
            self._exact_forecast_density_postcross_wrong_sign_weight()
        )
        exact_turn_summary = self._exact_v1_primary_density_exact_turn_summary(
            forecasts=forecasts,
            primary_density_scale=float(normalized_terms.get("primary_density_scale", 1.0)),
        )
        first["tracking_primary_density_exact_abs_min_horizon"] = float(
            exact_turn_summary["tracking_primary_density_exact_abs_min_horizon"]
        )
        first["tracking_primary_density_exact_zero_crossed_horizon"] = float(
            exact_turn_summary["tracking_primary_density_exact_zero_crossed_horizon"]
        )
        harmonic_terms = self._drive_harmonic_tracking_terms(
            forecasts=forecasts,
            weights=horizon_weights,
        )
        first["tracking_drive_harmonic_mismatch"] = float(
            harmonic_terms["drive_harmonic_mismatch"]
        )
        first["tracking_drive_harmonic_ctrl_real"] = float(
            harmonic_terms["drive_harmonic_ctrl_real"]
        )
        first["tracking_drive_harmonic_ctrl_imag"] = float(
            harmonic_terms["drive_harmonic_ctrl_imag"]
        )
        first["tracking_drive_harmonic_exact_real"] = float(
            harmonic_terms["drive_harmonic_exact_real"]
        )
        first["tracking_drive_harmonic_exact_imag"] = float(
            harmonic_terms["drive_harmonic_exact_imag"]
        )
        first["tracking_drive_harmonic_weight"] = float(
            self._exact_forecast_drive_harmonic_weight()
        )
        first["tracking_energy_slope_abs_error_mean"] = float(
            shape_terms["energy_slope_abs_error_mean"]
        )
        first["tracking_energy_curvature_abs_error_mean"] = float(
            shape_terms["energy_curvature_abs_error_mean"]
        )
        first["tracking_energy_slope_weight"] = float(slope_weight)
        first["tracking_energy_curvature_weight"] = float(curvature_weight)
        first["tracking_energy_excursion_under_response_mean"] = float(
            excursion_terms["energy_excursion_under_response_mean"]
        )
        first["tracking_energy_excursion_under_weight"] = float(excursion_under_weight)
        first["tracking_energy_excursion_over_response_mean"] = float(
            excursion_terms["energy_excursion_over_response_mean"]
        )
        first["tracking_energy_excursion_over_weight"] = float(excursion_over_weight)
        first["tracking_energy_excursion_rel_tolerance"] = float(excursion_rel_tolerance)
        if bool(d_shape_metrics_enabled):
            first["tracking_d_curvature_abs_error_mean"] = float(
                d_shape_terms["d_curvature_abs_error_mean"]
            )
            first["tracking_d_excursion_under_response_mean"] = float(
                d_excursion_terms["d_excursion_under_response_mean"]
            )
            first["tracking_d_excursion_over_response_mean"] = float(
                d_excursion_terms["d_excursion_over_response_mean"]
            )
            first["tracking_total_occupation_abs_error_next"] = float(
                total_occupation_terms["total_occupation_abs_error_next"]
            )
            first["tracking_total_occupation_abs_error_mean"] = float(
                total_occupation_terms["total_occupation_abs_error_mean"]
            )
        first["tracking_site_slope_abs_error_mean_by_site"] = [
            float(x) for x in site_shape_terms["site_slope_abs_error_mean_by_site"]
        ]
        first["tracking_site_curvature_abs_error_mean_by_site"] = [
            float(x) for x in site_shape_terms["site_curvature_abs_error_mean_by_site"]
        ]
        first["tracking_site_excursion_under_response_mean_by_site"] = [
            float(x) for x in site_excursion_terms["site_excursion_under_response_mean_by_site"]
        ]
        first["tracking_site_excursion_over_response_mean_by_site"] = [
            float(x) for x in site_excursion_terms["site_excursion_over_response_mean_by_site"]
        ]
        return first, forecasts, float(score)

    def _exact_forecast_override_reason(
        self,
        *,
        stay_forecast: Mapping[str, Any],
        selected_forecast: Mapping[str, Any],
    ) -> str | None:
        mode = str(getattr(self.cfg, "exact_forecast_guardrail_mode", "off"))
        if mode == "off":
            return None
        fidelity_loss_tol = float(getattr(self.cfg, "exact_forecast_fidelity_loss_tol", 0.0))
        energy_increase_tol = float(
            getattr(self.cfg, "exact_forecast_abs_energy_error_increase_tol", 0.0)
        )
        fidelity_delta = float(selected_forecast["fidelity_exact_next"]) - float(
            stay_forecast["fidelity_exact_next"]
        )
        energy_error_delta = float(selected_forecast["abs_energy_total_error_next"]) - float(
            stay_forecast["abs_energy_total_error_next"]
        )
        if mode == "dual_metric_v1":
            if (
                fidelity_delta < -float(fidelity_loss_tol)
                and energy_error_delta > float(energy_increase_tol)
            ):
                return "exact_forecast_dual_metric_regression"
            return None
        if mode not in {"d_shape_barrier_v1", "fidelity_first_barrier_v1"}:
            return None
        regression_prefix = (
            "exact_forecast_fidelity_first"
            if mode == "fidelity_first_barrier_v1"
            else "exact_forecast_d_shape"
        )
        if fidelity_delta < -float(fidelity_loss_tol):
            return f"{regression_prefix}_fidelity_regression"
        if energy_error_delta > float(energy_increase_tol):
            return f"{regression_prefix}_energy_regression"
        total_occupation_tol = float(
            getattr(self.cfg, "exact_forecast_total_occupation_error_increase_tol", 0.0)
        )
        total_occ_keys = (
            "tracking_total_occupation_abs_error_next",
            "tracking_total_occupation_abs_error_mean",
        )
        for key in total_occ_keys:
            if key not in stay_forecast or key not in selected_forecast:
                continue
            stay_value = float(stay_forecast.get(key, float("nan")))
            selected_value = float(selected_forecast.get(key, float("nan")))
            if np.isfinite(stay_value) and np.isfinite(selected_value):
                if float(selected_value) - float(stay_value) > float(total_occupation_tol):
                    return f"{regression_prefix}_total_occupation_regression"
        return None

    def _forecast_tracking_score_generic(
        self,
        *,
        forecast: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        curvature_anchor: Mapping[str, Any] | None = None,
        include_energy_core_terms: bool = True,
    ) -> float:
        if isinstance(forecast, Mapping):
            if include_energy_core_terms and "tracking_score_horizon" in forecast:
                return float(forecast["tracking_score_horizon"])
            forecasts = [forecast]
        else:
            forecasts = [item for item in forecast]
        if len(forecasts) == 0:
            return float("inf")
        weights = self._exact_forecast_tracking_horizon_weights(steps=len(forecasts))
        total_weight = float(sum(float(x) for x in weights))
        if total_weight <= 0.0:
            return float("inf")
        (
            fidelity_defect_weight,
            primary_density_error_weight,
            doublon_error_weight,
            site_occupations_error_weight,
            energy_total_error_weight,
        ) = self._exact_forecast_tracking_error_weights()
        if not bool(include_energy_core_terms):
            energy_total_error_weight = 0.0
        score = 0.0
        for weight, item in zip(weights, forecasts):
            fidelity_defect = max(0.0, 1.0 - float(item["fidelity_exact_next"]))
            primary_density_error = self._optional_forecast_metric(
                item,
                normalized_key="normalized_primary_density_error_next",
                raw_key="abs_primary_density_error_next",
                raw_fallback_key="abs_staggered_error_next",
            )
            doublon_error = self._optional_forecast_metric(
                item,
                normalized_key="normalized_doublon_error_next",
                raw_key="abs_doublon_error_next",
            )
            site_occupations_error = self._optional_forecast_metric(
                item,
                normalized_key="normalized_site_occupations_abs_error_max_next",
                raw_key="site_occupations_abs_error_max_next",
            )
            energy_total_error = self._optional_forecast_metric(
                item,
                normalized_key="normalized_energy_total_error_next",
                raw_key="abs_energy_total_error_next",
            )
            score += float(weight) * float(
                float(fidelity_defect_weight) * fidelity_defect
                + self._weighted_optional_term(
                    primary_density_error,
                    primary_density_error_weight,
                )
                + self._weighted_optional_term(doublon_error, doublon_error_weight)
                + self._weighted_optional_term(
                    site_occupations_error,
                    site_occupations_error_weight,
                )
                + self._weighted_optional_term(
                    energy_total_error,
                    energy_total_error_weight,
                )
            )
        total = float(score / total_weight)
        density_slope_weight = self._exact_forecast_density_slope_weight()
        density_curvature_weight = self._exact_forecast_density_curvature_weight()
        density_excursion_under_weight = self._exact_forecast_density_excursion_under_weight()
        density_excursion_over_weight = self._exact_forecast_density_excursion_over_weight()
        density_sign_lag_weight = self._exact_forecast_density_sign_lag_weight()
        density_postcross_wrong_sign_weight = (
            self._exact_forecast_density_postcross_wrong_sign_weight()
        )
        normalized_terms_cache: dict[str, float] | None = None
        if (
            float(density_slope_weight) > 0.0
            or float(density_curvature_weight) > 0.0
            or float(density_excursion_under_weight) > 0.0
            or float(density_excursion_over_weight) > 0.0
            or float(density_sign_lag_weight) > 0.0
            or float(density_postcross_wrong_sign_weight) > 0.0
        ):
            stored_tracking_terms_available = bool(
                isinstance(forecast, Mapping)
                and any(
                    key in forecast
                    for key in (
                        "tracking_primary_density_slope_error_mean",
                        "tracking_d_curvature_abs_error_mean",
                        "tracking_d_excursion_under_response_mean",
                        "tracking_d_excursion_over_response_mean",
                        "tracking_primary_density_sign_lag_error_mean",
                        "tracking_primary_density_postcross_wrong_sign_error_mean",
                    )
                )
            )
            if stored_tracking_terms_available:
                density_slope_error_mean = float(
                    forecast.get("tracking_primary_density_slope_error_mean", 0.0)
                )
                density_curvature_error_mean = float(
                    forecast.get("tracking_d_curvature_abs_error_mean", 0.0)
                )
                density_excursion_under_error_mean = float(
                    forecast.get("tracking_d_excursion_under_response_mean", 0.0)
                )
                density_excursion_over_error_mean = float(
                    forecast.get("tracking_d_excursion_over_response_mean", 0.0)
                )
                if float(density_slope_weight) > 0.0:
                    total += float(density_slope_weight) * float(density_slope_error_mean)
                if float(density_curvature_weight) > 0.0:
                    total += float(density_curvature_weight) * float(density_curvature_error_mean)
                if float(density_excursion_under_weight) > 0.0:
                    total += float(density_excursion_under_weight) * float(
                        density_excursion_under_error_mean
                    )
                if float(density_excursion_over_weight) > 0.0:
                    total += float(density_excursion_over_weight) * float(
                        density_excursion_over_error_mean
                    )
                if float(density_sign_lag_weight) > 0.0:
                    density_sign_lag_error_mean = float(
                        forecast.get("tracking_primary_density_sign_lag_error_mean", 0.0)
                    )
                    total += float(density_sign_lag_weight) * float(density_sign_lag_error_mean)
                if float(density_postcross_wrong_sign_weight) > 0.0:
                    density_postcross_wrong_sign_error_mean = float(
                        forecast.get("tracking_primary_density_postcross_wrong_sign_error_mean", 0.0)
                    )
                    total += float(density_postcross_wrong_sign_weight) * float(
                        density_postcross_wrong_sign_error_mean
                    )
            elif all(
                ("primary_density_controller_next" in item)
                and ("primary_density_exact_next" in item)
                and ("doublon_exact_next" in item)
                and ("site_occupations_exact_next" in item)
                and ("energy_total_exact_next" in item)
                for item in forecasts
            ):
                _, normalized_terms_cache = self._exact_forecast_normalize_terms(
                    forecasts=forecasts,
                    weights=weights,
                    anchor=curvature_anchor,
                )
                density_slope_error_mean = float(normalized_terms_cache["primary_density_slope_error_mean"])
                if float(density_slope_weight) > 0.0:
                    total += float(density_slope_weight) * float(density_slope_error_mean)
                d_shape_terms = self._d_shape_tracking_terms(
                    forecasts=forecasts,
                    weights=weights,
                    curvature_anchor=curvature_anchor,
                )
                d_excursion_terms = self._d_excursion_tracking_terms(
                    forecasts=forecasts,
                    weights=weights,
                    anchor=curvature_anchor,
                )
                if float(density_curvature_weight) > 0.0:
                    total += float(density_curvature_weight) * float(
                        d_shape_terms["d_curvature_abs_error_mean"]
                    )
                if float(density_excursion_under_weight) > 0.0:
                    total += float(density_excursion_under_weight) * float(
                        d_excursion_terms["d_excursion_under_response_mean"]
                    )
                if float(density_excursion_over_weight) > 0.0:
                    total += float(density_excursion_over_weight) * float(
                        d_excursion_terms["d_excursion_over_response_mean"]
                    )
                if float(density_sign_lag_weight) > 0.0:
                    density_sign_lag_error_mean = float(
                        normalized_terms_cache["primary_density_sign_lag_error_mean"]
                    )
                    total += float(density_sign_lag_weight) * float(density_sign_lag_error_mean)
                if float(density_postcross_wrong_sign_weight) > 0.0:
                    density_postcross_wrong_sign_error_mean = float(
                        normalized_terms_cache["primary_density_postcross_wrong_sign_error_mean"]
                    )
                    total += float(density_postcross_wrong_sign_weight) * float(
                        density_postcross_wrong_sign_error_mean
                    )
        drive_harmonic_weight = self._exact_forecast_drive_harmonic_weight()
        if float(drive_harmonic_weight) > 0.0:
            if isinstance(forecast, Mapping) and "tracking_drive_harmonic_mismatch" in forecast:
                total += float(drive_harmonic_weight) * float(
                    forecast["tracking_drive_harmonic_mismatch"]
                )
            else:
                harmonic_terms = self._drive_harmonic_tracking_terms(
                    forecasts=forecasts,
                    weights=weights,
                )
                total += float(drive_harmonic_weight) * float(
                    harmonic_terms["drive_harmonic_mismatch"]
                )
        if bool(include_energy_core_terms):
            slope_weight, curvature_weight = self._exact_forecast_energy_shape_weights()
            if float(slope_weight) > 0.0 or float(curvature_weight) > 0.0:
                shape_terms = self._energy_shape_tracking_terms(
                    forecasts=forecasts,
                    weights=weights,
                    curvature_anchor=curvature_anchor,
                )
                total += float(slope_weight) * float(shape_terms["energy_slope_abs_error_mean"])
                total += float(curvature_weight) * float(
                    shape_terms["energy_curvature_abs_error_mean"]
                )
            excursion_under_weight = self._exact_forecast_energy_excursion_under_weight()
            excursion_over_weight = self._exact_forecast_energy_excursion_over_weight()
            if float(excursion_under_weight) > 0.0 or float(excursion_over_weight) > 0.0:
                excursion_terms = self._energy_excursion_tracking_terms(
                    forecasts=forecasts,
                    weights=weights,
                    anchor=curvature_anchor,
                )
                total += float(excursion_under_weight) * float(
                    excursion_terms["energy_excursion_under_response_mean"]
                )
                total += float(excursion_over_weight) * float(
                    excursion_terms["energy_excursion_over_response_mean"]
                )
        return float(total)

    def _forecast_tracking_score(
        self,
        *,
        forecast: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        curvature_anchor: Mapping[str, Any] | None = None,
    ) -> float:
        if self._exact_v1_d_shape_barrier_ranking_active():
            return float(
                self._exact_v1_live_d_score(
                    forecast=forecast,
                    curvature_anchor=curvature_anchor,
                )
            )
        if self._exact_v1_fidelity_first_barrier_ranking_active():
            return float(
                self._exact_v1_fidelity_first_score(
                    forecast=forecast,
                    curvature_anchor=curvature_anchor,
                )
            )
        return float(
            self._forecast_tracking_score_generic(
                forecast=forecast,
                curvature_anchor=curvature_anchor,
                include_energy_core_terms=True,
            )
        )

    def _exact_v1_paired_metric(
        self,
        stay: Mapping[str, Any],
        selected: Mapping[str, Any],
        *,
        normalized_key: str | None,
        raw_key: str,
        raw_fallback_key: str | None = None,
    ) -> tuple[float, float]:
        use_normalized = (
            normalized_key is not None and (normalized_key in stay) and (normalized_key in selected)
        )
        if use_normalized:
            return float(stay[normalized_key]), float(selected[normalized_key])
        stay_value = stay.get(raw_key)
        selected_value = selected.get(raw_key)
        if stay_value is None and raw_fallback_key is not None:
            stay_value = stay.get(raw_fallback_key)
        if selected_value is None and raw_fallback_key is not None:
            selected_value = selected.get(raw_fallback_key)
        return float(stay_value if stay_value is not None else float("inf")), float(
            selected_value if selected_value is not None else float("inf")
        )

    def _exact_v1_sign_lag_window_active(
        self,
        *,
        stay_forecast: Mapping[str, Any],
        selected_forecast: Mapping[str, Any],
    ) -> bool:
        if float(self._exact_forecast_density_sign_lag_weight()) <= 0.0:
            return False
        target_floor = self._exact_v1_sign_lag_window_target_gain_floor()
        if target_floor is None:
            return False
        activation = self._exact_v1_sign_lag_window_activation()
        if float(activation) <= 0.0:
            return False
        stay_sign_lag_error, selected_sign_lag_error = self._exact_v1_paired_metric(
            stay_forecast,
            selected_forecast,
            normalized_key="primary_density_sign_lag_next",
            raw_key="abs_primary_density_sign_lag_next",
        )
        return bool(max(float(stay_sign_lag_error), float(selected_sign_lag_error)) >= float(activation))

    def _exact_v1_turn_escape_density_failure_active(
        self,
        *,
        stay_forecast: Mapping[str, Any],
        selected_forecast: Mapping[str, Any],
    ) -> bool:
        if float(self._exact_forecast_density_sign_lag_weight()) <= 0.0:
            return False
        activation = self._exact_v1_sign_lag_window_activation()
        if float(activation) <= 0.0:
            return False
        stay_sign_lag_error, _selected_sign_lag_error = self._exact_v1_paired_metric(
            stay_forecast,
            selected_forecast,
            normalized_key="primary_density_sign_lag_next",
            raw_key="abs_primary_density_sign_lag_next",
        )
        stay_primary_density_error, selected_primary_density_error = self._exact_v1_paired_metric(
            stay_forecast,
            selected_forecast,
            normalized_key="normalized_primary_density_error_next",
            raw_key="abs_primary_density_error_next",
            raw_fallback_key="abs_staggered_error_next",
        )
        gain_floor = self._exact_v1_target_gain_floor(
            stay_forecast=stay_forecast,
            selected_forecast=selected_forecast,
            below_floor_probe=True,
        )
        return bool(
            float(stay_sign_lag_error) >= float(activation)
            and float(stay_primary_density_error)
            > float(selected_primary_density_error) + float(gain_floor)
        )

    def _exact_v1_d_shape_shadow_only_total(self, forecast: Mapping[str, Any]) -> float | None:
        total = 0.0
        for key in (
            "tracking_d_curvature_abs_error_mean",
            "tracking_d_excursion_under_response_mean",
            "tracking_d_excursion_over_response_mean",
        ):
            if key not in forecast:
                return None
            value = float(forecast.get(key, float("nan")))
            if not np.isfinite(value):
                return None
            total += float(value)
        return float(total)

    def _exact_v1_d_shape_turn_window_active(
        self,
        *,
        stay_forecast: Mapping[str, Any],
    ) -> bool:
        crossed = bool(
            float(stay_forecast.get("tracking_primary_density_exact_zero_crossed_horizon", 0.0))
            > 0.0
        )
        if bool(crossed):
            return True
        activation = self._exact_v1_d_shape_turn_window_abs_activation()
        if float(activation) <= 0.0:
            return False
        min_abs = float(
            stay_forecast.get("tracking_primary_density_exact_abs_min_horizon", float("inf"))
        )
        current_abs = abs(float(stay_forecast.get("primary_density_exact_next", float("nan"))))
        if (not np.isfinite(min_abs)) or (not np.isfinite(current_abs)):
            return False
        return bool(
            float(min_abs) <= float(activation)
            and float(min_abs) < float(current_abs) - 1.0e-12
        )

    def _exact_v1_d_shape_pre_turn_shadow_bridge_enabled(self) -> bool:
        return bool(getattr(self.cfg, "exact_v1_d_shape_pre_turn_shadow_bridge", False))

    def _exact_v1_d_shape_exact_horizon_moving_toward_turn(
        self,
        *,
        stay_forecast: Mapping[str, Any],
    ) -> bool:
        min_abs = float(
            stay_forecast.get("tracking_primary_density_exact_abs_min_horizon", float("inf"))
        )
        current_abs = abs(float(stay_forecast.get("primary_density_exact_next", float("nan"))))
        if (not np.isfinite(min_abs)) or (not np.isfinite(current_abs)):
            return False
        return bool(float(min_abs) < float(current_abs) - 1.0e-12)

    def _exact_v1_primary_density_exact_turn_summary(
        self,
        *,
        forecasts: Sequence[Mapping[str, Any]],
        primary_density_scale: float,
    ) -> dict[str, float]:
        if len(forecasts) == 0 or any("primary_density_exact_next" not in item for item in forecasts):
            return {
                "tracking_primary_density_exact_abs_min_horizon": float("inf"),
                "tracking_primary_density_exact_zero_crossed_horizon": 0.0,
            }
        exact_primary_density_values = np.asarray(
            [float(item["primary_density_exact_next"]) for item in forecasts],
            dtype=float,
        )
        exact_primary_density_abs_min_horizon = (
            float(np.min(np.abs(exact_primary_density_values)))
            if exact_primary_density_values.size and np.all(np.isfinite(exact_primary_density_values))
            else float("inf")
        )
        exact_sign_eps = max(2.0e-2, 0.1 * float(primary_density_scale))
        exact_signs = [
            int(self._primary_density_sign_bucket(float(value), eps=float(exact_sign_eps)))
            for value in exact_primary_density_values.tolist()
        ]
        exact_zero_crossed_horizon = any(
            int(lhs) != 0 and int(rhs) != 0 and int(lhs) != int(rhs)
            for lhs, rhs in zip(exact_signs[:-1], exact_signs[1:])
        )
        return {
            "tracking_primary_density_exact_abs_min_horizon": float(
                exact_primary_density_abs_min_horizon
            ),
            "tracking_primary_density_exact_zero_crossed_horizon": float(
                1.0 if bool(exact_zero_crossed_horizon) else 0.0
            ),
        }

    def _exact_v1_d_shape_barrier_score_mode_active(
        self,
        *,
        forecast: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    ) -> bool:
        if not self._exact_v1_d_shape_barrier_ranking_active():
            return False
        if isinstance(forecast, Mapping):
            if (
                "tracking_primary_density_exact_abs_min_horizon" in forecast
                or "tracking_primary_density_exact_zero_crossed_horizon" in forecast
            ):
                return bool(self._exact_v1_d_shape_turn_window_active(stay_forecast=forecast))
            forecasts = [forecast]
        else:
            forecasts = [item for item in forecast]
        if len(forecasts) == 0:
            return False
        turn_summary = self._exact_v1_primary_density_exact_turn_summary(
            forecasts=forecasts,
            primary_density_scale=float(
                self._exact_forecast_scale_floor("exact_forecast_primary_density_scale_floor")
            ),
        )
        representative = dict(forecasts[0])
        representative.update(turn_summary)
        return bool(self._exact_v1_d_shape_turn_window_active(stay_forecast=representative))

    def _exact_v1_live_d_shape_core_breakdown(
        self,
        *,
        forecast: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        curvature_anchor: Mapping[str, Any] | None = None,
    ) -> dict[str, float]:
        if isinstance(forecast, Mapping):
            forecasts = [forecast]
        else:
            forecasts = [item for item in forecast]
        if len(forecasts) == 0:
            return {
                "d_value_mean": float("inf"),
                "d_slope_mean": 0.0,
                "d_curvature_mean": 0.0,
                "d_excursion_under_mean": 0.0,
                "d_excursion_over_mean": 0.0,
                "total": float("inf"),
            }
        weights = self._exact_forecast_tracking_horizon_weights(steps=len(forecasts))
        total_weight = float(sum(float(x) for x in weights))
        if total_weight <= 0.0:
            return {
                "d_value_mean": float("inf"),
                "d_slope_mean": 0.0,
                "d_curvature_mean": 0.0,
                "d_excursion_under_mean": 0.0,
                "d_excursion_over_mean": 0.0,
                "total": float("inf"),
            }

        d_value_mean = 0.0
        for weight, item in zip(weights, forecasts):
            d_error = float(
                item.get(
                    "normalized_primary_density_error_next",
                    item.get("abs_primary_density_error_next", item.get("abs_staggered_error_next", 0.0)),
                )
            )
            if not np.isfinite(d_error):
                return {
                    "d_value_mean": float("inf"),
                    "d_slope_mean": 0.0,
                    "d_curvature_mean": 0.0,
                    "d_excursion_under_mean": 0.0,
                    "d_excursion_over_mean": 0.0,
                    "total": float("inf"),
                }
            d_value_mean += float(weight) * float(d_error)
        d_value_mean = float(d_value_mean / total_weight)

        slope_mean = None
        curvature_mean = None
        under_mean = None
        over_mean = None
        if isinstance(forecast, Mapping):
            slope_mean = float(
                forecast.get(
                    "tracking_primary_density_slope_error_mean",
                    forecast.get(
                        "primary_density_slope_error_next",
                        forecast.get("abs_primary_density_slope_error_next", 0.0),
                    ),
                )
            )
            curvature_mean = float(forecast.get("tracking_d_curvature_abs_error_mean", 0.0))
            under_mean = float(forecast.get("tracking_d_excursion_under_response_mean", 0.0))
            over_mean = float(forecast.get("tracking_d_excursion_over_response_mean", 0.0))

        if slope_mean is None or curvature_mean is None or under_mean is None or over_mean is None:
            can_normalize = bool(
                all(
                    ("primary_density_controller_next" in item)
                    and ("primary_density_exact_next" in item)
                    and ("doublon_exact_next" in item)
                    and ("site_occupations_exact_next" in item)
                    and ("energy_total_exact_next" in item)
                    for item in forecasts
                )
                and (
                    curvature_anchor is None
                    or (
                        "primary_density_controller_next" in curvature_anchor
                        and "primary_density_exact_next" in curvature_anchor
                        and "doublon_exact_next" in curvature_anchor
                        and "site_occupations_exact_next" in curvature_anchor
                        and "energy_total_exact_next" in curvature_anchor
                    )
                )
            )
            if can_normalize:
                _, normalized_terms = self._exact_forecast_normalize_terms(
                    forecasts=forecasts,
                    weights=weights,
                    anchor=curvature_anchor,
                )
                d_shape_terms = self._d_shape_tracking_terms(
                    forecasts=forecasts,
                    weights=weights,
                    curvature_anchor=curvature_anchor,
                )
                d_excursion_terms = self._d_excursion_tracking_terms(
                    forecasts=forecasts,
                    weights=weights,
                    anchor=curvature_anchor,
                )
                slope_mean = float(normalized_terms["primary_density_slope_error_mean"])
                curvature_mean = float(d_shape_terms["d_curvature_abs_error_mean"])
                under_mean = float(d_excursion_terms["d_excursion_under_response_mean"])
                over_mean = float(d_excursion_terms["d_excursion_over_response_mean"])

        slope_mean = 0.0 if slope_mean is None else float(slope_mean)
        curvature_mean = 0.0 if curvature_mean is None else float(curvature_mean)
        under_mean = 0.0 if under_mean is None else float(under_mean)
        over_mean = 0.0 if over_mean is None else float(over_mean)
        total = float(
            float(d_value_mean)
            + float(slope_mean)
            + 0.5 * float(curvature_mean)
            + float(under_mean)
            + 0.0 * float(over_mean)
        )
        return {
            "d_value_mean": float(d_value_mean),
            "d_slope_mean": float(slope_mean),
            "d_curvature_mean": float(curvature_mean),
            "d_excursion_under_mean": float(under_mean),
            "d_excursion_over_mean": float(over_mean),
            "total": float(total),
        }

    def _exact_v1_live_d_shape_core_score(
        self,
        *,
        forecast: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        curvature_anchor: Mapping[str, Any] | None = None,
    ) -> float:
        return float(
            self._exact_v1_live_d_shape_core_breakdown(
                forecast=forecast,
                curvature_anchor=curvature_anchor,
            )["total"]
        )

    def _exact_v1_fidelity_first_core_score(
        self,
        *,
        forecast: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        curvature_anchor: Mapping[str, Any] | None = None,
    ) -> float:
        return float(
            self._forecast_tracking_score_generic(
                forecast=forecast,
                curvature_anchor=curvature_anchor,
                include_energy_core_terms=False,
            )
        )

    def _exact_v1_guarded_turn_window_core_score(
        self,
        *,
        forecast: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        curvature_anchor: Mapping[str, Any] | None = None,
    ) -> float:
        if self._exact_v1_fidelity_first_barrier_ranking_active():
            return float(
                self._exact_v1_fidelity_first_core_score(
                    forecast=forecast,
                    curvature_anchor=curvature_anchor,
                )
            )
        return float(
            self._exact_v1_live_d_shape_core_score(
                forecast=forecast,
                curvature_anchor=curvature_anchor,
            )
        )

    def _exact_v1_live_d_score(
        self,
        *,
        forecast: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        curvature_anchor: Mapping[str, Any] | None = None,
    ) -> float:
        if isinstance(forecast, Mapping) and "tracking_score_horizon" in forecast:
            return float(forecast["tracking_score_horizon"])
        core_score = self._exact_v1_live_d_shape_core_score(
            forecast=forecast,
            curvature_anchor=curvature_anchor,
        )
        if not np.isfinite(core_score):
            return float(core_score)
        return float(
            float(core_score)
            + float(
                self._exact_v1_live_d_barrier_penalty(
                    forecast=forecast,
                    curvature_anchor=curvature_anchor,
                )
            )
        )

    def _exact_v1_fidelity_first_score(
        self,
        *,
        forecast: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        curvature_anchor: Mapping[str, Any] | None = None,
    ) -> float:
        if isinstance(forecast, Mapping) and "tracking_score_horizon" in forecast:
            return float(forecast["tracking_score_horizon"])
        core_score = self._exact_v1_fidelity_first_core_score(
            forecast=forecast,
            curvature_anchor=curvature_anchor,
        )
        if not np.isfinite(core_score):
            return float(core_score)
        return float(
            float(core_score)
            + float(
                self._exact_v1_live_d_barrier_penalty(
                    forecast=forecast,
                    curvature_anchor=curvature_anchor,
                )
            )
        )

    def _exact_v1_live_d_barrier_breakdown(
        self,
        *,
        forecast: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        curvature_anchor: Mapping[str, Any] | None = None,
    ) -> dict[str, float]:
        if isinstance(forecast, Mapping):
            forecasts = [forecast]
            first = dict(forecast)
        else:
            forecasts = [item for item in forecast]
            if len(forecasts) == 0:
                return {
                    "energy_total_penalty": 0.0,
                    "energy_slope_penalty": 0.0,
                    "energy_curvature_penalty": 0.0,
                    "energy_excursion_under_penalty": 0.0,
                    "energy_excursion_over_penalty": 0.0,
                    "total_occupation_next_penalty": 0.0,
                    "total_occupation_mean_penalty": 0.0,
                    "total": 0.0,
                }
            first = dict(forecasts[0])

        weights = self._exact_forecast_tracking_horizon_weights(steps=len(forecasts))
        if (
            "tracking_total_occupation_abs_error_next" not in first
            or "tracking_total_occupation_abs_error_mean" not in first
        ):
            total_occupation_terms = self._total_occupation_tracking_terms(
                forecasts=forecasts,
                weights=weights,
            )
            first.setdefault(
                "tracking_total_occupation_abs_error_next",
                float(total_occupation_terms["total_occupation_abs_error_next"]),
            )
            first.setdefault(
                "tracking_total_occupation_abs_error_mean",
                float(total_occupation_terms["total_occupation_abs_error_mean"]),
            )

        can_compute_energy_terms = bool(
            all(
                ("energy_total_controller_next" in item)
                and ("energy_total_exact_next" in item)
                for item in forecasts
            )
            and (
                curvature_anchor is None
                or (
                    "energy_total_controller_next" in curvature_anchor
                    and "energy_total_exact_next" in curvature_anchor
                )
            )
        )
        if can_compute_energy_terms and any(
            key not in first
            for key in (
                "tracking_energy_slope_abs_error_mean",
                "tracking_energy_curvature_abs_error_mean",
                "tracking_energy_excursion_under_response_mean",
                "tracking_energy_excursion_over_response_mean",
            )
        ):
            energy_shape_terms = self._energy_shape_tracking_terms(
                forecasts=forecasts,
                weights=weights,
                curvature_anchor=curvature_anchor,
            )
            energy_excursion_terms = self._energy_excursion_tracking_terms(
                forecasts=forecasts,
                weights=weights,
                anchor=curvature_anchor,
            )
            first.setdefault(
                "tracking_energy_slope_abs_error_mean",
                float(energy_shape_terms["energy_slope_abs_error_mean"]),
            )
            first.setdefault(
                "tracking_energy_curvature_abs_error_mean",
                float(energy_shape_terms["energy_curvature_abs_error_mean"]),
            )
            first.setdefault(
                "tracking_energy_excursion_under_response_mean",
                float(energy_excursion_terms["energy_excursion_under_response_mean"]),
            )
            first.setdefault(
                "tracking_energy_excursion_over_response_mean",
                float(energy_excursion_terms["energy_excursion_over_response_mean"]),
            )

        energy_total_error = float(first.get("abs_energy_total_error_next", 0.0))
        energy_slope_mean = float(first.get("tracking_energy_slope_abs_error_mean", 0.0))
        energy_curvature_mean = float(first.get("tracking_energy_curvature_abs_error_mean", 0.0))
        energy_excursion_under_mean = float(
            first.get("tracking_energy_excursion_under_response_mean", 0.0)
        )
        energy_excursion_over_mean = float(
            first.get("tracking_energy_excursion_over_response_mean", 0.0)
        )
        total_occ_next = float(first.get("tracking_total_occupation_abs_error_next", 0.0))
        total_occ_mean = float(first.get("tracking_total_occupation_abs_error_mean", 0.0))

        def _hinge_sq(value: float, cap: float) -> float:
            if not np.isfinite(value):
                return float("inf")
            if float(cap) <= 0.0:
                return float(value > 0.0)
            excess = max(0.0, float(value) - float(cap))
            return float((excess / float(cap)) ** 2)

        energy_total_cap = 1.8e-1
        energy_slope_cap = 6.0e-2
        energy_curvature_cap = 8.0e-2
        energy_excursion_cap = 8.0e-2
        total_occ_next_cap = 3.0e-2
        total_occ_mean_cap = 3.0e-2

        breakdown = {
            "energy_total_penalty": 6.0 * _hinge_sq(float(energy_total_error), float(energy_total_cap)),
            "energy_slope_penalty": 4.0 * _hinge_sq(float(energy_slope_mean), float(energy_slope_cap)),
            "energy_curvature_penalty": 2.0
            * _hinge_sq(float(energy_curvature_mean), float(energy_curvature_cap)),
            "energy_excursion_under_penalty": 4.0
            * _hinge_sq(float(energy_excursion_under_mean), float(energy_excursion_cap)),
            "energy_excursion_over_penalty": 4.0
            * _hinge_sq(float(energy_excursion_over_mean), float(energy_excursion_cap)),
            "total_occupation_next_penalty": 5.0
            * _hinge_sq(float(total_occ_next), float(total_occ_next_cap)),
            "total_occupation_mean_penalty": 5.0
            * _hinge_sq(float(total_occ_mean), float(total_occ_mean_cap)),
        }
        breakdown["total"] = float(sum(float(value) for value in breakdown.values()))
        return {str(key): float(value) for key, value in breakdown.items()}

    def _exact_v1_live_d_barrier_penalty(
        self,
        *,
        forecast: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        curvature_anchor: Mapping[str, Any] | None = None,
    ) -> float:
        return float(
            self._exact_v1_live_d_barrier_breakdown(
                forecast=forecast,
                curvature_anchor=curvature_anchor,
            )["total"]
        )

    def _exact_v1_live_d_score_breakdown(
        self,
        *,
        forecast: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        curvature_anchor: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        core = self._exact_v1_live_d_shape_core_breakdown(
            forecast=forecast,
            curvature_anchor=curvature_anchor,
        )
        barrier = self._exact_v1_live_d_barrier_breakdown(
            forecast=forecast,
            curvature_anchor=curvature_anchor,
        )
        core_total = float(core["total"])
        total = float("inf") if not np.isfinite(core_total) else float(core_total) + float(barrier["total"])
        return {
            "core": dict(core),
            "barrier": dict(barrier),
            "total": float(total),
        }

    def _exact_v1_d_shape_turn_window_target_win_result(
        self,
        *,
        stay_forecast: Mapping[str, Any],
        selected_forecast: Mapping[str, Any],
    ) -> tuple[bool, str | None]:
        if not self._exact_v1_guarded_turn_window_ranking_active():
            return False, None
        turn_window_active = self._exact_v1_d_shape_turn_window_active(stay_forecast=stay_forecast)
        pre_turn_bridge_active = bool(
            (not turn_window_active)
            and self._exact_v1_d_shape_pre_turn_shadow_bridge_enabled()
            and self._exact_v1_d_shape_exact_horizon_moving_toward_turn(
                stay_forecast=stay_forecast
            )
        )
        if not turn_window_active and not pre_turn_bridge_active:
            return False, "outside_exact_turn_window"
        stay_score = float(self._exact_v1_guarded_turn_window_core_score(forecast=stay_forecast))
        selected_score = float(
            self._exact_v1_guarded_turn_window_core_score(forecast=selected_forecast)
        )
        if not np.isfinite(stay_score) or not np.isfinite(selected_score):
            return False, None
        if float(selected_score) >= float(stay_score) - 1.0e-12:
            fidelity_first_turn_local_win, _ = self._exact_v1_fidelity_first_turn_local_target_win_result(
                stay_forecast=stay_forecast,
                selected_forecast=selected_forecast,
                below_floor_probe=False,
            )
            if not fidelity_first_turn_local_win:
                return False, "no_tracking_win_vs_stay"
        stay_shadow_total = self._exact_v1_d_shape_shadow_only_total(stay_forecast)
        selected_shadow_total = self._exact_v1_d_shape_shadow_only_total(selected_forecast)
        if stay_shadow_total is None or selected_shadow_total is None:
            return False, "missing_shadow_turn_signal"
        if float(selected_shadow_total) >= float(stay_shadow_total) - 1.0e-12:
            return False, "no_shadow_turn_win_vs_stay"
        barrier_reason = self._exact_forecast_override_reason(
            stay_forecast=stay_forecast,
            selected_forecast=selected_forecast,
        )
        if barrier_reason is not None:
            return False, str(barrier_reason)
        if bool(pre_turn_bridge_active):
            return True, self._exact_v1_guarded_pre_turn_shadow_bridge_reason()
        return True, None

    def _exact_v1_d_shape_barrier_protected_horizon_result(
        self,
        *,
        stay_forecast: Mapping[str, Any],
        selected_forecast: Mapping[str, Any],
    ) -> tuple[bool, str | None]:
        return self._exact_v1_d_shape_turn_window_target_win_result(
            stay_forecast=stay_forecast,
            selected_forecast=selected_forecast,
        )

    def _exact_v1_fidelity_first_turn_local_target_win_result(
        self,
        *,
        stay_forecast: Mapping[str, Any],
        selected_forecast: Mapping[str, Any],
        below_floor_probe: bool = False,
    ) -> tuple[bool, str | None]:
        if not self._exact_v1_fidelity_first_barrier_ranking_active():
            return False, None
        turn_window_active = self._exact_v1_d_shape_turn_window_active(stay_forecast=stay_forecast)
        pre_turn_bridge_active = bool(
            (not turn_window_active)
            and self._exact_v1_d_shape_pre_turn_shadow_bridge_enabled()
            and self._exact_v1_d_shape_exact_horizon_moving_toward_turn(
                stay_forecast=stay_forecast
            )
        )
        if not turn_window_active and not pre_turn_bridge_active:
            return False, "outside_exact_turn_window"
        target_gain_floor = self._exact_v1_target_gain_floor(
            stay_forecast=stay_forecast,
            selected_forecast=selected_forecast,
            below_floor_probe=bool(below_floor_probe),
        )
        stay_shadow_total = self._exact_v1_d_shape_shadow_only_total(stay_forecast)
        selected_shadow_total = self._exact_v1_d_shape_shadow_only_total(selected_forecast)
        stay_site_turn_total = self._exact_v1_site_turn_error_total(stay_forecast)
        selected_site_turn_total = self._exact_v1_site_turn_error_total(selected_forecast)
        shadow_gain = (
            None
            if stay_shadow_total is None or selected_shadow_total is None
            else float(stay_shadow_total) - float(selected_shadow_total)
        )
        site_turn_gain = (
            None
            if stay_site_turn_total is None or selected_site_turn_total is None
            else float(stay_site_turn_total) - float(selected_site_turn_total)
        )
        shadow_win = bool(shadow_gain is not None and float(shadow_gain) > float(target_gain_floor))
        site_turn_win = bool(site_turn_gain is not None and float(site_turn_gain) > float(target_gain_floor))
        onset_combined_win = False
        if not (shadow_win or site_turn_win) and not bool(below_floor_probe):
            onset_gain_floor = self._exact_v1_fidelity_first_turn_local_onset_gain_floor()
            onset_combined_win = bool(
                shadow_gain is not None
                and site_turn_gain is not None
                and float(shadow_gain) > float(onset_gain_floor)
                and float(site_turn_gain) > float(onset_gain_floor)
                and not self._exact_v1_sign_lag_window_active(
                    stay_forecast=stay_forecast,
                    selected_forecast=selected_forecast,
                )
                and not self._exact_v1_postcross_wrong_sign_window_active(
                    stay_forecast=stay_forecast,
                    selected_forecast=selected_forecast,
                )
            )
        if not (shadow_win or site_turn_win or onset_combined_win):
            return False, "no_turn_local_target_win_vs_stay"
        if bool(pre_turn_bridge_active):
            return True, self._exact_v1_guarded_pre_turn_shadow_bridge_reason()
        return True, None

    def _exact_v1_total_occupation_not_worse(
        self,
        *,
        stay_forecast: Mapping[str, Any],
        selected_forecast: Mapping[str, Any],
    ) -> bool:
        for key in (
            "tracking_total_occupation_abs_error_next",
            "tracking_total_occupation_abs_error_mean",
        ):
            if key not in stay_forecast or key not in selected_forecast:
                return False
            stay_value = float(stay_forecast.get(key, float("nan")))
            selected_value = float(selected_forecast.get(key, float("nan")))
            if not np.isfinite(stay_value) or not np.isfinite(selected_value):
                return False
            if float(selected_value) > float(stay_value) + 1.0e-12:
                return False
        return True

    def _exact_v1_below_floor_d_shape_escape_active(
        self,
        *,
        stay_forecast: Mapping[str, Any],
        selected_forecast: Mapping[str, Any],
    ) -> bool:
        if not self._exact_v1_below_floor_energy_safe_d_shape_escape_enabled():
            return False
        stay_d_shape_total = self._exact_v1_d_shape_shadow_only_total(stay_forecast)
        selected_d_shape_total = self._exact_v1_d_shape_shadow_only_total(selected_forecast)
        if stay_d_shape_total is None or selected_d_shape_total is None:
            return False
        stay_score = float(self._forecast_tracking_score(forecast=stay_forecast))
        selected_score = float(self._forecast_tracking_score(forecast=selected_forecast))
        stay_next_energy_error = float(stay_forecast.get("abs_energy_total_error_next", float("nan")))
        selected_next_energy_error = float(
            selected_forecast.get("abs_energy_total_error_next", float("nan"))
        )
        if not all(
            np.isfinite(value)
            for value in (
                stay_score,
                selected_score,
                stay_next_energy_error,
                selected_next_energy_error,
            )
        ):
            return False
        return bool(
            float(selected_score) < float(stay_score) - 1.0e-12
            and float(selected_d_shape_total) < float(stay_d_shape_total) - 1.0e-12
            and float(selected_next_energy_error) <= float(stay_next_energy_error) + 1.0e-12
            and self._exact_v1_total_occupation_not_worse(
                stay_forecast=stay_forecast,
                selected_forecast=selected_forecast,
            )
        )

    def _exact_v1_postcross_wrong_sign_window_active(
        self,
        *,
        stay_forecast: Mapping[str, Any],
        selected_forecast: Mapping[str, Any],
    ) -> bool:
        target_floor = self._exact_v1_postcross_wrong_sign_target_gain_floor()
        if target_floor is None:
            return False
        activation = self._exact_v1_postcross_wrong_sign_activation()
        if float(activation) <= 0.0:
            return False
        stay_active = float(stay_forecast.get("tracking_primary_density_postcross_wrong_sign_active", 0.0))
        selected_active = float(selected_forecast.get("tracking_primary_density_postcross_wrong_sign_active", 0.0))
        if max(float(stay_active), float(selected_active)) <= 0.0:
            return False
        stay_debt, selected_debt = self._exact_v1_paired_metric(
            stay_forecast,
            selected_forecast,
            normalized_key="tracking_primary_density_postcross_wrong_sign_error_mean",
            raw_key="tracking_primary_density_postcross_wrong_sign_abs_error_mean",
        )
        if max(float(stay_debt), float(selected_debt)) < float(activation):
            return False
        return bool(float(selected_debt) + 1.0e-12 < float(stay_debt))

    def _exact_v1_target_gain_floor(
        self,
        *,
        stay_forecast: Mapping[str, Any],
        selected_forecast: Mapping[str, Any],
        below_floor_probe: bool,
    ) -> float:
        floor_value = (
            self._exact_v1_below_floor_probe_target_gain_floor()
            if bool(below_floor_probe)
            else self._exact_v1_density_first_target_gain_floor()
        )
        if self._exact_v1_sign_lag_window_active(
            stay_forecast=stay_forecast,
            selected_forecast=selected_forecast,
        ):
            lowered_floor = self._exact_v1_sign_lag_window_target_gain_floor()
            if lowered_floor is not None:
                floor_value = min(float(floor_value), float(lowered_floor))
        if self._exact_v1_postcross_wrong_sign_window_active(
            stay_forecast=stay_forecast,
            selected_forecast=selected_forecast,
        ):
            lowered_floor = self._exact_v1_postcross_wrong_sign_target_gain_floor()
            if lowered_floor is not None:
                floor_value = min(float(floor_value), float(lowered_floor))
        return float(floor_value)

    def _exact_v1_componentwise_aspiration_result(
        self,
        *,
        stay_forecast: Mapping[str, Any],
        selected_forecast: Mapping[str, Any],
        below_floor_probe: bool = False,
    ) -> tuple[bool, str | None]:
        if self._exact_v1_guarded_turn_window_ranking_active():
            stay_score = float(self._exact_v1_guarded_commit_compare_score(forecast=stay_forecast))
            selected_score = float(
                self._exact_v1_guarded_commit_compare_score(forecast=selected_forecast)
            )
            guarded_target_win = bool(
                np.isfinite(stay_score)
                and np.isfinite(selected_score)
                and float(selected_score) < float(stay_score) - 1.0e-12
            )
            if not guarded_target_win:
                fidelity_first_turn_local_win, _ = self._exact_v1_fidelity_first_turn_local_target_win_result(
                    stay_forecast=stay_forecast,
                    selected_forecast=selected_forecast,
                    below_floor_probe=bool(below_floor_probe),
                )
                if not fidelity_first_turn_local_win:
                    return False, "no_target_win_vs_stay"
            if bool(below_floor_probe):
                energy_window_ok, energy_window_reason = self._exact_v1_below_floor_energy_safe_window(
                    stay_forecast=stay_forecast,
                    selected_forecast=selected_forecast,
                )
                if not energy_window_ok:
                    return False, str(energy_window_reason or "outside_energy_safe_window")
            barrier_reason = self._exact_forecast_override_reason(
                stay_forecast=stay_forecast,
                selected_forecast=selected_forecast,
            )
            if barrier_reason is not None:
                return False, str(barrier_reason)
            return True, None
        stay_primary_density_error, selected_primary_density_error = self._exact_v1_paired_metric(
            stay_forecast,
            selected_forecast,
            normalized_key="normalized_primary_density_error_next",
            raw_key="abs_primary_density_error_next",
            raw_fallback_key="abs_staggered_error_next",
        )
        stay_primary_density_slope_error, selected_primary_density_slope_error = self._exact_v1_paired_metric(
            stay_forecast,
            selected_forecast,
            normalized_key="primary_density_slope_error_next",
            raw_key="abs_primary_density_slope_error_next",
        )
        stay_site_error, selected_site_error = self._exact_v1_paired_metric(
            stay_forecast,
            selected_forecast,
            normalized_key=None,
            raw_key="site_occupations_abs_error_max_next",
        )
        stay_energy_error, selected_energy_error = self._exact_v1_paired_metric(
            stay_forecast,
            selected_forecast,
            normalized_key="normalized_energy_total_error_next",
            raw_key="abs_energy_total_error_next",
        )
        stay_fidelity_defect = max(0.0, 1.0 - float(stay_forecast["fidelity_exact_next"]))
        selected_fidelity_defect = max(0.0, 1.0 - float(selected_forecast["fidelity_exact_next"]))

        primary_density_gain = float(stay_primary_density_error) - float(selected_primary_density_error)
        primary_density_slope_gain = float(stay_primary_density_slope_error) - float(
            selected_primary_density_slope_error
        )
        stay_sign_lag_error, selected_sign_lag_error = self._exact_v1_paired_metric(
            stay_forecast,
            selected_forecast,
            normalized_key=(
                "primary_density_sign_lag_next"
                if float(self._exact_forecast_density_sign_lag_weight()) > 0.0
                else None
            ),
            raw_key="abs_primary_density_sign_lag_next",
        )
        sign_lag_gain = float(stay_sign_lag_error) - float(selected_sign_lag_error)
        site_gain = float(stay_site_error) - float(selected_site_error)
        target_gain_floor = self._exact_v1_target_gain_floor(
            stay_forecast=stay_forecast,
            selected_forecast=selected_forecast,
            below_floor_probe=bool(below_floor_probe),
        )
        material_density_gain = float(primary_density_gain) > float(target_gain_floor)
        material_density_slope_gain = float(primary_density_slope_gain) > float(target_gain_floor)
        material_sign_lag_gain = float(sign_lag_gain) > float(target_gain_floor)
        material_site_gain = float(site_gain) > float(target_gain_floor)
        d_shape_turn_target_win = False
        if not (
            material_density_gain
            or material_density_slope_gain
            or material_sign_lag_gain
            or material_site_gain
        ):
            d_shape_turn_target_win, _d_shape_turn_target_reason = (
                self._exact_v1_d_shape_turn_window_target_win_result(
                    stay_forecast=stay_forecast,
                    selected_forecast=selected_forecast,
                )
            )
            if not bool(d_shape_turn_target_win):
                return False, "no_target_win_vs_stay"

        if bool(below_floor_probe):
            energy_window_ok, energy_window_reason = self._exact_v1_below_floor_energy_safe_window(
                stay_forecast=stay_forecast,
                selected_forecast=selected_forecast,
            )
            if not energy_window_ok:
                return False, str(energy_window_reason or "outside_energy_safe_window")

        fidelity_regression = float(selected_fidelity_defect) - float(stay_fidelity_defect)
        energy_regression = float(selected_energy_error) - float(stay_energy_error)
        site_regression = float(selected_site_error) - float(stay_site_error)
        fidelity_cap = 1.0e-2 if bool(below_floor_probe) else 5.0e-2
        energy_cap = 5.0e-2 if bool(below_floor_probe) else 2.5e-1
        site_cap = 1.0e-2 if bool(below_floor_probe) else 2.0e-2
        dual_fidelity_cap = 5.0e-3 if bool(below_floor_probe) else 1.0e-2
        dual_energy_cap = 2.0e-2 if bool(below_floor_probe) else 5.0e-2
        if float(fidelity_regression) > float(fidelity_cap):
            return False, "fails_fidelity_cap"
        if float(energy_regression) > float(energy_cap):
            return False, "fails_energy_cap"
        if float(site_regression) > float(site_cap):
            return False, "fails_site_cap"
        if (
            float(fidelity_regression) > float(dual_fidelity_cap)
            and float(energy_regression) > float(dual_energy_cap)
            and not material_density_slope_gain
        ):
            return False, "fails_dual_regression_without_slope_win"
        return True, None

    def _exact_v1_fidelity_first_turn_local_target_admission_reason(
        self,
        *,
        stay_forecast: Mapping[str, Any],
        selected_forecast: Mapping[str, Any],
        below_floor_probe: bool = False,
    ) -> str | None:
        if not self._exact_v1_fidelity_first_barrier_ranking_active():
            return None
        stay_score = float(self._forecast_tracking_score(forecast=stay_forecast))
        selected_score = float(self._forecast_tracking_score(forecast=selected_forecast))
        if (
            not np.isfinite(stay_score)
            or not np.isfinite(selected_score)
            or float(selected_score) < float(stay_score) - 1.0e-12
        ):
            return None
        fidelity_first_turn_local_win, _ = self._exact_v1_fidelity_first_turn_local_target_win_result(
            stay_forecast=stay_forecast,
            selected_forecast=selected_forecast,
            below_floor_probe=bool(below_floor_probe),
        )
        if not fidelity_first_turn_local_win:
            return None
        return self._exact_v1_fidelity_first_turn_local_target_win_reason()

    def _exact_v1_nonimproving_score_allows_density_first_append(
        self,
        *,
        stay_forecast: Mapping[str, Any],
        selected_forecast: Mapping[str, Any],
    ) -> bool:
        if self._exact_v1_d_shape_barrier_ranking_active():
            return False
        sign_lag_enabled = float(self._exact_forecast_density_sign_lag_weight()) > 0.0
        stay_primary_density_error, selected_primary_density_error = self._exact_v1_paired_metric(
            stay_forecast,
            selected_forecast,
            normalized_key="normalized_primary_density_error_next",
            raw_key="abs_primary_density_error_next",
            raw_fallback_key="abs_staggered_error_next",
        )
        stay_primary_density_slope_error, selected_primary_density_slope_error = self._exact_v1_paired_metric(
            stay_forecast,
            selected_forecast,
            normalized_key="primary_density_slope_error_next",
            raw_key="abs_primary_density_slope_error_next",
        )
        stay_energy_error, selected_energy_error = self._exact_v1_paired_metric(
            stay_forecast,
            selected_forecast,
            normalized_key="normalized_energy_total_error_next",
            raw_key="abs_energy_total_error_next",
        )
        stay_site_error, selected_site_error = self._exact_v1_paired_metric(
            stay_forecast,
            selected_forecast,
            normalized_key=(
                "normalized_site_occupations_abs_error_max_next" if bool(sign_lag_enabled) else None
            ),
            raw_key="site_occupations_abs_error_max_next",
        )
        stay_sign_lag_error, selected_sign_lag_error = self._exact_v1_paired_metric(
            stay_forecast,
            selected_forecast,
            normalized_key=("primary_density_sign_lag_next" if bool(sign_lag_enabled) else None),
            raw_key="abs_primary_density_sign_lag_next",
        )
        stay_fidelity_defect = max(0.0, 1.0 - float(stay_forecast["fidelity_exact_next"]))
        selected_fidelity_defect = max(0.0, 1.0 - float(selected_forecast["fidelity_exact_next"]))
        primary_density_gain = float(stay_primary_density_error) - float(selected_primary_density_error)
        primary_density_slope_gain = float(stay_primary_density_slope_error) - float(
            selected_primary_density_slope_error
        )
        sign_lag_gain = float(stay_sign_lag_error) - float(selected_sign_lag_error)
        site_gain = float(stay_site_error) - float(selected_site_error)
        target_gain_floor = self._exact_v1_target_gain_floor(
            stay_forecast=stay_forecast,
            selected_forecast=selected_forecast,
            below_floor_probe=False,
        )
        material_density_gain = float(primary_density_gain) > float(target_gain_floor)
        material_density_slope_gain = float(primary_density_slope_gain) > float(target_gain_floor)
        material_sign_lag_gain = bool(sign_lag_enabled) and float(sign_lag_gain) > float(target_gain_floor)
        material_site_gain = bool(sign_lag_enabled) and float(site_gain) > float(target_gain_floor)
        if not (
            material_density_gain
            or material_density_slope_gain
            or material_sign_lag_gain
            or material_site_gain
        ):
            return False
        fidelity_regression = float(selected_fidelity_defect) - float(stay_fidelity_defect)
        energy_regression = float(selected_energy_error) - float(stay_energy_error)
        site_regression = float(selected_site_error) - float(stay_site_error)
        if (
            float(fidelity_regression) > 5.0e-2
            or float(energy_regression) > 2.5e-1
            or float(site_regression) > 2.0e-2
        ):
            return False
        if (
            float(fidelity_regression) > 1.0e-2
            and float(energy_regression) > 5.0e-2
            and not (material_density_slope_gain or material_sign_lag_gain)
        ):
            return False
        return True

    def _stay_forecast_within_exact_v1_bounded_defect(
        self,
        *,
        forecast: Mapping[str, Any],
    ) -> bool:
        if self._exact_v1_d_shape_barrier_ranking_active():
            return False
        fidelity_defect = max(0.0, 1.0 - float(forecast["fidelity_exact_next"]))
        primary_density_error = self._optional_forecast_metric(
            forecast,
            normalized_key=None,
            raw_key="abs_primary_density_error_next",
            raw_fallback_key="abs_staggered_error_next",
        )
        slope_required = bool(
            ("abs_primary_density_slope_error_next" in forecast)
            or float(self._exact_forecast_density_slope_weight()) > 0.0
        )
        primary_density_slope_error = self._finite_float_or_none(
            forecast.get("abs_primary_density_slope_error_next", None)
        )
        doublon_error = self._finite_float_or_none(forecast.get("abs_doublon_error_next", None))
        site_error = self._finite_float_or_none(
            forecast.get("site_occupations_abs_error_max_next", None)
        )
        energy_error = self._finite_float_or_none(forecast.get("abs_energy_total_error_next", None))
        return bool(
            fidelity_defect <= 1.0e-3
            and primary_density_error is not None
            and float(primary_density_error) <= 2.0e-2
            and (
                not slope_required
                or (
                    primary_density_slope_error is not None
                    and float(primary_density_slope_error) <= 2.0e-2
                )
            )
            and (doublon_error is None or float(doublon_error) <= 2.0e-3)
            and site_error is not None
            and float(site_error) <= 2.0e-2
            and energy_error is not None
            and float(energy_error) <= 2.0e-3
        )

    def _exact_v1_forecast_override_reason(
        self,
        *,
        stay_forecast: Mapping[str, Any],
        selected_forecast: Mapping[str, Any],
        action_kind: str,
        selected: Mapping[str, Any] | None,
    ) -> str | None:
        if str(self.cfg.mode) != "exact_v1":
            return None
        if str(action_kind) != "append_candidate" or selected is None:
            return None
        if self._stay_forecast_within_exact_v1_bounded_defect(forecast=stay_forecast):
            return "exact_forecast_stay_within_bounded_defect"
        if self._exact_v1_guarded_turn_window_ranking_active():
            admission_reason = str(selected.get("exact_v1_admission_reason", ""))
            if admission_reason in {
                self._exact_v1_guarded_protected_horizon_admission_reason(),
                self._exact_v1_fidelity_first_turn_local_target_win_reason(),
            }:
                return None
        selected_score = self._forecast_tracking_score(forecast=selected_forecast)
        stay_score = self._forecast_tracking_score(forecast=stay_forecast)
        if float(selected_score) >= float(stay_score) - 1.0e-12:
            if self._exact_v1_nonimproving_score_allows_density_first_append(
                stay_forecast=stay_forecast,
                selected_forecast=selected_forecast,
            ):
                return None
            return "exact_forecast_nonimproving_tracking_score"
        return None

    def _select_exact_v1_candidate_step_scale(
        self,
        *,
        checkpoint_index: int | None = None,
        baseline_theta_dot: np.ndarray | Sequence[float],
        selected: Mapping[str, Any],
        dt: float,
        time_stop: float,
        anchor_summary: BaselineGeometrySummary | None = None,
        anchor_predicted_displacement: float | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        candidate_data = dict(selected["candidate_data"])
        best_selected: dict[str, Any] | None = None
        best_forecast: dict[str, Any] | None = None
        best_score: float | None = None
        best_scale: float | None = None
        for step_scale in self._candidate_step_scales_for_selected(
            selected=selected,
            time_stop=time_stop,
        ):
            scaled_theta_dot_aug, scaled_theta_dot_existing, scaled_eta_dot = self._scale_candidate_theta_dot(
                candidate_data=candidate_data,
                baseline_theta_dot=baseline_theta_dot,
                theta_dot_aug=selected["theta_dot_aug"],
                step_scale=float(step_scale),
            )
            theta_runtime = np.asarray(
                candidate_data["theta_aug"] + float(dt) * np.asarray(scaled_theta_dot_aug, dtype=float),
                dtype=float,
            ).reshape(-1)
            forecast, _forecast_rollout, score = self._local_projective_forecast_rollout(
                checkpoint_index=checkpoint_index,
                time_stop=float(time_stop),
                executor=candidate_data["aug_executor"],
                layout=candidate_data["aug_layout"],
                theta_runtime_start=theta_runtime,
                theta_dot_step=np.asarray(scaled_theta_dot_aug, dtype=float).reshape(-1),
                planning_audit=self._build_planning_audit_for_terms(candidate_data["aug_terms"]),
                scaffold_labels=[str(carrier.label) for carrier in candidate_data["aug_terms"]],
                immediate_gain_ratio=float(selected.get("gain_ratio", 0.0)),
                anchor_summary=anchor_summary,
                anchor_predicted_displacement=anchor_predicted_displacement,
            )
            choose = False
            if best_selected is None or best_forecast is None or best_score is None or best_scale is None:
                choose = True
            elif float(score) < float(best_score) - 1.0e-12:
                choose = True
            elif abs(float(score) - float(best_score)) <= 1.0e-12 and float(step_scale) < float(best_scale):
                choose = True
            if choose:
                best_selected = dict(selected)
                best_selected["theta_dot_aug"] = np.asarray(scaled_theta_dot_aug, dtype=float).reshape(-1)
                best_selected["theta_dot_aug_existing"] = np.asarray(
                    scaled_theta_dot_existing, dtype=float
                ).reshape(-1)
                best_selected["eta_dot"] = np.asarray(scaled_eta_dot, dtype=float).reshape(-1)
                best_selected["candidate_step_scale"] = float(step_scale)
                candidate_summary = best_selected.get("candidate_summary")
                if candidate_summary is not None:
                    best_selected["candidate_summary"] = replace(
                        candidate_summary,
                        selected_step_scale=float(step_scale),
                    )
                best_forecast = dict(forecast)
                best_score = float(score)
                best_scale = float(step_scale)
        if best_selected is None or best_forecast is None:
            raise RuntimeError("no exact-v1 candidate step-scale forecasts were produced")
        return best_selected, best_forecast

    def _current_scaffold_labels(self) -> list[str]:
        return [str(carrier.label) for carrier in self.current_terms]

    def _record_compile_audit_prune_event(
        self,
        *,
        checkpoint_index: int,
        time_value: float,
        selected_candidate_label: str | None,
        removed_label: str,
        logical_before: int,
        runtime_before: int,
        reduced_state: Mapping[str, Any],
    ) -> None:
        """Keep in-memory before/after scaffolds for later fake-backend compile audit."""
        reduced_terms = list(reduced_state["reduced_terms"])
        reduced_layout = reduced_state["reduced_layout"]
        reduced_theta = np.asarray(reduced_state["reduced_theta"], dtype=float).reshape(-1)
        self._compile_audit_prune_events.append(
            {
                "checkpoint_index": int(checkpoint_index),
                "time": float(time_value),
                "candidate_label": None if selected_candidate_label is None else str(selected_candidate_label),
                "removed_label": str(removed_label),
                "logical_block_count_before": int(logical_before),
                "logical_block_count_after": int(getattr(reduced_layout, "logical_parameter_count")),
                "runtime_parameter_count_before": int(runtime_before),
                "runtime_parameter_count_after": int(getattr(reduced_layout, "runtime_parameter_count")),
                "runtime_parameter_count_delta": int(getattr(reduced_layout, "runtime_parameter_count")) - int(runtime_before),
                "before": {
                    "layout": self.current_layout,
                    "theta_runtime": np.asarray(self.current_theta, dtype=float).reshape(-1).copy(),
                    "labels": self._current_scaffold_labels(),
                },
                "after": {
                    "layout": reduced_layout,
                    "theta_runtime": reduced_theta.copy(),
                    "labels": [str(carrier.label) for carrier in reduced_terms],
                },
            }
        )

    def _current_source_labels(self) -> set[str]:
        return {str(carrier.source_label) for carrier in self.current_terms}

    def _build_planning_audit_for_terms(
        self,
        terms: Sequence[RuntimeTermCarrier],
    ) -> MeasurementCacheAudit:
        audit = MeasurementCacheAudit(
            nominal_shots_per_group=1,
            plan_version="phase1_qwc_basis_cover_reuse",
            grouping_mode=str(self.cfg.grouping_mode),
        )
        for carrier in terms:
            audit.commit(planning_group_keys_for_term(_carrier_to_term(carrier)))
        return audit

    def _block_theta_snapshot(
        self,
        *,
        terms: Sequence[RuntimeTermCarrier] | None = None,
        layout: AnsatzParameterLayout | None = None,
        theta_runtime: np.ndarray | Sequence[float] | None = None,
    ) -> dict[str, np.ndarray]:
        resolved_terms = list(self.current_terms if terms is None else terms)
        resolved_layout = self.current_layout if layout is None else layout
        theta_arr = np.asarray(
            self.current_theta if theta_runtime is None else theta_runtime,
            dtype=float,
        ).reshape(-1)
        out: dict[str, np.ndarray] = {}
        for carrier, block in zip(resolved_terms, resolved_layout.blocks):
            out[str(carrier.label)] = np.asarray(
                theta_arr[int(block.runtime_start) : int(block.runtime_stop)],
                dtype=float,
            ).reshape(-1)
        return out

    def _initialize_prune_state(self) -> None:
        mature_birth = -max(1, int(getattr(self.cfg, "prune_protection_steps", 0)) + 1)
        for carrier in self.current_terms:
            label = str(carrier.label)
            self._block_birth_checkpoint[label] = int(mature_birth)
            self._block_cooldown[label] = 0
            self._block_burden[label] = float(max(1, len(carrier.runtime_specs)))
            self._block_origin[label] = "initial_scaffold"
            self._block_motion_history.setdefault(label, [])
            self._block_fit_history.setdefault(label, [])
        self._previous_block_theta_snapshot = self._block_theta_snapshot()

    def _decrement_prune_cooldowns(self) -> None:
        for carrier in self.current_terms:
            label = str(carrier.label)
            self._block_cooldown[label] = max(0, int(self._block_cooldown.get(label, 0)) - 1)

    def _set_previous_block_theta_snapshot(self) -> None:
        self._previous_block_theta_snapshot = self._block_theta_snapshot()

    def _record_prune_histories(self, *, baseline: Mapping[str, Any]) -> None:
        window = max(1, int(getattr(self.cfg, "prune_stagnation_window", 1)))
        current_snapshot = self._block_theta_snapshot()
        f_vec = np.asarray(baseline.get("f", np.zeros(0, dtype=float)), dtype=float).reshape(-1)
        delta_norms: dict[str, float] = {}
        fit_numerators: dict[str, float] = {}
        max_delta = 0.0
        fit_den = 0.0
        for carrier, block in zip(self.current_terms, self.current_layout.blocks):
            label = str(carrier.label)
            current_block = np.asarray(current_snapshot.get(label, np.zeros(0, dtype=float)), dtype=float).reshape(-1)
            prev_block = np.asarray(
                self._previous_block_theta_snapshot.get(label, np.zeros_like(current_block)),
                dtype=float,
            ).reshape(-1)
            if int(prev_block.size) != int(current_block.size):
                prev_block = np.zeros_like(current_block)
            delta_block = np.asarray(current_block - prev_block, dtype=float).reshape(-1)
            delta_norm = float(np.linalg.norm(delta_block))
            delta_norms[label] = float(delta_norm)
            max_delta = max(float(max_delta), float(delta_norm))
            start = int(block.runtime_start)
            stop = int(block.runtime_stop)
            f_block = np.asarray(f_vec[start:stop], dtype=float).reshape(-1)
            overlap = min(int(delta_block.size), int(f_block.size))
            fit_num = float(
                np.sum(np.abs(delta_block[:overlap]) * np.abs(f_block[:overlap]))
            ) if overlap > 0 else 0.0
            fit_numerators[label] = float(fit_num)
            fit_den += float(fit_num)
        for carrier in self.current_terms:
            label = str(carrier.label)
            motion_value = 0.0 if float(max_delta) <= 1.0e-12 else float(delta_norms.get(label, 0.0) / max_delta)
            fit_value = 0.0 if float(fit_den) <= 1.0e-12 else float(fit_numerators.get(label, 0.0) / fit_den)
            motion_hist = list(self._block_motion_history.get(label, []))
            fit_hist = list(self._block_fit_history.get(label, []))
            motion_hist.append(float(motion_value))
            fit_hist.append(float(fit_value))
            self._block_motion_history[label] = motion_hist[-window:]
            self._block_fit_history[label] = fit_hist[-window:]

    def _block_stagnation_statistics(self, label: str) -> tuple[float, float, float]:
        motion_hist = list(self._block_motion_history.get(str(label), []))
        fit_hist = list(self._block_fit_history.get(str(label), []))
        if not motion_hist or not fit_hist:
            return 0.0, 1.0, 1.0
        motion_mean = float(np.mean(np.asarray(motion_hist, dtype=float)))
        fit_mean = float(np.mean(np.asarray(fit_hist, dtype=float)))
        alpha = float(getattr(self.cfg, "prune_stagnation_alpha", 0.5))
        stagnation_score = float(
            alpha * (1.0 - min(1.0, max(0.0, motion_mean)))
            + (1.0 - alpha) * (1.0 - min(1.0, max(0.0, fit_mean)))
        )
        return stagnation_score, motion_mean, fit_mean

    def _prune_permitted(
        self,
        *,
        rho_miss: float,
        motion: MotionSchedulerTelemetry,
    ) -> tuple[bool, str]:
        if str(getattr(self.cfg, "prune_mode", "off")) == "off":
            return False, "prune_disabled"
        if int(self.current_layout.logical_parameter_count) < 2:
            return False, "scaffold_too_small"
        if float(rho_miss) > float(getattr(self.cfg, "prune_miss_threshold", 0.0)):
            if not (
                self._recoverability_prune_enabled()
                and bool(getattr(self.cfg, "prune_high_miss_differential_enabled", True))
            ):
                return False, "rho_miss_above_prune_threshold"
            return True, "prune_permitted_high_miss_differential"
        direction_cosine = motion.direction_cosine
        if direction_cosine is None or float(direction_cosine) < float(self.cfg.motion_calm_direction_cosine_threshold):
            return False, "motion_not_calm_direction"
        rate_change_ratio = motion.rate_change_ratio
        if rate_change_ratio is None or float(rate_change_ratio) > float(self.cfg.motion_calm_rate_change_ratio_threshold):
            return False, "motion_not_calm_rate"
        return True, "prune_permitted"

    def _prune_blocker_category(self, reason: str | None) -> str:
        text = str(reason or "unknown").strip().lower()
        if text in {"", "none"}:
            return "unknown"
        if "disabled" in text or text == "controller_mode_off":
            return "disabled"
        if "terminal" in text:
            return "terminal"
        if "not_calm" in text or "motion_" in text:
            return "not_calm"
        if "high_miss" in text or "above_threshold" in text or "rho_miss_above" in text:
            return "high_miss"
        if "appended_prune" in text or "appended_origin" in text:
            return "append_cleanup"
        if "no_prune" in text or "no_eligible" in text or "scaffold_too_small" in text:
            return "no_eligible"
        if "loss" in text or "rho_miss_increase" in text:
            return "loss"
        if "state_jump" in text:
            return "state_jump"
        if "degraded" in text or "error" in text:
            return "error"
        if "available" in text or "permitted" in text:
            return "available"
        return "other"

    def _record_prune_blocker_reason(self, reason: str | None) -> None:
        reason_text = str(reason or "unknown")
        self._prune_blocker_reason_counts[reason_text] = (
            int(self._prune_blocker_reason_counts.get(reason_text, 0)) + 1
        )
        category = f"category:{self._prune_blocker_category(reason_text)}"
        self._prune_blocker_reason_counts[category] = (
            int(self._prune_blocker_reason_counts.get(category, 0)) + 1
        )

    def _recoverability_prune_enabled(self) -> bool:
        return str(getattr(self.cfg, "prune_mode", "off")) == "schur_projected_shadow_v1"

    def _schur_gain_for_indices(
        self,
        *,
        K: np.ndarray,
        f_vec: np.ndarray,
        indices: Sequence[int],
    ) -> float:
        idx = [int(i) for i in indices if 0 <= int(i) < int(f_vec.size)]
        if not idx:
            return 0.0
        K_sub = np.asarray(K[np.ix_(idx, idx)], dtype=float)
        f_sub = np.asarray(f_vec[idx], dtype=float).reshape(-1)
        if K_sub.size <= 0 or f_sub.size <= 0:
            return 0.0
        try:
            K_pinv = np.linalg.pinv(K_sub, rcond=self._cfg_float("pinv_rcond"))
            value = float(f_sub @ (K_pinv @ f_sub))
        except np.linalg.LinAlgError:
            return float("nan")
        return float(max(0.0, value)) if np.isfinite(value) else float("nan")

    def _schur_prune_loss_ladder(
        self,
        *,
        baseline: Mapping[str, Any],
        runtime_indices: Sequence[int],
        logical_index: int | None = None,
    ) -> list[dict[str, Any]]:
        """McLachlan-Schur prune loss ladder from Math 17A.

        This uses the controller's ridge-consistent K/f geometry.  It does not
        inspect ED target states or reference trajectories.
        """
        remove = sorted({int(idx) for idx in runtime_indices})
        f_vec = np.asarray(baseline.get("f", np.zeros(0, dtype=float)), dtype=float).reshape(-1)
        if f_vec.size <= 0:
            return []
        K_raw = baseline.get("K", None)
        if K_raw is None:
            G = np.asarray(baseline.get("G", np.zeros((0, 0), dtype=float)), dtype=float)
            K = np.asarray(G + self._cfg_float("regularization_lambda") * np.eye(int(G.shape[0])), dtype=float)
        else:
            K = np.asarray(K_raw, dtype=float)
        if K.shape != (int(f_vec.size), int(f_vec.size)):
            return []
        full_indices = list(range(int(f_vec.size)))
        keep_full = [idx for idx in full_indices if idx not in set(remove)]
        norm_b_sq = max(
            float(baseline.get("norm_b_sq", 0.0)),
            float(getattr(self.cfg, "prune_loss_norm_epsilon", 1.0e-14)),
            1.0e-14,
        )
        full_gain = self._schur_gain_for_indices(K=K, f_vec=f_vec, indices=full_indices)
        if not np.isfinite(full_gain):
            return []

        rung_specs: list[tuple[str, list[int]]] = [("empty", [])]
        if logical_index is not None and self.current_layout.blocks:
            radius = max(0, int(getattr(self.cfg, "prune_schur_ladder_local_radius", 1)))
            lo = max(0, int(logical_index) - radius)
            hi = min(len(self.current_layout.blocks), int(logical_index) + radius + 1)
            local: list[int] = []
            remove_set = set(remove)
            for block in self.current_layout.blocks[lo:hi]:
                for idx in range(int(block.runtime_start), int(block.runtime_stop)):
                    if idx not in remove_set and idx in keep_full:
                        local.append(int(idx))
            rung_specs.append(("local", sorted(set(local))))
        rung_specs.append(("full", keep_full))

        rows: list[dict[str, Any]] = []
        seen: set[tuple[int, ...]] = set()
        previous: float | None = None
        tol = float(getattr(self.cfg, "prune_schur_monotonicity_tol", 1.0e-9))
        for kind, compensators in rung_specs:
            key = tuple(int(x) for x in sorted(set(compensators)))
            if key in seen:
                continue
            seen.add(key)
            gain_keep = self._schur_gain_for_indices(K=K, f_vec=f_vec, indices=key)
            if np.isfinite(gain_keep):
                raw_loss = float(max(0.0, full_gain - gain_keep))
                normalized_loss = float(raw_loss / norm_b_sq)
            else:
                raw_loss = float("inf")
                normalized_loss = float("inf")
            monotone = bool(np.isfinite(normalized_loss)) and (
                True if previous is None else bool(normalized_loss <= previous + tol)
            )
            rows.append(
                {
                    "rung_index": int(len(rows)),
                    "rung_kind": str(kind),
                    "compensator_runtime_indices": [int(x) for x in key],
                    "raw_loss": float(raw_loss),
                    "normalized_loss": float(normalized_loss),
                    "monotone_nonincreasing": bool(monotone),
                    "loss_semantics": "schur_normalized_v1",
                }
            )
            previous = normalized_loss
        return rows

    def _phase_aligned_state(
        self,
        *,
        target: np.ndarray | Sequence[complex],
        state: np.ndarray | Sequence[complex],
    ) -> np.ndarray:
        target_arr = np.asarray(target, dtype=complex).reshape(-1)
        state_arr = np.asarray(state, dtype=complex).reshape(-1)
        if target_arr.size != state_arr.size or target_arr.size <= 0:
            return np.asarray(state_arr, dtype=complex).reshape(-1)
        overlap = complex(np.vdot(target_arr, state_arr))
        if abs(overlap) <= 1.0e-14:
            return np.asarray(state_arr, dtype=complex).reshape(-1)
        return np.asarray(state_arr * np.exp(-1.0j * float(np.angle(overlap))), dtype=complex).reshape(-1)

    def _state_ray_distance(
        self,
        lhs: np.ndarray | Sequence[complex],
        rhs: np.ndarray | Sequence[complex],
    ) -> float:
        lhs_arr = np.asarray(lhs, dtype=complex).reshape(-1)
        rhs_arr = np.asarray(rhs, dtype=complex).reshape(-1)
        if lhs_arr.size != rhs_arr.size or lhs_arr.size <= 0:
            return float("inf")
        lhs_norm = float(np.linalg.norm(lhs_arr))
        rhs_norm = float(np.linalg.norm(rhs_arr))
        if lhs_norm <= 1.0e-14 or rhs_norm <= 1.0e-14:
            return float("inf")
        overlap = abs(complex(np.vdot(lhs_arr, rhs_arr))) / max(lhs_norm * rhs_norm, 1.0e-14)
        overlap = float(np.clip(overlap, 0.0, 1.0))
        return float(np.sqrt(max(0.0, 1.0 - overlap * overlap)))

    def _prune_projection_objective(
        self,
        *,
        incumbent_psi: np.ndarray,
        candidate_psi: np.ndarray,
        theta_runtime: np.ndarray,
        raw_theta: np.ndarray,
    ) -> float:
        ray = self._state_ray_distance(incumbent_psi, candidate_psi)
        theta_delta = float(np.linalg.norm(np.asarray(theta_runtime, dtype=float).reshape(-1) - np.asarray(raw_theta, dtype=float).reshape(-1)))
        state_weight = float(getattr(self.cfg, "prune_projection_state_weight", 1.0))
        reg = float(getattr(self.cfg, "prune_projection_regularization", 1.0e-8))
        value = state_weight * float(ray) * float(ray) + reg * theta_delta * theta_delta
        return float(value if np.isfinite(value) else float("inf"))

    def _project_pruned_runtime_state(
        self,
        *,
        reduced_executor: Any,
        reduced_theta_raw: np.ndarray | Sequence[float],
        incumbent_psi: np.ndarray | Sequence[complex],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Project a pruned scaffold back toward the current prepared ansatz state.

        This is a reduced-state ansatz projection.  It targets the incumbent
        prepared controller state, not an ED target/reference trajectory.
        """
        raw_theta = np.asarray(reduced_theta_raw, dtype=float).reshape(-1)
        incumbent = np.asarray(incumbent_psi, dtype=complex).reshape(-1)
        diagnostics: dict[str, Any] = {
            "prune_projection_mode": str(getattr(self.cfg, "prune_projection_mode", "state_tangent_ls_v1")),
            "prune_projection_uses_exact_reference": False,
            "prune_projection_rounds_requested": int(getattr(self.cfg, "prune_projection_rounds", 2)),
        }
        def _prepare(theta_vec: np.ndarray) -> np.ndarray:
            return np.asarray(
                reduced_executor.prepare_state(np.asarray(theta_vec, dtype=float).reshape(-1), self.replay_context.psi_ref),
                dtype=complex,
            ).reshape(-1)

        if str(getattr(self.cfg, "prune_projection_mode", "state_tangent_ls_v1")) == "raw_delete":
            psi_raw = _prepare(raw_theta)
            aligned_raw = self._phase_aligned_state(target=incumbent, state=psi_raw)
            diagnostics.update(
                {
                    "prune_projection_rounds_completed": 0,
                    "prune_projection_raw_state_jump_l2": float(np.linalg.norm(aligned_raw - incumbent)),
                    "prune_projected_state_jump_l2": float(np.linalg.norm(aligned_raw - incumbent)),
                    "prune_ray_distance": float(self._state_ray_distance(incumbent, psi_raw)),
                    "prune_projection_objective": float(self._prune_projection_objective(
                        incumbent_psi=incumbent,
                        candidate_psi=psi_raw,
                        theta_runtime=raw_theta,
                        raw_theta=raw_theta,
                    )),
                }
            )
            return raw_theta, psi_raw, diagnostics

        theta = np.asarray(raw_theta, dtype=float).reshape(-1).copy()
        n_param = int(theta.size)
        active_cap = max(1, int(getattr(self.cfg, "prune_projection_max_active_runtime", 64)))
        active = list(range(min(n_param, active_cap)))
        rounds = max(0, int(getattr(self.cfg, "prune_projection_rounds", 2)))
        trust = max(0.0, float(getattr(self.cfg, "prune_projection_trust_radius", 5.0e-2)))
        reg = max(0.0, float(getattr(self.cfg, "prune_projection_regularization", 1.0e-8)))
        eps = 1.0e-5

        psi_raw = _prepare(raw_theta)
        aligned_raw = self._phase_aligned_state(target=incumbent, state=psi_raw)
        best_theta = theta.copy()
        best_psi = np.asarray(psi_raw, dtype=complex).reshape(-1)
        best_objective = self._prune_projection_objective(
            incumbent_psi=incumbent,
            candidate_psi=best_psi,
            theta_runtime=best_theta,
            raw_theta=raw_theta,
        )
        completed = 0
        for _round_idx in range(rounds):
            if not active:
                break
            psi = _prepare(theta)
            aligned = self._phase_aligned_state(target=incumbent, state=psi)
            residual = np.asarray(incumbent - aligned, dtype=complex).reshape(-1)
            cols: list[np.ndarray] = []
            for runtime_idx in active:
                theta_eps = theta.copy()
                theta_eps[int(runtime_idx)] += eps
                psi_eps = _prepare(theta_eps)
                aligned_eps = self._phase_aligned_state(target=incumbent, state=psi_eps)
                cols.append(np.asarray((aligned_eps - aligned) / eps, dtype=complex).reshape(-1))
            if not cols:
                break
            J_complex = np.column_stack(cols)
            A = np.vstack([np.real(J_complex), np.imag(J_complex)])
            b = np.concatenate([np.real(residual), np.imag(residual)])
            if reg > 0.0:
                A = np.vstack([A, np.sqrt(reg) * np.eye(len(active))])
                b = np.concatenate([b, np.zeros(len(active), dtype=float)])
            try:
                delta_active = np.linalg.lstsq(A, b, rcond=self._cfg_float("pinv_rcond"))[0]
            except Exception:
                break
            delta_active = np.asarray(delta_active, dtype=float).reshape(-1)
            delta_norm = float(np.linalg.norm(delta_active))
            if delta_norm > trust > 0.0:
                delta_active = np.asarray(delta_active * (trust / max(delta_norm, 1.0e-14)), dtype=float)
            trial_theta = theta.copy()
            for slot, runtime_idx in enumerate(active):
                trial_theta[int(runtime_idx)] += float(delta_active[int(slot)])
            trial_psi = _prepare(trial_theta)
            trial_objective = self._prune_projection_objective(
                incumbent_psi=incumbent,
                candidate_psi=trial_psi,
                theta_runtime=trial_theta,
                raw_theta=raw_theta,
            )
            completed += 1
            if float(trial_objective) <= float(best_objective) + 1.0e-15:
                theta = trial_theta
                best_theta = trial_theta.copy()
                best_psi = np.asarray(trial_psi, dtype=complex).reshape(-1)
                best_objective = float(trial_objective)
            else:
                break

        aligned_best = self._phase_aligned_state(target=incumbent, state=best_psi)
        diagnostics.update(
            {
                "prune_projection_rounds_completed": int(completed),
                "prune_projection_active_runtime_count": int(len(active)),
                "prune_projection_raw_state_jump_l2": float(np.linalg.norm(aligned_raw - incumbent)),
                "prune_projected_state_jump_l2": float(np.linalg.norm(aligned_best - incumbent)),
                "prune_ray_distance": float(self._state_ray_distance(incumbent, best_psi)),
                "prune_projection_objective": float(best_objective),
            }
        )
        return np.asarray(best_theta, dtype=float).reshape(-1), np.asarray(best_psi, dtype=complex).reshape(-1), diagnostics

    def _prune_persistence_key(self, proposed: Mapping[str, Any]) -> str:
        return "|".join(
            [
                str(proposed.get("candidate_label", "unknown")),
                str(proposed.get("origin_kind", "unknown")),
                str(proposed.get("birth_checkpoint", "unknown")),
                str(proposed.get("position_id", "unknown")),
            ]
        )

    def _update_prune_persistence(self, *, key: str, passed: bool) -> tuple[int, int, bool]:
        window = max(1, int(getattr(self.cfg, "prune_persistence_window", 1)))
        required = max(1, min(int(getattr(self.cfg, "prune_persistence_required", 1)), window))
        hist = list(self._prune_persistence_history.get(str(key), []))
        hist.append(bool(passed))
        hist = hist[-window:]
        self._prune_persistence_history[str(key)] = hist
        count = int(sum(1 for item in hist if bool(item)))
        return count, required, bool(count >= required)

    def _cached_prune_loss(
        self,
        *,
        baseline: Mapping[str, Any],
        runtime_indices: Sequence[int],
    ) -> float:
        idx_remove = {int(idx) for idx in runtime_indices}
        G = np.asarray(baseline.get("G", np.zeros((0, 0), dtype=float)), dtype=float)
        f_vec = np.asarray(baseline.get("f", np.zeros(0, dtype=float)), dtype=float).reshape(-1)
        keep = [idx for idx in range(int(f_vec.size)) if idx not in idx_remove]
        if G.size == 0 or f_vec.size == 0:
            return 0.0
        baseline_objective = float(np.asarray(baseline.get("theta_dot_proj", np.zeros_like(f_vec)), dtype=float).reshape(-1) @ f_vec)
        if not keep:
            reduced_objective = 0.0
        else:
            G_red = np.asarray(G[np.ix_(keep, keep)], dtype=float)
            f_red = np.asarray(f_vec[keep], dtype=float).reshape(-1)
            G_red_pinv = (
                np.linalg.pinv(G_red, rcond=self._cfg_float("pinv_rcond"))
                if G_red.size
                else np.zeros((0, 0), dtype=float)
            )
            reduced_objective = float(f_red @ (G_red_pinv @ f_red)) if f_red.size else 0.0
        norm_b_sq = float(max(float(baseline.get("norm_b_sq", 0.0)), 1.0e-14))
        return float(max(0.0, baseline_objective - reduced_objective) / norm_b_sq)

    def _appended_origin_prune_bias_factor(
        self,
        *,
        label: str,
        age_checkpoints: int,
    ) -> tuple[float, bool, bool, int]:
        enabled = bool(getattr(self.cfg, "prune_appended_origin_bias_enabled", True))
        origin_kind = str(self._block_origin.get(str(label), "initial_scaffold"))
        is_appended = origin_kind == "append"
        grace_steps = max(0, int(getattr(self.cfg, "prune_appended_origin_grace_steps", 1)))
        if not enabled or not is_appended:
            return 0.0, False, is_appended, int(grace_steps)
        if int(age_checkpoints) <= int(grace_steps):
            return 0.0, False, True, int(grace_steps)
        post_grace_age = max(0, int(age_checkpoints) - int(grace_steps))
        scale = float(getattr(self.cfg, "prune_appended_origin_bias_scale", 0.10))
        max_factor = float(getattr(self.cfg, "prune_appended_origin_bias_max_factor", 0.50))
        factor = float(min(float(max_factor), float(scale) * float(post_grace_age)))
        factor = float(max(0.0, factor))
        return factor, bool(factor > 0.0), True, int(grace_steps)

    def _prune_candidates(
        self,
        *,
        checkpoint_index: int,
        baseline: Mapping[str, Any],
        motion: MotionSchedulerTelemetry,
    ) -> tuple[list[dict[str, Any]], str]:
        permitted, reason = self._prune_permitted(
            rho_miss=float(baseline["summary"].rho_miss),
            motion=motion,
        )
        if not permitted:
            return [], str(reason)
        protection_steps = int(getattr(self.cfg, "prune_protection_steps", 0))
        stale_threshold = float(getattr(self.cfg, "prune_stale_score_threshold", 1.0))
        appended_bias_enabled = bool(
            getattr(self.cfg, "prune_appended_origin_bias_enabled", True)
        )
        target_policy = str(
            getattr(self.cfg, "prune_appended_origin_target_policy", "append_only")
        )
        appended_block_count = int(
            sum(
                1
                for carrier in self.current_terms
                if str(self._block_origin.get(str(carrier.label), "initial_scaffold")) == "append"
            )
        )
        theta_dot_step = np.asarray(
            baseline.get("theta_dot_step", np.zeros(0, dtype=float)),
            dtype=float,
        ).reshape(-1)
        theta_dot_total_norm = float(np.linalg.norm(theta_dot_step))
        active_rel_tol = float(getattr(self.cfg, "prune_active_block_theta_dot_rel_tol", 0.03))
        active_abs_tol = float(getattr(self.cfg, "prune_active_block_theta_dot_abs_tol", 1.0e-8))
        active_abs_hard_tol = float(
            getattr(self.cfg, "prune_active_block_theta_dot_abs_hard_tol", 5.0e-2)
        )
        rows: list[dict[str, Any]] = []
        for logical_index, (carrier, block) in enumerate(zip(self.current_terms, self.current_layout.blocks)):
            label = str(carrier.label)
            birth = int(self._block_birth_checkpoint.get(label, -protection_steps - 1))
            age = int(checkpoint_index) - int(birth)
            cooldown = int(self._block_cooldown.get(label, 0))
            runtime_indices = list(range(int(block.runtime_start), int(block.runtime_stop)))
            theta_dot_block = (
                np.asarray(theta_dot_step[runtime_indices], dtype=float).reshape(-1)
                if theta_dot_step.size > 0
                and runtime_indices
                and max(runtime_indices) < int(theta_dot_step.size)
                else np.zeros(len(runtime_indices), dtype=float)
            )
            theta_dot_block_norm = float(np.linalg.norm(theta_dot_block))
            theta_dot_block_rel = (
                0.0
                if float(theta_dot_total_norm) <= 1.0e-14
                else float(theta_dot_block_norm / max(theta_dot_total_norm, 1.0e-14))
            )
            stagnation_score, motion_mean, fit_mean = self._block_stagnation_statistics(label)
            theta_block = np.asarray(self.current_theta[int(block.runtime_start) : int(block.runtime_stop)], dtype=float).reshape(-1)
            theta_block_norm = float(np.linalg.norm(theta_block))
            bias_factor, bias_applied, append_origin, grace_steps = self._appended_origin_prune_bias_factor(
                label=str(label),
                age_checkpoints=int(age),
            )
            origin_kind = str(self._block_origin.get(label, "initial_scaffold"))
            in_appended_grace = bool(
                appended_bias_enabled
                and append_origin
                and int(age) <= int(grace_steps)
            )
            initial_scaffold_grace_steps = max(
                0, int(getattr(self.cfg, "prune_initial_scaffold_grace_steps", 64))
            )
            in_initial_scaffold_grace = bool(
                origin_kind == "initial_scaffold"
                and int(age) <= int(initial_scaffold_grace_steps)
            )
            if in_appended_grace or in_initial_scaffold_grace:
                continue
            active_by_relative_share = bool(
                float(theta_dot_block_norm) > float(active_abs_tol)
                and float(theta_dot_block_rel) >= float(active_rel_tol)
            )
            active_by_absolute_hard_cap = bool(
                float(active_abs_hard_tol) > 0.0
                and float(theta_dot_block_norm) > float(active_abs_hard_tol)
            )
            if bool(active_by_relative_share or active_by_absolute_hard_cap):
                continue
            if age < int(protection_steps):
                continue
            if cooldown > 0:
                continue
            stale_score_for_gate = float(
                min(1.0, float(stagnation_score) * (1.0 + float(bias_factor)))
            )
            recoverability_mode = self._recoverability_prune_enabled()
            if (not recoverability_mode) and float(stale_score_for_gate) < float(stale_threshold):
                continue
            if recoverability_mode:
                schur_ladder = self._schur_prune_loss_ladder(
                    baseline=baseline,
                    runtime_indices=runtime_indices,
                    logical_index=int(logical_index),
                )
                selected_schur = dict(schur_ladder[-1]) if schur_ladder else {}
                cached_prune_loss = float(selected_schur.get("normalized_loss", float("inf")))
                cached_prune_loss_semantics = "schur_normalized_v1"
                monotone_ok = bool(all(bool(row.get("monotone_nonincreasing", True)) for row in schur_ladder))
                schur_status = "ok" if monotone_ok else "nonmonotone_ladder"
                compat_denominator = max(
                    float(baseline.get("norm_b_sq", 0.0)),
                    float(getattr(self.cfg, "prune_loss_norm_epsilon", 1.0e-14)),
                    1.0e-14,
                )
                prune_loss_payload = compute_prune_loss_payload(
                    G=baseline.get("G"),
                    K=baseline.get("K"),
                    f_vec=baseline.get("f", np.zeros(0, dtype=float)),
                    norm_b_sq=float(baseline.get("norm_b_sq", 0.0)),
                    removed_runtime_indices=runtime_indices,
                    pinv_rcond=self._cfg_float("pinv_rcond"),
                    regularization_lambda=self._cfg_float("regularization_lambda"),
                    epsilon=float(getattr(self.cfg, "prune_loss_norm_epsilon", 1.0e-14)),
                    selected_loss=float(cached_prune_loss),
                    selected_loss_kind=COMPAT_SCHUR_NORMALIZED_V1,
                    selected_denominator=float(compat_denominator),
                    selected_denominator_kind=DENOM_MAX_NORM_B_EPS_COMPAT_V1,
                    selected_matrix_for_selection=MATRIX_COMPAT_SCHUR_K,
                    monotonicity_status=str(schur_status),
                )
                permit_path = (
                    "high_miss_differential"
                    if float(baseline["summary"].rho_miss) > float(getattr(self.cfg, "prune_miss_threshold", 0.0))
                    else "low_miss_standard"
                )
            else:
                schur_ladder = []
                selected_schur = {}
                cached_prune_loss = float(
                    self._cached_prune_loss(
                        baseline=baseline,
                        runtime_indices=runtime_indices,
                    )
                )
                cached_prune_loss_semantics = "legacy_proxy_v1"
                schur_status = None
                compat_denominator = max(float(baseline.get("norm_b_sq", 0.0)), 1.0e-14)
                prune_loss_payload = compute_prune_loss_payload(
                    G=baseline.get("G"),
                    K=baseline.get("K"),
                    f_vec=baseline.get("f", np.zeros(0, dtype=float)),
                    norm_b_sq=float(baseline.get("norm_b_sq", 0.0)),
                    removed_runtime_indices=runtime_indices,
                    pinv_rcond=self._cfg_float("pinv_rcond"),
                    regularization_lambda=self._cfg_float("regularization_lambda"),
                    epsilon=float(getattr(self.cfg, "prune_loss_norm_epsilon", 1.0e-14)),
                    selected_loss=float(cached_prune_loss),
                    selected_loss_kind=LEGACY_PROXY_V1,
                    selected_denominator=float(compat_denominator),
                    selected_denominator_kind=DENOM_MAX_NORM_B_EPS_COMPAT_V1,
                    selected_matrix_for_selection=MATRIX_LEGACY_PROXY,
                    legacy_proxy_loss=float(cached_prune_loss),
                )
                permit_path = "legacy_low_miss"
            prune_selection_score = float(
                cached_prune_loss / max(1.0, 1.0 + float(bias_factor))
            )
            prune_rank_score_terms = {
                "selected_loss": float(cached_prune_loss),
                "appended_origin_bias_factor": float(bias_factor),
                "appended_origin_bias_applied": bool(bias_applied),
                "burden": float(self._block_burden.get(label, max(1, len(runtime_indices)))),
                "position_id": int(logical_index),
            }
            rows.append(
                {
                    "candidate_label": str(label),
                    "position_id": int(logical_index),
                    "runtime_block_indices": [int(x) for x in runtime_indices],
                    "cached_prune_loss": float(cached_prune_loss),
                    "cached_prune_loss_semantics": str(cached_prune_loss_semantics),
                    "prune_selection_score": float(prune_selection_score),
                    "prune_rank_score": float(prune_selection_score),
                    "prune_rank_score_kind": "compat_selected_loss_with_appended_origin_bias_v1",
                    "prune_rank_score_terms": dict(prune_rank_score_terms),
                    "stagnation_score": float(stagnation_score),
                    "stagnation_score_for_gate": float(stale_score_for_gate),
                    "motion_mean": float(motion_mean),
                    "fit_mean": float(fit_mean),
                    "theta_block_norm": float(theta_block_norm),
                    "theta_dot_block_norm": float(theta_dot_block_norm),
                    "theta_dot_block_rel": float(theta_dot_block_rel),
                    "burden": float(self._block_burden.get(label, max(1, len(runtime_indices)))),
                    "origin_kind": str(origin_kind),
                    "append_origin": bool(append_origin),
                    "birth_checkpoint": int(birth),
                    "age_checkpoints": int(age),
                    "appended_origin_grace_steps": int(grace_steps),
                    "appended_origin_bias_enabled": bool(appended_bias_enabled),
                    "appended_origin_target_policy": str(target_policy),
                    "appended_origin_target_policy_applied": False,
                    "appended_origin_bias_factor": float(bias_factor),
                    "appended_origin_bias_applied": bool(bias_applied),
                    "prune_permit_path": str(permit_path),
                    "prune_schur_ladder": [dict(row) for row in schur_ladder],
                    "prune_schur_raw_loss": (
                        None if selected_schur.get("raw_loss") is None else float(selected_schur.get("raw_loss"))
                    ),
                    "prune_schur_normalized_loss": (
                        None
                        if selected_schur.get("normalized_loss") is None
                        else float(selected_schur.get("normalized_loss"))
                    ),
                    "prune_schur_selected_rung": (
                        None
                        if selected_schur.get("rung_index") is None
                        else int(selected_schur.get("rung_index"))
                    ),
                    "prune_schur_monotonicity_status": (
                        None if schur_status is None else str(schur_status)
                    ),
                    **dict(prune_loss_payload),
                    "removed_carrier": carrier,
                }
            )
        if target_policy in {"append_only", "prefer_append"}:
            appended_rows = [row for row in rows if bool(row.get("append_origin", False))]
            if appended_rows:
                rows = appended_rows
                for row in rows:
                    row["appended_origin_target_policy_applied"] = True
            elif target_policy == "append_only":
                reason = (
                    "no_appended_prune_targets"
                    if int(appended_block_count) <= 0
                    else "no_appended_prune_eligible_coordinates"
                )
                return [], reason
        rows.sort(
            key=lambda rec: (
                float(rec["prune_selection_score"]),
                -int(bool(rec.get("appended_origin_bias_applied", False))),
                -float(rec["burden"]),
                int(rec["position_id"]),
            )
        )
        if not rows:
            return [], "no_prune_eligible_coordinates"
        return rows[: max(1, int(getattr(self.cfg, "prune_max_candidates", 1)))], "prune_candidates_available"

    def _build_pruned_runtime_state(
        self,
        *,
        logical_index: int,
        baseline: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        idx = int(logical_index)
        removed_block = self.current_layout.blocks[idx]
        removed_carrier = self.current_terms[idx]
        reduced_terms = list(self.current_terms[:idx]) + list(self.current_terms[idx + 1 :])
        reduced_layout = _layout_from_carriers(reduced_terms, template=self.current_layout)
        reduced_theta_raw = _delete_theta_block(
            self.current_theta,
            runtime_start=int(removed_block.runtime_start),
            runtime_stop=int(removed_block.runtime_stop),
        )
        reduced_executor = self._build_executor(reduced_terms, reduced_layout)
        projection_diagnostics: dict[str, Any] = {
            "prune_projection_mode": "raw_delete",
            "prune_projection_uses_exact_reference": False,
        }
        if self._recoverability_prune_enabled() and baseline is not None:
            reduced_theta, reduced_psi, projection_diagnostics = self._project_pruned_runtime_state(
                reduced_executor=reduced_executor,
                reduced_theta_raw=np.asarray(reduced_theta_raw, dtype=float).reshape(-1),
                incumbent_psi=np.asarray(baseline["psi"], dtype=complex).reshape(-1),
            )
        else:
            reduced_theta = np.asarray(reduced_theta_raw, dtype=float).reshape(-1)
            reduced_psi = np.asarray(
                reduced_executor.prepare_state(reduced_theta, self.replay_context.psi_ref),
                dtype=complex,
            ).reshape(-1)
            incumbent = (
                None
                if baseline is None
                else np.asarray(baseline.get("psi"), dtype=complex).reshape(-1)
            )
            if incumbent is not None and incumbent.size == reduced_psi.size:
                aligned = self._phase_aligned_state(target=incumbent, state=reduced_psi)
                projection_diagnostics.update(
                    {
                        "prune_projected_state_jump_l2": float(np.linalg.norm(aligned - incumbent)),
                        "prune_ray_distance": float(self._state_ray_distance(incumbent, reduced_psi)),
                        "prune_projection_objective": float(self._prune_projection_objective(
                            incumbent_psi=incumbent,
                            candidate_psi=reduced_psi,
                            theta_runtime=reduced_theta,
                            raw_theta=np.asarray(reduced_theta_raw, dtype=float).reshape(-1),
                        )),
                    }
                )
        return {
            "removed_label": str(removed_carrier.label),
            "removed_source_label": str(removed_carrier.source_label),
            "removed_carrier": removed_carrier,
            "removed_block": removed_block,
            "reduced_terms": reduced_terms,
            "reduced_layout": reduced_layout,
            "reduced_theta": np.asarray(reduced_theta, dtype=float).reshape(-1),
            "reduced_executor": reduced_executor,
            "reduced_psi": np.asarray(reduced_psi, dtype=complex).reshape(-1),
            "reduced_planning_audit": self._build_planning_audit_for_terms(reduced_terms),
            "prune_projection_diagnostics": dict(projection_diagnostics),
        }

    def _oracle_wallclock_hit(self) -> bool:
        if self._wallclock_cap_s is None or self._run_wallclock_start is None:
            return False
        return bool((time.perf_counter() - float(self._run_wallclock_start)) >= float(self._wallclock_cap_s))

    def _oracle_estimate_kind(self) -> str | None:
        if self._oracle_base_config is None:
            return None
        return f"oracle_{str(self._oracle_base_config.noise_mode).strip().lower()}"

    def _projection_sample_time(self, time_start: float, time_stop: float | None) -> float:
        if self._drive_config is None:
            return float(time_start)
        sampling = str(self._drive_config.drive_time_sampling).strip().lower()
        if sampling not in {"midpoint", "left", "right"}:
            raise ValueError(
                f"Unsupported drive_time_sampling {self._drive_config.drive_time_sampling!r}."
            )
        if time_stop is None:
            return float(time_start)
        if sampling == "midpoint":
            return 0.5 * (float(time_start) + float(time_stop))
        if sampling == "left":
            return float(time_start)
        return float(time_stop)

    def _physical_time(self, time_value: float) -> float:
        if self._drive_config is None:
            return float(time_value)
        return float(time_value) + float(self._drive_config.drive_t0)

    def _drive_diagnostic_payload(
        self,
        *,
        physical_time: float,
        drive_term_count: int,
    ) -> dict[str, Any]:
        """Return additive drive telemetry derived only from run config/step data."""

        profile = dict(self._drive_profile or {})
        operator_label = profile.get("operator_label", None)
        family_key = profile.get("family_key", None)
        coefficient: float | None = None
        coefficient_linf: float | None = None
        if self._drive_model is not None:
            coefficient = float(self._drive_model.coefficient_at(float(physical_time)))
            if operator_label in {None, ""}:
                operator_label = getattr(self._drive_model, "operator_label", None)
            if family_key in {None, ""}:
                family_key = getattr(self._drive_model, "family_key", None)
        elif self._drive_coeff_provider_exyz is not None:
            try:
                coeff_map = {
                    str(label): complex(coeff)
                    for label, coeff in dict(self._drive_coeff_provider_exyz(float(physical_time))).items()
                }
                if coeff_map:
                    coefficient_linf = float(max(abs(complex(value)) for value in coeff_map.values()))
            except Exception:
                coefficient_linf = None
        return {
            "drive_enabled": bool(self._drive_config is not None),
            "drive_operator_label": (None if operator_label in {None, ""} else str(operator_label)),
            "drive_family_key": (None if family_key in {None, ""} else str(family_key)),
            "drive_coefficient": coefficient,
            "drive_coefficient_linf": coefficient_linf,
            "drive_term_count": int(drive_term_count),
        }

    def _static_hmat_for_artifacts(self) -> np.ndarray:
        if self.hmat is None:
            if bool(self.strict_qpu_faithful):
                return np.zeros((0, 0), dtype=complex)
            raise ValueError("static Hamiltonian matrix is unavailable")
        return np.asarray(self.hmat, dtype=complex)

    def _strict_step_hamiltonian_artifacts_from_poly(
        self,
        *,
        physical_time: float,
        h_poly_step: Any,
        drive_term_count: int,
    ) -> StepHamiltonianArtifacts:
        try:
            h_poly_step._reduce()
        except Exception:
            pass
        compiled_h_step = compile_polynomial_action(
            h_poly_step,
            tol=1e-12,
            pauli_action_cache=self._pauli_action_cache,
        )
        oracle_observable = (
            None
            if self._oracle_base_config is None
            else pauli_poly_to_sparse_pauli_op(h_poly_step)
        )
        return StepHamiltonianArtifacts(
            physical_time=float(physical_time),
            h_poly=h_poly_step,
            hmat=np.zeros((0, 0), dtype=complex),
            compiled_h=compiled_h_step,
            oracle_observable=oracle_observable,
            drive_term_count=int(drive_term_count),
        )

    def _step_hamiltonian_artifacts(self, time_value: float) -> StepHamiltonianArtifacts:
        if self._drive_model is not None:
            physical_time = self._physical_time(float(time_value))
            drive_coeff = float(self._drive_model.coefficient_at(float(physical_time)))
            if abs(float(drive_coeff)) <= 1.0e-15:
                return StepHamiltonianArtifacts(
                    physical_time=float(physical_time),
                    h_poly=self.h_poly,
                    hmat=self._static_hmat_for_artifacts(),
                    compiled_h=self._compiled_h,
                    oracle_observable=self._oracle_qop,
                    drive_term_count=0,
                )
            h_poly_step = self.h_poly + (float(drive_coeff) * self._drive_model.drive_poly)
            h_poly_step._reduce()
            if bool(self.strict_qpu_faithful):
                return self._strict_step_hamiltonian_artifacts_from_poly(
                    physical_time=float(physical_time),
                    h_poly_step=h_poly_step,
                    drive_term_count=int(self._drive_model.drive_term_count),
                )
            hmat_step = np.asarray(hamiltonian_matrix(h_poly_step), dtype=complex)
            compiled_h_step = compile_polynomial_action(
                h_poly_step,
                tol=1e-12,
                pauli_action_cache=self._pauli_action_cache,
            )
            return StepHamiltonianArtifacts(
                physical_time=float(physical_time),
                h_poly=h_poly_step,
                hmat=np.asarray(hmat_step, dtype=complex),
                compiled_h=compiled_h_step,
                oracle_observable=None,
                drive_term_count=int(self._drive_model.drive_term_count),
            )
        if self._drive_config is None or self._drive_coeff_provider_exyz is None:
            return StepHamiltonianArtifacts(
                physical_time=float(time_value),
                h_poly=self.h_poly,
                hmat=self._static_hmat_for_artifacts(),
                compiled_h=self._compiled_h,
                oracle_observable=self._oracle_qop,
                drive_term_count=0,
            )

        from pipelines.exact_bench.noise_oracle_runtime import pauli_poly_to_sparse_pauli_op

        physical_time = self._physical_time(float(time_value))
        if bool(self.strict_qpu_faithful):
            geom_cfg = self._measured_geometry_config()
            drive_coeff_map = {
                str(label): complex(coeff)
                for label, coeff in dict(self._drive_coeff_provider_exyz(float(physical_time))).items()
            }
            drive_poly = _pauli_poly_from_real_coeff_map(
                drive_coeff_map,
                nq=int(self._num_qubits),
                drop_abs_tol=1.0e-15,
                hermiticity_tol=float(geom_cfg.observable_hermiticity_tol),
                context="strict_qpu_faithful drive Hamiltonian",
            )
            drive_terms = list(drive_poly.return_polynomial())
            if not drive_terms:
                return StepHamiltonianArtifacts(
                    physical_time=float(physical_time),
                    h_poly=self.h_poly,
                    hmat=np.zeros((0, 0), dtype=complex),
                    compiled_h=self._compiled_h,
                    oracle_observable=self._oracle_qop,
                    drive_term_count=0,
                )
            h_poly_step = self.h_poly + drive_poly
            h_poly_step._reduce()
            kept_map = {
                str(term.pw2strng()): complex(term.p_coeff)
                for term in drive_terms
            }
            return self._strict_step_hamiltonian_artifacts_from_poly(
                physical_time=float(physical_time),
                h_poly_step=h_poly_step,
                drive_term_count=int(len(kept_map)),
            )

        from pipelines.hardcoded.hh_fixed_manifold_measured import (
            FixedManifoldMeasuredConfig,
            _build_driven_hamiltonian,
        )

        h_poly_step, hmat_step, drive_coeff_map = _build_driven_hamiltonian(
            h_poly_static=self.h_poly,
            hmat_static=self.hmat,
            drive_coeff_provider_exyz=self._drive_coeff_provider_exyz,
            physical_time=float(physical_time),
            nq=int(self._num_qubits),
            geom_cfg=FixedManifoldMeasuredConfig(),
            drive_drop_abs_tol=1.0e-15,
        )
        compiled_h_step = compile_polynomial_action(
            h_poly_step,
            tol=1e-12,
            pauli_action_cache=self._pauli_action_cache,
        )
        oracle_observable = (
            None
            if self._oracle_base_config is None
            else pauli_poly_to_sparse_pauli_op(h_poly_step)
        )
        return StepHamiltonianArtifacts(
            physical_time=float(physical_time),
            h_poly=h_poly_step,
            hmat=np.asarray(hmat_step, dtype=complex),
            compiled_h=compiled_h_step,
            oracle_observable=oracle_observable,
            drive_term_count=int(len(drive_coeff_map)),
        )

    def _predicted_displacement(self, *, dt: float, baseline: Mapping[str, Any]) -> float:
        G = np.asarray(baseline.get("G", np.zeros((0, 0), dtype=float)), dtype=float)
        theta_dot = np.asarray(baseline.get("theta_dot_step", np.zeros(0, dtype=float)), dtype=float).reshape(-1)
        if G.size == 0 or theta_dot.size == 0:
            return 0.0
        quad = float(theta_dot @ G @ theta_dot)
        return float(abs(float(dt)) * np.sqrt(max(quad, 0.0)))

    def _motion_telemetry(
        self,
        *,
        theta_dot: np.ndarray | Sequence[float],
        predicted_displacement: float,
    ) -> MotionSchedulerTelemetry:
        return _realtime_motion.build_motion_telemetry(
            cfg=self.cfg,
            theta_dot=theta_dot,
            theta_dot_history=self._theta_dot_history,
            predicted_displacement=float(predicted_displacement),
        )

    def _effective_refresh_pressure(
        self,
        *,
        base_refresh_pressure: str,
        motion: MotionSchedulerTelemetry,
    ) -> str:
        return _realtime_motion.effective_refresh_pressure(
            base_refresh_pressure=str(base_refresh_pressure),
            motion=motion,
        )

    def _shortlist_cfg_for_motion(self, motion: MotionSchedulerTelemetry) -> FullScoreConfig:
        cfg = self._active_cfg()
        base_shortlist_cfg = FullScoreConfig(
            shortlist_fraction=float(getattr(cfg, "shortlist_fraction", 0.15)),
            shortlist_size=int(getattr(cfg, "shortlist_size", 4)),
        )
        return _realtime_motion.shortlist_cfg_for_motion(
            cfg=cfg,
            base_shortlist_cfg=base_shortlist_cfg,
            motion=motion,
        )

    def _oracle_confirm_limit_for_motion(
        self,
        *,
        confirmed_count: int,
        refresh_pressure: str,
        motion: MotionSchedulerTelemetry,
    ) -> int:
        return _realtime_motion.oracle_confirm_limit_for_motion(
            confirmed_count=int(confirmed_count),
            refresh_pressure=str(refresh_pressure),
            motion=motion,
        )

    def _oracle_budget_scale_for_motion(
        self,
        *,
        refresh_pressure: str,
        motion: MotionSchedulerTelemetry,
    ) -> float:
        return _realtime_motion.oracle_budget_scale_for_motion(
            cfg=self._active_cfg(),
            refresh_pressure=str(refresh_pressure),
            motion=motion,
        )

    def _high_miss_relative(self, *, baseline: Mapping[str, Any]) -> bool:
        summary = baseline["summary"]
        rho_miss = float(getattr(summary, "rho_miss"))
        return bool(rho_miss > float(self.cfg.miss_threshold))

    def _high_miss_current(self, *, baseline: Mapping[str, Any]) -> bool:
        summary = baseline["summary"]
        epsilon_proj_sq_raw = getattr(summary, "epsilon_proj_sq", None)
        epsilon_proj_sq = (
            float("inf") if epsilon_proj_sq_raw is None else float(epsilon_proj_sq_raw)
        )
        return bool(
            self._high_miss_relative(baseline=baseline)
            and epsilon_proj_sq > float(getattr(self.cfg, "miss_abs_threshold", 0.0))
        )

    def _high_miss_active(self, *, baseline: Mapping[str, Any]) -> bool:
        current_high = bool(self._high_miss_current(baseline=baseline))
        window = max(1, int(getattr(self.cfg, "miss_persistence_window", 1)))
        count_required = max(1, int(getattr(self.cfg, "miss_persistence_count", 1)))
        recent_relative = (
            list(self._high_miss_relative_history)
            + [self._high_miss_relative(baseline=baseline)]
        )[-window:]
        persistent_high = bool(
            sum(1 for item in recent_relative if bool(item)) >= count_required
        )
        return bool(current_high or persistent_high)

    def _record_high_miss_history(self, *, baseline: Mapping[str, Any]) -> None:
        window = max(1, int(getattr(self.cfg, "miss_persistence_window", 1)))
        self._high_miss_history.append(bool(self._high_miss_current(baseline=baseline)))
        self._high_miss_history = list(self._high_miss_history[-window:])
        self._high_miss_relative_history.append(
            bool(self._high_miss_relative(baseline=baseline))
        )
        self._high_miss_relative_history = list(self._high_miss_relative_history[-window:])

    """
    lane_k =
    append, if time_stop exists and Chapter 17A high-miss is active;
    prune,  if low-miss prune candidates are available;
    stay,   otherwise.
    """
    def _controller_lane(
        self,
        *,
        time_stop: float | None,
        baseline: Mapping[str, Any],
        prune_candidates_available: bool = False,
        prune_reason: str | None = None,
    ) -> tuple[str, str]:
        if time_stop is None:
            return "stay", "terminal_checkpoint"
        if str(self.cfg.mode) == "off":
            return "stay", "controller_disabled"
        if self._high_miss_active(baseline=baseline) and not (
            self._recoverability_prune_enabled() and bool(prune_candidates_available)
        ):
            return "append", "exact_rho_miss_above_threshold"
        if bool(prune_candidates_available):
            return "prune", str(prune_reason or "prune_candidates_available")
        return "stay", str(prune_reason or "exact_rho_miss_below_threshold")

    def _record_theta_dot_history(self, theta_dot: np.ndarray | Sequence[float]) -> None:
        value = np.asarray(theta_dot, dtype=float).reshape(-1)
        self._previous_theta_dot = np.asarray(value, dtype=float)
        self._theta_dot_history.append(np.asarray(value, dtype=float))
        if len(self._theta_dot_history) > 3:
            self._theta_dot_history = list(self._theta_dot_history[-3:])

    def _oracle_sampling_targets(
        self,
        *,
        tier_name: str,
        budget_scale: float,
        floor_to_base_config: bool = False,
    ) -> tuple[int, int]:
        tier_cfg = self._oracle_tier_configs[str(tier_name)]
        scale = max(float(budget_scale), 0.25)
        base_samples = max(1, int(tier_cfg.oracle_repeats))
        base_shots = max(1, int(tier_cfg.shots))
        min_samples = max(1, int(np.ceil(float(base_samples) * float(scale))))
        min_total_shots = max(
            1,
            int(np.ceil(float(base_shots) * float(base_samples) * float(scale))),
        )
        if bool(floor_to_base_config) and self._oracle_base_config is not None:
            base_cfg_samples = max(1, int(self._oracle_base_config.oracle_repeats))
            base_cfg_shots = max(1, int(self._oracle_base_config.shots))
            min_samples = max(int(min_samples), int(base_cfg_samples))
            min_total_shots = max(
                int(min_total_shots),
                int(base_cfg_shots) * int(base_cfg_samples),
            )
        return int(min_total_shots), int(min_samples)

    def _measured_geometry_config(self) -> FixedManifoldMeasuredConfig:
        from pipelines.hardcoded.hh_fixed_manifold_measured import (
            FixedManifoldMeasuredConfig,
        )

        observable_max_terms = 512
        if (
            bool(getattr(self, "strict_qpu_faithful", False))
            and self._oracle_base_config is not None
            and str(getattr(self._oracle_base_config, "noise_mode", "")).strip().lower()
            == "ideal"
        ):
            # The local ideal oracle is an infinite-shot observable-interface
            # emulator.  It should not fail on HH recovery routes just because
            # the direct exact-state observable path can evaluate large
            # anticommutators and the measured helper's planning guard
            # defaulted to 512.  L=3 strict runs can exceed 10k terms in AHsym
            # observables.  This is a cap increase, not term dropping.
            observable_max_terms = 32768
        return FixedManifoldMeasuredConfig(
            regularization_lambda=self._cfg_float("regularization_lambda"),
            pinv_rcond=self._cfg_float("pinv_rcond"),
            observable_max_terms=int(observable_max_terms),
        )

    def _measurement_state_key(
        self,
        *,
        layout: AnsatzParameterLayout,
        theta_runtime: np.ndarray | Sequence[float],
    ) -> str:
        scaffold_labels = [str(block.candidate_label) for block in layout.blocks]
        return hash_measurement_state(
            scaffold_labels=scaffold_labels,
            logical_count=int(layout.logical_parameter_count),
            runtime_count=int(layout.runtime_parameter_count),
            theta=theta_runtime,
        )

    def _strict_measurement_active_window_size(self) -> int:
        if not bool(getattr(self, "strict_qpu_faithful", False)):
            return 0
        if self._oracle_base_config is None:
            return 0
        if str(getattr(self._oracle_base_config, "noise_mode", "")).strip().lower() != "ideal":
            return 0
        return max(0, int(getattr(self.cfg, "measurement_active_window_size", 0)))

    def _selected_measurement_runtime_indices(
        self,
        *,
        layout: AnsatzParameterLayout,
    ) -> tuple[int, ...] | None:
        window_size = int(self._strict_measurement_active_window_size())
        if window_size <= 0:
            return None
        blocks = tuple(getattr(layout, "blocks", ()))
        if not blocks or int(window_size) >= int(len(blocks)):
            return None
        selected_blocks = blocks[-int(window_size) :]
        indices: list[int] = []
        for block in selected_blocks:
            indices.extend(
                int(idx)
                for idx in range(int(block.runtime_start), int(block.runtime_stop))
            )
        selected = tuple(sorted({int(idx) for idx in indices}))
        total_runtime = int(layout.runtime_parameter_count)
        if not selected or tuple(selected) == tuple(range(total_runtime)):
            return None
        return selected

    def _strict_qpu_observable_spec_from_poly(
        self,
        *,
        name: str,
        kind: str,
        poly: PauliPolynomial,
    ) -> ObservableSpec:
        try:
            poly._reduce()
        except Exception:
            pass
        term_count = int(poly.count_number_terms())
        sparse = pauli_poly_to_sparse_pauli_op(poly, tol=1.0e-12)
        return ObservableSpec(
            name=str(name),
            kind=str(kind),
            runtime_index=None,
            runtime_pair=None,
            poly=poly,
            sparse_op=sparse,
            term_count=int(term_count),
            is_zero=bool(term_count <= 0),
        )

    def _strict_qpu_observable_specs(
        self,
    ) -> tuple[Any | None, list[ObservableSpec], dict[str, Any]]:
        n_sites = int(max(1, self._num_sites))
        family_hint = str(getattr(self, "_family_key", "hh") or "hh")
        try:
            bundle = observable_measurement_bundle_for_problem(
                resolved_problem=self.resolved_problem,
                num_sites=int(n_sites),
                ordering=str(self._ordering),
                num_qubits=int(self._num_qubits),
            )
        except Exception as exc:
            return None, [], {
                "observable_telemetry_supported": False,
                "observable_telemetry_reason": f"observable measurement bundle unavailable: {type(exc).__name__}: {exc}",
                "observable_family": family_hint,
            }
        specs = [
            self._strict_qpu_observable_spec_from_poly(
                name=str(definition.name),
                kind=str(definition.kind),
                poly=definition.poly,
            )
            for definition in tuple(bundle.definitions)
        ]
        return bundle, specs, {
            "observable_telemetry_supported": True,
            "observable_telemetry_reason": None,
            "observable_family": str(bundle.observable_family),
            "observable_telemetry_primary_density_mode": str(
                self._exact_forecast_primary_density_target_mode()
            ),
            "observable_telemetry_spec_count": int(len(specs)),
            "observable_telemetry_max_terms": int(
                max((int(spec.term_count) for spec in specs), default=0)
            ),
        }

    # Backwards-compatible private aliases for tests and older in-repo probes.
    def _strict_qpu_hh_observable_spec_from_poly(
        self,
        *,
        name: str,
        kind: str,
        poly: PauliPolynomial,
    ) -> ObservableSpec:
        return self._strict_qpu_observable_spec_from_poly(
            name=str(name),
            kind=str(kind),
            poly=poly,
        )

    def _strict_qpu_hh_observable_specs(self) -> tuple[list[ObservableSpec], dict[str, Any]]:
        _bundle, specs, meta = self._strict_qpu_observable_specs()
        return specs, dict(meta)

    def _strict_observable_unsupported_payload(
        self,
        *,
        family: str,
        reason: str,
        meta: Mapping[str, Any] | None = None,
        estimates: Mapping[str, Any] | None = None,
        backend_info: Mapping[str, Any] | None = None,
        backend_info_count: int = 0,
    ) -> dict[str, Any]:
        return {
            **dict(meta or {}),
            "observable_telemetry_supported": False,
            "observable_telemetry_reason": str(reason),
            "observable_telemetry_kind": "oracle_measured",
            "observable_telemetry_noise_mode": (
                None
                if self._oracle_base_config is None
                else str(self._oracle_base_config.noise_mode)
            ),
            "observable_telemetry_backend_info": dict(backend_info or {}),
            "observable_telemetry_backend_info_count": int(backend_info_count),
            "observable_telemetry_estimates": dict(estimates or {}),
            "observable_family": str(family),
            "site_occupations": [],
            "site_occupations_up": [],
            "site_occupations_dn": [],
            "n_up_site": [],
            "n_dn_site": [],
            "staggered": None,
            "doublon": None,
            "primary_density": None,
        }

    def _strict_qpu_measured_observable_telemetry(
        self,
        *,
        checkpoint_ctx: Any,
        raw_group_pool: BackendScheduledRawGroupPool | None,
        layout: AnsatzParameterLayout,
        theta_runtime: np.ndarray | Sequence[float],
        tier_name: str,
        budget_scale: float,
    ) -> dict[str, Any]:
        bundle, specs, meta = self._strict_qpu_observable_specs()
        family = str(meta.get("observable_family", getattr(self, "_family_key", "hh")))
        if bundle is None or not bool(meta.get("observable_telemetry_supported", False)):
            return self._strict_observable_unsupported_payload(
                family=family,
                reason=str(meta.get("observable_telemetry_reason", "observable telemetry unavailable")),
                meta=meta,
            )
        if self._oracle_base_config is None:
            return self._strict_observable_unsupported_payload(
                family=family,
                reason="oracle_base_config unavailable",
                meta=meta,
            )
        theta_arr = np.asarray(theta_runtime, dtype=float).reshape(-1)
        circuit = build_runtime_layout_circuit(
            layout,
            theta_arr,
            int(self._num_qubits),
            reference_state=np.asarray(self.replay_context.psi_ref, dtype=complex).reshape(-1),
        )
        oracle = self._oracle_for_tier(str(tier_name))
        min_total_shots, min_samples = self._oracle_sampling_targets(
            tier_name=str(tier_name),
            budget_scale=float(budget_scale),
            floor_to_base_config=True,
        )
        state_key = self._measurement_state_key(layout=layout, theta_runtime=theta_arr)
        measured = estimate_observable_specs(
            oracle=oracle,
            raw_group_pool=raw_group_pool,
            circuit=circuit,
            specs=specs,
            observable_family_prefix="strict_observable_telemetry",
            candidate_label=None,
            position_id=None,
            state_key=str(state_key),
            min_total_shots=int(min_total_shots),
            min_samples=int(min_samples),
            zero_abs_tol=max(0.0, 10.0 * float(self._measured_geometry_config().observable_drop_abs_tol)),
        )
        estimates = dict(measured.get("observable_estimates", {}))
        try:
            snapshot = measured_snapshot_from_estimates(
                bundle,
                estimates,
                resolved_problem=self.resolved_problem,
                num_sites=int(max(1, self._num_sites)),
                requested_primary_density_mode=str(
                    self._exact_forecast_primary_density_target_mode()
                ),
            )
        except Exception as exc:
            return self._strict_observable_unsupported_payload(
                family=family,
                reason=f"measured snapshot reconstruction failed: {type(exc).__name__}: {exc}",
                meta=meta,
                estimates=estimates,
                backend_info=dict(measured.get("backend_info", {})),
                backend_info_count=int(measured.get("backend_info_count", 0)),
            )
        return {
            **dict(meta),
            "observable_telemetry_supported": True,
            "observable_telemetry_reason": None,
            "observable_telemetry_kind": "oracle_measured",
            "observable_telemetry_noise_mode": str(self._oracle_base_config.noise_mode),
            "observable_telemetry_backend_info": dict(measured.get("backend_info", {})),
            "observable_telemetry_backend_info_count": int(
                measured.get("backend_info_count", 0)
            ),
            "observable_telemetry_estimates": estimates,
            **dict(snapshot),
            "observable_family": str(snapshot.get("observable_family", family)),
            "site_occupations_up": list(snapshot.get("site_occupations_up", snapshot.get("n_up_site", []))),
            "site_occupations_dn": list(snapshot.get("site_occupations_dn", snapshot.get("n_dn_site", []))),
        }

    def _strict_qpu_hh_measured_observable_telemetry(
        self,
        *,
        checkpoint_ctx: Any,
        raw_group_pool: BackendScheduledRawGroupPool | None,
        layout: AnsatzParameterLayout,
        theta_runtime: np.ndarray | Sequence[float],
        tier_name: str,
        budget_scale: float,
    ) -> dict[str, Any]:
        return self._strict_qpu_measured_observable_telemetry(
            checkpoint_ctx=checkpoint_ctx,
            raw_group_pool=raw_group_pool,
            layout=layout,
            theta_runtime=theta_runtime,
            tier_name=str(tier_name),
            budget_scale=float(budget_scale),
        )

    def _candidate_step_scales(self) -> tuple[float, ...]:
        raw_values = tuple(getattr(self.cfg, "candidate_step_scales", (1.0,)))
        out: list[float] = []
        seen: set[float] = set()
        for raw in raw_values:
            value = float(raw)
            if (not np.isfinite(value)) or value <= 0.0:
                continue
            rounded = round(value, 12)
            if rounded in seen:
                continue
            seen.add(rounded)
            out.append(value)
        return tuple(out) if out else (1.0,)

    def _candidate_step_scales_for_selected(
        self,
        *,
        selected: Mapping[str, Any],
        time_stop: float | None,
    ) -> tuple[float, ...]:
        del selected, time_stop
        return tuple(self._candidate_step_scales())

    def _drive_aligned_baseline_step_scales(self) -> tuple[float, ...]:
        raw_values = (0.0, 0.05, 0.1) + tuple(self._candidate_step_scales())
        out: list[float] = []
        seen: set[float] = set()
        for raw in raw_values:
            value = float(raw)
            if (not np.isfinite(value)) or value < 0.0:
                continue
            rounded = round(value, 12)
            if rounded in seen:
                continue
            seen.add(rounded)
            out.append(value)
        return tuple(out) if out else (0.0, 1.0)

    def _drive_aligned_baseline_step_scales_for_time(self, *, time_stop: float | None) -> tuple[float, ...]:
        del time_stop
        return tuple(self._drive_aligned_baseline_step_scales())

    def _candidate_scale_tag(self, scale: float) -> str:
        text = f"{float(scale):.6f}".rstrip("0").rstrip(".")
        if text == "":
            text = "1"
        return text.replace("-", "m").replace(".", "p")

    def _exact_forecast_baseline_step_refine_rounds(self) -> int:
        return max(0, int(getattr(self.cfg, "exact_forecast_baseline_step_refine_rounds", 0)))

    def _exact_forecast_baseline_blend_weights(self) -> tuple[float, ...]:
        raw = tuple(float(x) for x in getattr(self.cfg, "exact_forecast_baseline_blend_weights", ()))
        if not raw:
            return (0.0,)
        out: list[float] = []
        seen: set[float] = set()
        for value in raw:
            weight = float(value)
            rounded = round(weight, 12)
            if rounded in seen:
                continue
            seen.add(rounded)
            out.append(weight)
        return tuple(out) if out else (0.0,)

    def _exact_forecast_baseline_gain_scales(self) -> tuple[float, ...]:
        raw = tuple(float(x) for x in getattr(self.cfg, "exact_forecast_baseline_gain_scales", ()))
        if not raw:
            return (1.0,)
        out: list[float] = []
        seen: set[float] = set()
        saw_one = False
        for value in raw:
            scale = float(value)
            if scale <= 0.0:
                continue
            rounded = round(scale, 12)
            if rounded in seen:
                continue
            seen.add(rounded)
            out.append(scale)
            if abs(scale - 1.0) <= 1.0e-12:
                saw_one = True
        if not saw_one:
            out.insert(0, 1.0)
        return tuple(out) if out else (1.0,)

    def _exact_forecast_include_tangent_secant_proposal(self) -> bool:
        return False

    def _exact_forecast_tangent_secant_trust_radius(self) -> float:
        return max(
            0.0,
            float(
                getattr(self.cfg, "exact_forecast_tangent_secant_trust_radius", 0.0)
            ),
        )

    def _exact_forecast_tangent_secant_signed_energy_lead_limit(self) -> float:
        if self._exact_v1_fidelity_first_barrier_ranking_active():
            return 0.0
        return max(
            0.0,
            float(
                getattr(
                    self.cfg,
                    "exact_forecast_tangent_secant_signed_energy_lead_limit",
                    0.0,
                )
            ),
        )

    def _exact_forecast_baseline_proposal_mode(self) -> str:
        return str(
            getattr(self.cfg, "exact_forecast_baseline_proposal_mode", "norm_locked_blend_v1")
        ).strip().lower()

    def _proposal_metric_norm(
        self,
        *,
        baseline: Mapping[str, Any] | None,
        theta_dot: np.ndarray | Sequence[float],
    ) -> float:
        vec = np.asarray(theta_dot, dtype=float).reshape(-1)
        if vec.size <= 0:
            return 0.0
        if baseline is not None:
            G = np.asarray(baseline.get("G", np.zeros((0, 0), dtype=float)), dtype=float)
            if G.size and G.shape == (vec.size, vec.size):
                quad = float(vec @ G @ vec)
                if np.isfinite(quad) and quad > 1.0e-18:
                    return float(np.sqrt(max(quad, 0.0)))
        norm = float(np.linalg.norm(vec))
        return float(norm if np.isfinite(norm) else 0.0)

    def _normalize_proposal_direction(
        self,
        *,
        baseline: Mapping[str, Any] | None,
        theta_dot: np.ndarray | Sequence[float],
    ) -> tuple[np.ndarray | None, float]:
        vec = np.asarray(theta_dot, dtype=float).reshape(-1)
        norm = float(self._proposal_metric_norm(baseline=baseline, theta_dot=vec))
        if norm <= 1.0e-10:
            return None, float(norm)
        return np.asarray(vec / float(norm), dtype=float).reshape(-1), float(norm)

    def _lookahead_drive_baseline(
        self,
        *,
        checkpoint_index: int,
    ) -> dict[str, Any] | None:
        if not bool(self._drive_aligned_density_active):
            return None
        next_idx = int(checkpoint_index) + 1
        if next_idx >= int(self.times.size):
            return None
        future_time_start = float(self.times[int(next_idx)])
        future_time_stop = (
            None
            if int(next_idx) + 1 >= int(self.times.size)
            else float(self.times[int(next_idx) + 1])
        )
        future_sample_time = self._projection_sample_time(
            float(future_time_start),
            future_time_stop,
        )
        future_step_hamiltonian = self._step_hamiltonian_artifacts(float(future_sample_time))
        psi_current = self.current_executor.prepare_state(self.current_theta, self.replay_context.psi_ref)
        checkpoint_ctx = make_checkpoint_context(
            checkpoint_index=int(next_idx),
            time_start=float(future_time_start),
            time_stop=(None if future_time_stop is None else float(future_time_stop)),
            scaffold_labels=self._current_scaffold_labels(),
            theta=self.current_theta,
            psi=psi_current,
            logical_count=int(self.current_layout.logical_parameter_count),
            runtime_count=int(self.current_layout.runtime_parameter_count),
            resolved_family=str(self.replay_context.family_info.get("resolved", "unknown")),
            grouping_mode=str(self.cfg.grouping_mode),
            structure_locked=False,
        )
        cache = ExactCheckpointValueCache(
            checkpoint_id=str(checkpoint_ctx.checkpoint_id),
            grouping_mode=str(self.cfg.grouping_mode),
        )
        return self._compute_baseline_geometry_for_runtime_state(
            checkpoint_ctx=checkpoint_ctx,
            cache=cache,
            executor=self.current_executor,
            layout=self.current_layout,
            theta_runtime=self.current_theta,
            planning_audit=self._planning_audit,
            step_hamiltonian=future_step_hamiltonian,
        )

    def _exact_tangent_secant_proposal(
        self,
        *,
        baseline: Mapping[str, Any] | None,
        dt: float,
        time_stop: float,
    ) -> dict[str, Any] | None:
        del baseline, dt, time_stop
        return None

    def _baseline_theta_dot_proposals(
        self,
        *,
        checkpoint_index: int | None,
        baseline_theta_dot: np.ndarray | Sequence[float],
        baseline: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        baseline_vec = np.asarray(baseline_theta_dot, dtype=float).reshape(-1)
        baseline_direction, baseline_norm = self._normalize_proposal_direction(
            baseline=baseline,
            theta_dot=baseline_vec,
        )
        drive_theta_dot = self._drive_only_theta_dot_from_baseline(baseline=baseline)
        drive_direction, drive_norm = self._normalize_proposal_direction(
            baseline=baseline,
            theta_dot=(np.zeros_like(baseline_vec) if drive_theta_dot is None else drive_theta_dot),
        )
        lookahead_baseline = (
            None
            if checkpoint_index is None
            else self._lookahead_drive_baseline(checkpoint_index=int(checkpoint_index))
        )
        lookahead_drive_theta_dot = (
            None
            if lookahead_baseline is None
            else self._drive_only_theta_dot_from_baseline(baseline=lookahead_baseline)
        )
        lookahead_direction, lookahead_norm = self._normalize_proposal_direction(
            baseline=baseline,
            theta_dot=(
                np.zeros_like(baseline_vec)
                if lookahead_drive_theta_dot is None
                else lookahead_drive_theta_dot
            ),
        )

        proposals: list[dict[str, Any]] = []
        seen: set[tuple[float, ...]] = set()

        def _append(kind: str, vec: np.ndarray | None, *, blend_weight: float | None = None) -> None:
            if vec is None:
                return
            normed, _ = self._normalize_proposal_direction(baseline=baseline, theta_dot=vec)
            if normed is None:
                return
            key = tuple(np.round(np.asarray(normed, dtype=float).reshape(-1), 12).tolist())
            if key in seen:
                return
            seen.add(key)
            proposals.append(
                {
                    "proposal_kind": str(kind),
                    "blend_weight": (
                        None if blend_weight is None else float(blend_weight)
                    ),
                    "theta_dot_direction": np.asarray(normed, dtype=float).reshape(-1),
                    "current_baseline_norm": float(baseline_norm),
                    "current_drive_norm": float(drive_norm),
                    "lookahead_drive_norm": float(lookahead_norm),
                }
            )

        _append("baseline_current", baseline_direction, blend_weight=0.0)
        _append("drive_only_current", drive_direction, blend_weight=None)
        _append("drive_only_lookahead", lookahead_direction, blend_weight=None)
        if baseline_direction is not None and lookahead_direction is not None:
            for blend_weight in self._exact_forecast_baseline_blend_weights():
                blended = np.asarray(
                    baseline_direction + float(blend_weight) * lookahead_direction,
                    dtype=float,
                ).reshape(-1)
                _append(
                    "baseline_drive_lookahead_blend",
                    blended,
                    blend_weight=float(blend_weight),
                )
        if not proposals:
            _append("baseline_current", baseline_vec, blend_weight=0.0)
        return proposals

    def _drive_aligned_runtime_indices(
        self,
        *,
        layout: AnsatzParameterLayout | None = None,
    ) -> tuple[int, ...]:
        if not bool(self._drive_aligned_density_active) or self._drive_aligned_density_label is None:
            return tuple()
        active_layout = self.current_layout if layout is None else layout
        target_label = str(self._drive_aligned_density_label)
        out: list[int] = []
        for block in active_layout.blocks:
            block_label = str(block.candidate_label)
            if block_label != target_label and not block_label.startswith(f"{target_label}__r"):
                continue
            out.extend(range(int(block.runtime_start), int(block.runtime_stop)))
        return tuple(out)

    def _drive_only_theta_dot_from_baseline(
        self,
        *,
        baseline: Mapping[str, Any] | None,
        layout: AnsatzParameterLayout | None = None,
    ) -> np.ndarray | None:
        if baseline is None:
            return None
        runtime_indices = self._drive_aligned_runtime_indices(layout=layout)
        if not runtime_indices:
            return None
        theta_dot_step = np.asarray(baseline.get("theta_dot_step", np.zeros(0, dtype=float)), dtype=float).reshape(-1)
        K = np.asarray(baseline.get("K", np.zeros((0, 0), dtype=float)), dtype=float)
        f = np.asarray(baseline.get("f", np.zeros(0, dtype=float)), dtype=float).reshape(-1)
        if theta_dot_step.size <= max(runtime_indices) or f.size <= max(runtime_indices):
            return None
        if K.ndim != 2 or K.shape[0] <= max(runtime_indices) or K.shape[1] <= max(runtime_indices):
            return None
        idx = np.asarray(runtime_indices, dtype=int)
        K_drive = np.asarray(K[np.ix_(idx, idx)], dtype=float)
        f_drive = np.asarray(f[idx], dtype=float).reshape(-1)
        if K_drive.size == 0 or f_drive.size == 0:
            return None
        K_drive_pinv = np.linalg.pinv(K_drive, rcond=self._cfg_float("pinv_rcond"))
        theta_dot_drive_block = np.asarray(K_drive_pinv @ f_drive, dtype=float).reshape(-1)
        theta_dot_drive = np.zeros_like(theta_dot_step, dtype=float)
        theta_dot_drive[idx] = theta_dot_drive_block
        return np.asarray(theta_dot_drive, dtype=float).reshape(-1)

    def _blend_theta_dot_with_drive_direction(
        self,
        *,
        baseline_theta_dot: np.ndarray | Sequence[float],
        drive_theta_dot: np.ndarray | Sequence[float] | None,
        blend_weight: float,
        baseline: Mapping[str, Any] | None,
    ) -> np.ndarray:
        baseline_vec = np.asarray(baseline_theta_dot, dtype=float).reshape(-1)
        weight = float(blend_weight)
        if drive_theta_dot is None or abs(float(weight)) <= 1.0e-15:
            return np.asarray(baseline_vec, dtype=float).reshape(-1)
        drive_vec = np.asarray(drive_theta_dot, dtype=float).reshape(-1)
        if drive_vec.shape != baseline_vec.shape:
            return np.asarray(baseline_vec, dtype=float).reshape(-1)
        G = None if baseline is None else np.asarray(baseline.get("G", np.zeros((0, 0), dtype=float)), dtype=float)
        use_metric = bool(
            G is not None
            and G.size
            and G.shape[0] == baseline_vec.size
            and G.shape[1] == baseline_vec.size
        )

        def _inner(lhs: np.ndarray, rhs: np.ndarray) -> float:
            if use_metric:
                return float(lhs @ G @ rhs)
            return float(lhs @ rhs)

        def _quad(vec: np.ndarray) -> float:
            return float(_inner(vec, vec))

        baseline_quad = _quad(baseline_vec)
        if baseline_quad <= 1.0e-18:
            return np.asarray(baseline_vec, dtype=float).reshape(-1)

        # Add only the drive direction component that is new relative to the
        # baseline McLachlan flow, then renormalize so forecast scoring compares
        # direction changes rather than norm inflation. Negative blend weights
        # are allowed and subtract the residual drive direction.
        residual = np.asarray(drive_vec, dtype=float).reshape(-1)
        overlap = _inner(baseline_vec, residual)
        residual = residual - (float(overlap) / float(baseline_quad)) * baseline_vec
        residual_quad = _quad(residual)
        if residual_quad <= 1.0e-18:
            return np.asarray(baseline_vec, dtype=float).reshape(-1)
        residual = np.asarray(
            np.sqrt(float(baseline_quad) / float(residual_quad)) * residual,
            dtype=float,
        ).reshape(-1)
        blended = np.asarray(baseline_vec + weight * residual, dtype=float).reshape(-1)
        blended_quad = _quad(blended)
        if blended_quad <= 1.0e-18:
            return np.asarray(baseline_vec, dtype=float).reshape(-1)
        blended = np.asarray(
            np.sqrt(float(baseline_quad) / float(blended_quad)) * blended,
            dtype=float,
        ).reshape(-1)
        return np.asarray(blended, dtype=float).reshape(-1)

    def _baseline_theta_dot_candidates(
        self,
        *,
        baseline_theta_dot: np.ndarray | Sequence[float],
        baseline: Mapping[str, Any] | None = None,
    ) -> list[tuple[float, np.ndarray]]:
        baseline_vec = np.asarray(baseline_theta_dot, dtype=float).reshape(-1)
        drive_theta_dot = self._drive_only_theta_dot_from_baseline(baseline=baseline)
        out: list[tuple[float, np.ndarray]] = []
        seen: set[tuple[float, ...]] = set()
        for blend_weight in self._exact_forecast_baseline_blend_weights():
            candidate = self._blend_theta_dot_with_drive_direction(
                baseline_theta_dot=baseline_vec,
                drive_theta_dot=drive_theta_dot,
                blend_weight=float(blend_weight),
                baseline=baseline,
            )
            key = tuple(np.round(np.asarray(candidate, dtype=float).reshape(-1), 12).tolist())
            if key in seen:
                continue
            seen.add(key)
            out.append((float(blend_weight), np.asarray(candidate, dtype=float).reshape(-1)))
        if not out:
            out.append((0.0, np.asarray(baseline_vec, dtype=float).reshape(-1)))
        return out

    def _select_exact_v1_baseline_step_scale(
        self,
        *,
        checkpoint_index: int | None = None,
        baseline_theta_dot: np.ndarray | Sequence[float],
        baseline: Mapping[str, Any] | None = None,
        dt: float,
        time_stop: float,
        debug_variants: list[dict[str, Any]] | None = None,
    ) -> tuple[np.ndarray, float, float, float, dict[str, Any]]:
        best_candidate_theta_dot: np.ndarray | None = None
        best_step_theta_dot: np.ndarray | None = None
        best_step_forecast: dict[str, Any] | None = None
        best_step_score: float | None = None
        best_scale: float | None = None
        best_blend_weight: float | None = None
        best_proposal_kind: str | None = None
        best_current_baseline_norm: float | None = None
        best_current_drive_norm: float | None = None
        best_lookahead_drive_norm: float | None = None
        best_tangent_secant_displacement_norm: float | None = None
        best_tangent_secant_projection_quality: float | None = None
        best_tangent_secant_raw_metric_norm: float | None = None
        best_tangent_secant_metric_norm: float | None = None
        best_tangent_secant_current_energy_bias: float | None = None
        best_tangent_secant_next_exact_energy_delta: float | None = None
        best_tangent_secant_signed_energy_lead: float | None = None
        best_tangent_secant_signed_energy_lead_limit: float | None = None
        best_tangent_secant_signed_energy_lead_taper: float | None = None
        evaluated: dict[
            tuple[float, float, tuple[float, ...]],
            tuple[np.ndarray, dict[str, Any], float],
        ] = {}

        def _record_debug_variant(
            *,
            stage: str,
            proposal_kind: str,
            blend_weight: float,
            step_scale: float,
            gain_scale: float,
            theta_dot_direction: np.ndarray,
            scaled_theta_dot: np.ndarray,
            forecast: Mapping[str, Any],
            score: float,
            current_baseline_norm: float | None,
            current_drive_norm: float | None,
            lookahead_drive_norm: float | None,
        ) -> None:
            if debug_variants is None:
                return
            live_d_breakdown = (
                self._exact_v1_live_d_score_breakdown(forecast=forecast)
                if self._exact_v1_d_shape_barrier_ranking_active()
                else None
            )
            debug_variants.append(
                {
                    "stage": str(stage),
                    "proposal_kind": str(proposal_kind),
                    "blend_weight": float(blend_weight),
                    "step_scale": float(step_scale),
                    "gain_scale": float(gain_scale),
                    "tracking_score_total": float(score),
                    "current_baseline_norm": (
                        None if current_baseline_norm is None else float(current_baseline_norm)
                    ),
                    "current_drive_norm": (
                        None if current_drive_norm is None else float(current_drive_norm)
                    ),
                    "lookahead_drive_norm": (
                        None if lookahead_drive_norm is None else float(lookahead_drive_norm)
                    ),
                    "theta_dot_direction_norm": float(
                        np.linalg.norm(np.asarray(theta_dot_direction, dtype=float).reshape(-1))
                    ),
                    "scaled_theta_dot_norm": float(
                        np.linalg.norm(np.asarray(scaled_theta_dot, dtype=float).reshape(-1))
                    ),
                    "forecast_fidelity_exact_next": float(
                        forecast.get("fidelity_exact_next", float("nan"))
                    ),
                    "forecast_abs_energy_total_error_next": float(
                        forecast.get("abs_energy_total_error_next", float("nan"))
                    ),
                    "forecast_abs_primary_density_error_next": float(
                        forecast.get(
                            "abs_primary_density_error_next",
                            forecast.get("abs_staggered_error_next", float("nan")),
                        )
                    ),
                    "forecast_site_occupations_abs_error_max_next": float(
                        forecast.get("site_occupations_abs_error_max_next", float("nan"))
                    ),
                    "live_d_core": (
                        None if live_d_breakdown is None else dict(live_d_breakdown["core"])
                    ),
                    "live_d_barrier": (
                        None if live_d_breakdown is None else dict(live_d_breakdown["barrier"])
                    ),
                    "live_d_total": (
                        None if live_d_breakdown is None else float(live_d_breakdown["total"])
                    ),
                }
            )

        def _evaluate_step(
            blend_weight: float,
            theta_dot: np.ndarray,
            step_scale: float,
        ) -> tuple[np.ndarray, dict[str, Any], float]:
            cache_key = (
                round(float(blend_weight), 12),
                round(float(step_scale), 12),
                tuple(np.round(np.asarray(theta_dot, dtype=float).reshape(-1), 12).tolist()),
            )
            cached = evaluated.get(cache_key)
            if cached is not None:
                return cached
            scaled_theta_dot = float(step_scale) * np.asarray(theta_dot, dtype=float).reshape(-1)
            theta_runtime = np.asarray(
                self.current_theta + float(dt) * np.asarray(scaled_theta_dot, dtype=float),
                dtype=float,
            ).reshape(-1)
            forecast, _forecast_rollout, score = self._local_projective_forecast_rollout(
                checkpoint_index=checkpoint_index,
                time_stop=float(time_stop),
                executor=self.current_executor,
                layout=self.current_layout,
                theta_runtime_start=theta_runtime,
                theta_dot_step=np.asarray(scaled_theta_dot, dtype=float).reshape(-1),
                planning_audit=self._planning_audit,
                scaffold_labels=self._current_scaffold_labels(),
                immediate_gain_ratio=None,
                anchor_summary=(None if baseline is None else baseline.get("summary")),
                anchor_predicted_displacement=(
                    None
                    if baseline is None
                    else self._predicted_displacement(dt=float(dt), baseline=baseline)
                ),
            )
            cached = (
                np.asarray(scaled_theta_dot, dtype=float).reshape(-1),
                dict(forecast),
                float(score),
            )
            evaluated[cache_key] = cached
            return cached

        def _consider_step(
            proposal: Mapping[str, Any],
            proposal_kind: str,
            current_baseline_norm: float | None,
            current_drive_norm: float | None,
            lookahead_drive_norm: float | None,
            blend_weight: float,
            theta_dot: np.ndarray,
            step_scale: float,
        ) -> None:
            nonlocal best_candidate_theta_dot
            nonlocal best_step_theta_dot
            nonlocal best_step_forecast
            nonlocal best_step_score
            nonlocal best_scale
            nonlocal best_blend_weight
            nonlocal best_proposal_kind
            nonlocal best_current_baseline_norm
            nonlocal best_current_drive_norm
            nonlocal best_lookahead_drive_norm
            nonlocal best_tangent_secant_displacement_norm
            nonlocal best_tangent_secant_projection_quality
            nonlocal best_tangent_secant_raw_metric_norm
            nonlocal best_tangent_secant_metric_norm
            nonlocal best_tangent_secant_current_energy_bias
            nonlocal best_tangent_secant_next_exact_energy_delta
            nonlocal best_tangent_secant_signed_energy_lead
            nonlocal best_tangent_secant_signed_energy_lead_limit
            nonlocal best_tangent_secant_signed_energy_lead_taper
            scaled_theta_dot, forecast, score = _evaluate_step(
                float(blend_weight),
                theta_dot,
                float(step_scale),
            )
            _record_debug_variant(
                stage="step",
                proposal_kind=str(proposal_kind),
                blend_weight=float(blend_weight),
                step_scale=float(step_scale),
                gain_scale=1.0,
                theta_dot_direction=np.asarray(theta_dot, dtype=float).reshape(-1),
                scaled_theta_dot=np.asarray(scaled_theta_dot, dtype=float).reshape(-1),
                forecast=dict(forecast),
                score=float(score),
                current_baseline_norm=current_baseline_norm,
                current_drive_norm=current_drive_norm,
                lookahead_drive_norm=lookahead_drive_norm,
            )
            choose = False
            if (
                best_candidate_theta_dot is None
                or best_step_theta_dot is None
                or best_step_forecast is None
                or best_step_score is None
                or best_scale is None
                or best_blend_weight is None
            ):
                choose = True
            elif float(score) < float(best_step_score) - 1.0e-12:
                choose = True
            elif abs(float(score) - float(best_step_score)) <= 1.0e-12:
                if float(blend_weight) < float(best_blend_weight) - 1.0e-12:
                    choose = True
                elif (
                    abs(float(blend_weight) - float(best_blend_weight)) <= 1.0e-12
                ) and float(step_scale) < float(best_scale):
                    choose = True
            if choose:
                best_candidate_theta_dot = np.asarray(theta_dot, dtype=float).reshape(-1)
                best_step_theta_dot = np.asarray(scaled_theta_dot, dtype=float).reshape(-1)
                best_step_forecast = dict(forecast)
                best_step_score = float(score)
                best_scale = float(step_scale)
                best_blend_weight = float(blend_weight)
                best_proposal_kind = str(proposal_kind)
                best_current_baseline_norm = (
                    None if current_baseline_norm is None else float(current_baseline_norm)
                )
                best_current_drive_norm = (
                    None if current_drive_norm is None else float(current_drive_norm)
                )
                best_lookahead_drive_norm = (
                    None if lookahead_drive_norm is None else float(lookahead_drive_norm)
                )
                best_tangent_secant_displacement_norm = (
                    None
                    if proposal.get("tangent_secant_displacement_norm") is None
                    else float(proposal["tangent_secant_displacement_norm"])
                )
                best_tangent_secant_projection_quality = (
                    None
                    if proposal.get("tangent_secant_projection_quality") is None
                    else float(proposal["tangent_secant_projection_quality"])
                )
                best_tangent_secant_raw_metric_norm = (
                    None
                    if proposal.get("tangent_secant_raw_metric_norm") is None
                    else float(proposal["tangent_secant_raw_metric_norm"])
                )
                best_tangent_secant_metric_norm = (
                    None
                    if proposal.get("tangent_secant_metric_norm") is None
                    else float(proposal["tangent_secant_metric_norm"])
                )
                best_tangent_secant_current_energy_bias = (
                    None
                    if proposal.get("tangent_secant_current_energy_bias") is None
                    else float(proposal["tangent_secant_current_energy_bias"])
                )
                best_tangent_secant_next_exact_energy_delta = (
                    None
                    if proposal.get("tangent_secant_next_exact_energy_delta") is None
                    else float(proposal["tangent_secant_next_exact_energy_delta"])
                )
                best_tangent_secant_signed_energy_lead = (
                    None
                    if proposal.get("tangent_secant_signed_energy_lead") is None
                    else float(proposal["tangent_secant_signed_energy_lead"])
                )
                best_tangent_secant_signed_energy_lead_limit = (
                    None
                    if proposal.get("tangent_secant_signed_energy_lead_limit") is None
                    else float(proposal["tangent_secant_signed_energy_lead_limit"])
                )
                best_tangent_secant_signed_energy_lead_taper = (
                    None
                    if proposal.get("tangent_secant_signed_energy_lead_taper") is None
                    else float(proposal["tangent_secant_signed_energy_lead_taper"])
                )
        proposal_mode = self._exact_forecast_baseline_proposal_mode()
        proposal_records: list[dict[str, Any]] = []
        if (
            proposal_mode == "anticipatory_drive_basis_v1"
            and checkpoint_index is not None
            and bool(self._drive_aligned_density_active)
        ):
            proposal_records = self._baseline_theta_dot_proposals(
                checkpoint_index=int(checkpoint_index),
                baseline_theta_dot=np.asarray(baseline_theta_dot, dtype=float).reshape(-1),
                baseline=baseline,
            )
        else:
            baseline_norm = self._proposal_metric_norm(
                baseline=baseline,
                theta_dot=np.asarray(baseline_theta_dot, dtype=float).reshape(-1),
            )
            drive_theta_dot = self._drive_only_theta_dot_from_baseline(baseline=baseline)
            drive_norm = self._proposal_metric_norm(
                baseline=baseline,
                theta_dot=(
                    np.zeros_like(np.asarray(baseline_theta_dot, dtype=float).reshape(-1))
                    if drive_theta_dot is None
                    else drive_theta_dot
                ),
            )
            base_candidates = self._baseline_theta_dot_candidates(
                baseline_theta_dot=np.asarray(baseline_theta_dot, dtype=float).reshape(-1),
                baseline=baseline,
            )
            for blend_weight, theta_dot in base_candidates:
                proposal_records.append(
                    {
                        "proposal_kind": "norm_locked_blend_v1",
                        "blend_weight": float(blend_weight),
                        "theta_dot_direction": np.asarray(theta_dot, dtype=float).reshape(-1),
                        "current_baseline_norm": float(baseline_norm),
                        "current_drive_norm": float(drive_norm),
                        "lookahead_drive_norm": None,
                    }
                )
        if self._exact_forecast_include_tangent_secant_proposal():
            secant_proposal = self._exact_tangent_secant_proposal(
                baseline=baseline,
                dt=float(dt),
                time_stop=float(time_stop),
            )
            if secant_proposal is not None:
                proposal_records.append(dict(secant_proposal))
        step_scale_candidates = list(
            self._drive_aligned_baseline_step_scales_for_time(time_stop=time_stop)
        )
        positive_scales = [
            float(scale) for scale in step_scale_candidates if float(scale) > 1.0e-12
        ]
        if positive_scales:
            step_scale_candidates = positive_scales
        for proposal in proposal_records:
            blend_weight = float(proposal.get("blend_weight", 0.0) or 0.0)
            theta_dot = np.asarray(proposal["theta_dot_direction"], dtype=float).reshape(-1)
            for step_scale in step_scale_candidates:
                _consider_step(
                    proposal,
                    str(proposal.get("proposal_kind", "norm_locked_blend_v1")),
                    proposal.get("current_baseline_norm"),
                    proposal.get("current_drive_norm"),
                    proposal.get("lookahead_drive_norm"),
                    float(blend_weight),
                    theta_dot,
                    float(step_scale),
                )
        for _ in range(self._exact_forecast_baseline_step_refine_rounds()):
            if (
                best_scale is None
                or best_blend_weight is None
                or best_candidate_theta_dot is None
            ):
                break
            known_scales = sorted(
                {
                    float(cache_key[1])
                    for cache_key in evaluated
                    if (
                        abs(float(cache_key[0]) - float(best_blend_weight)) <= 1.0e-12
                    )
                }
            )
            try:
                idx = known_scales.index(round(float(best_scale), 12))
            except ValueError:
                break
            proposals: list[float] = []
            if idx > 0:
                proposals.append(0.5 * (known_scales[idx - 1] + float(best_scale)))
            if idx + 1 < len(known_scales):
                proposals.append(0.5 * (float(best_scale) + known_scales[idx + 1]))
            new_proposal_found = False
            for step_scale in proposals:
                cache_key = (
                    round(float(best_blend_weight), 12),
                    round(float(step_scale), 12),
                    tuple(
                        np.round(np.asarray(best_candidate_theta_dot, dtype=float).reshape(-1), 12).tolist()
                    ),
                )
                if cache_key in evaluated:
                    continue
                new_proposal_found = True
                _consider_step(
                    {
                        "tangent_secant_displacement_norm": best_tangent_secant_displacement_norm,
                        "tangent_secant_projection_quality": best_tangent_secant_projection_quality,
                        "tangent_secant_raw_metric_norm": best_tangent_secant_raw_metric_norm,
                        "tangent_secant_metric_norm": best_tangent_secant_metric_norm,
                        "tangent_secant_current_energy_bias": best_tangent_secant_current_energy_bias,
                        "tangent_secant_next_exact_energy_delta": best_tangent_secant_next_exact_energy_delta,
                        "tangent_secant_signed_energy_lead": best_tangent_secant_signed_energy_lead,
                        "tangent_secant_signed_energy_lead_limit": best_tangent_secant_signed_energy_lead_limit,
                        "tangent_secant_signed_energy_lead_taper": best_tangent_secant_signed_energy_lead_taper,
                    },
                    str(best_proposal_kind),
                    best_current_baseline_norm,
                    best_current_drive_norm,
                    best_lookahead_drive_norm,
                    float(best_blend_weight),
                    np.asarray(best_candidate_theta_dot, dtype=float).reshape(-1),
                    float(step_scale),
                )
            if not new_proposal_found:
                break
        if (
            best_step_theta_dot is None
            or best_step_forecast is None
            or best_step_score is None
            or best_scale is None
            or best_blend_weight is None
            or best_proposal_kind is None
        ):
            raise RuntimeError("no exact-v1 baseline step-scale forecasts were produced")
        best_theta_dot = np.asarray(best_step_theta_dot, dtype=float).reshape(-1)
        best_forecast = dict(best_step_forecast)
        best_score = float(best_step_score)
        best_gain_scale = 1.0
        gain_evaluated: dict[float, tuple[np.ndarray, dict[str, Any], float]] = {}

        def _evaluate_gain(gain_scale: float) -> tuple[np.ndarray, dict[str, Any], float]:
            rounded = round(float(gain_scale), 12)
            cached = gain_evaluated.get(rounded)
            if cached is not None:
                return cached
            gained_theta_dot = float(gain_scale) * np.asarray(best_step_theta_dot, dtype=float).reshape(-1)
            theta_runtime = np.asarray(
                self.current_theta + float(dt) * np.asarray(gained_theta_dot, dtype=float),
                dtype=float,
            ).reshape(-1)
            forecast, _forecast_rollout, score = self._local_projective_forecast_rollout(
                checkpoint_index=checkpoint_index,
                time_stop=float(time_stop),
                executor=self.current_executor,
                layout=self.current_layout,
                theta_runtime_start=theta_runtime,
                theta_dot_step=np.asarray(gained_theta_dot, dtype=float).reshape(-1),
                planning_audit=self._planning_audit,
                scaffold_labels=self._current_scaffold_labels(),
                immediate_gain_ratio=None,
                anchor_summary=(None if baseline is None else baseline.get("summary")),
                anchor_predicted_displacement=(
                    None
                    if baseline is None
                    else self._predicted_displacement(dt=float(dt), baseline=baseline)
                ),
            )
            cached = (
                np.asarray(gained_theta_dot, dtype=float).reshape(-1),
                dict(forecast),
                float(score),
            )
            gain_evaluated[rounded] = cached
            _record_debug_variant(
                stage="gain",
                proposal_kind=(best_proposal_kind or "unknown"),
                blend_weight=(0.0 if best_blend_weight is None else float(best_blend_weight)),
                step_scale=(1.0 if best_scale is None else float(best_scale)),
                gain_scale=float(gain_scale),
                theta_dot_direction=np.asarray(best_step_theta_dot, dtype=float).reshape(-1),
                scaled_theta_dot=np.asarray(gained_theta_dot, dtype=float).reshape(-1),
                forecast=dict(forecast),
                score=float(score),
                current_baseline_norm=best_current_baseline_norm,
                current_drive_norm=best_current_drive_norm,
                lookahead_drive_norm=best_lookahead_drive_norm,
            )
            return cached

        def _consider_gain(gain_scale: float) -> None:
            nonlocal best_theta_dot, best_forecast, best_score, best_gain_scale
            gained_theta_dot, forecast, score = _evaluate_gain(float(gain_scale))
            choose = False
            if float(score) < float(best_score) - 1.0e-12:
                choose = True
            elif (
                abs(float(score) - float(best_score)) <= 1.0e-12
                and float(gain_scale) < float(best_gain_scale) - 1.0e-12
            ):
                choose = True
            if choose:
                best_theta_dot = np.asarray(gained_theta_dot, dtype=float).reshape(-1)
                best_forecast = dict(forecast)
                best_score = float(score)
                best_gain_scale = float(gain_scale)

        for gain_scale in self._exact_forecast_baseline_gain_scales():
            _consider_gain(float(gain_scale))
        best_forecast["baseline_proposal_mode"] = str(proposal_mode)
        best_forecast["baseline_proposal_kind"] = str(best_proposal_kind)
        best_forecast["baseline_current_theta_dot_norm"] = (
            None if best_current_baseline_norm is None else float(best_current_baseline_norm)
        )
        best_forecast["baseline_current_drive_only_norm"] = (
            None if best_current_drive_norm is None else float(best_current_drive_norm)
        )
        best_forecast["baseline_lookahead_drive_only_norm"] = (
            None if best_lookahead_drive_norm is None else float(best_lookahead_drive_norm)
        )
        best_forecast["baseline_include_tangent_secant_proposal"] = bool(
            self._exact_forecast_include_tangent_secant_proposal()
        )
        best_forecast["baseline_tangent_secant_trust_radius"] = float(
            self._exact_forecast_tangent_secant_trust_radius()
        )
        best_forecast["baseline_tangent_secant_displacement_norm"] = (
            None
            if best_tangent_secant_displacement_norm is None
            else float(best_tangent_secant_displacement_norm)
        )
        best_forecast["baseline_tangent_secant_projection_quality"] = (
            None
            if best_tangent_secant_projection_quality is None
            else float(best_tangent_secant_projection_quality)
        )
        best_forecast["baseline_tangent_secant_raw_metric_norm"] = (
            None
            if best_tangent_secant_raw_metric_norm is None
            else float(best_tangent_secant_raw_metric_norm)
        )
        best_forecast["baseline_tangent_secant_metric_norm"] = (
            None
            if best_tangent_secant_metric_norm is None
            else float(best_tangent_secant_metric_norm)
        )
        best_forecast["baseline_tangent_secant_current_energy_bias"] = (
            None
            if best_tangent_secant_current_energy_bias is None
            else float(best_tangent_secant_current_energy_bias)
        )
        best_forecast["baseline_tangent_secant_next_exact_energy_delta"] = (
            None
            if best_tangent_secant_next_exact_energy_delta is None
            else float(best_tangent_secant_next_exact_energy_delta)
        )
        best_forecast["baseline_tangent_secant_signed_energy_lead"] = (
            None
            if best_tangent_secant_signed_energy_lead is None
            else float(best_tangent_secant_signed_energy_lead)
        )
        best_forecast["baseline_tangent_secant_signed_energy_lead_limit"] = (
            None
            if best_tangent_secant_signed_energy_lead_limit is None
            else float(best_tangent_secant_signed_energy_lead_limit)
        )
        best_forecast["baseline_tangent_secant_signed_energy_lead_taper"] = (
            None
            if best_tangent_secant_signed_energy_lead_taper is None
            else float(best_tangent_secant_signed_energy_lead_taper)
        )
        return (
            best_theta_dot,
            float(best_scale),
            float(best_blend_weight),
            float(best_gain_scale),
            best_forecast,
        )

    def _baseline_theta_dot_augmented_for_candidate(
        self,
        *,
        candidate_data: Mapping[str, Any],
        baseline_theta_dot: np.ndarray | Sequence[float],
    ) -> np.ndarray:
        theta_dot_step = np.asarray(baseline_theta_dot, dtype=float).reshape(-1)
        runtime_pos = int(candidate_data["runtime_insert_position"])
        width = int(len(candidate_data["runtime_block_indices"]))
        return np.concatenate(
            [
                theta_dot_step[:runtime_pos],
                np.zeros(width, dtype=float),
                theta_dot_step[runtime_pos:],
            ]
        )

    def _scale_candidate_theta_dot(
        self,
        *,
        candidate_data: Mapping[str, Any],
        baseline_theta_dot: np.ndarray | Sequence[float],
        theta_dot_aug: np.ndarray | Sequence[float],
        step_scale: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        theta_dot_full = np.asarray(theta_dot_aug, dtype=float).reshape(-1)
        theta_dot_baseline_aug = self._baseline_theta_dot_augmented_for_candidate(
            candidate_data=candidate_data,
            baseline_theta_dot=baseline_theta_dot,
        )
        scaled = np.asarray(
            theta_dot_baseline_aug + float(step_scale) * (theta_dot_full - theta_dot_baseline_aug),
            dtype=float,
        ).reshape(-1)
        runtime_pos = int(candidate_data["runtime_insert_position"])
        width = int(len(candidate_data["runtime_block_indices"]))
        eta_dot = np.asarray(scaled[runtime_pos : runtime_pos + width], dtype=float).reshape(-1)
        theta_dot_existing = np.concatenate(
            [
                scaled[:runtime_pos],
                scaled[runtime_pos + width :],
            ]
        )
        return scaled, theta_dot_existing, eta_dot

    def _oracle_for_tier(self, tier_name: str) -> Any:
        if self._oracle_base_config is None or str(self.cfg.mode) not in {"oracle_v1", "off"}:
            raise ValueError("Oracle tier access requested while controller oracle surface is unavailable.")
        tier_key = str(tier_name)
        oracle = self._oracle_instances.get(tier_key)
        if oracle is None:
            from pipelines.exact_bench.noise_oracle_runtime import ExpectationOracle

            oracle = ExpectationOracle(self._oracle_tier_configs[tier_key])
            self._oracle_instances[tier_key] = oracle
        return oracle

    def _close_oracles(self) -> None:
        for oracle in self._oracle_instances.values():
            try:
                oracle.close()
            except Exception:
                pass
        self._oracle_instances.clear()

    def _build_runtime_circuit(
        self,
        *,
        layout: AnsatzParameterLayout,
        theta_runtime: np.ndarray | Sequence[float],
    ) -> Any:
        from pipelines.exact_bench.noise_oracle_runtime import build_runtime_layout_circuit

        return build_runtime_layout_circuit(
            layout,
            theta_runtime,
            int(self._num_qubits),
            reference_state=np.asarray(self.replay_context.psi_ref, dtype=complex).reshape(-1),
        )

    def _oracle_energy_estimate(
        self,
        *,
        checkpoint_ctx: Any,
        cache: OracleCheckpointValueCache,
        raw_group_pool: BackendScheduledRawGroupPool | None,
        tier_name: str,
        observable_family: str,
        candidate_label: str | None,
        position_id: int | None,
        layout: AnsatzParameterLayout,
        theta_runtime: np.ndarray | Sequence[float],
        observable: Any | None = None,
        state_key: str | None = None,
        budget_scale: float = 1.0,
    ) -> tuple[dict[str, Any], bool]:
        value_key = OracleValueKey(
            checkpoint_id=str(checkpoint_ctx.checkpoint_id),
            tier_name=str(tier_name),
            observable_family=str(observable_family),
            candidate_label=(None if candidate_label is None else str(candidate_label)),
            position_id=(None if position_id is None else int(position_id)),
        )

        def _compute() -> dict[str, Any]:
            target_observable = self._oracle_qop if observable is None else observable
            if target_observable is None:
                raise ValueError("Oracle energy estimate requires an observable.")
            if self._oracle_wallclock_hit():
                self._write_progress(
                    stage="wallclock_cap_hit",
                    force=True,
                    status="timeout",
                    checkpoint_index=int(checkpoint_ctx.checkpoint_index),
                    tier_name=str(tier_name),
                    observable_family=str(observable_family),
                    candidate_label=(None if candidate_label is None else str(candidate_label)),
                    position_id=(None if position_id is None else int(position_id)),
                )
                raise TimeoutError("checkpoint controller oracle_v1 wallclock cap reached")
            oracle = self._oracle_for_tier(str(tier_name))
            circuit = self._build_runtime_circuit(layout=layout, theta_runtime=theta_runtime)
            if (
                raw_group_pool is not None
                and self._oracle_base_config is not None
                and str(self._oracle_base_config.noise_mode).strip().lower() in {"backend_scheduled", "runtime"}
            ):
                min_total_shots, min_samples = self._oracle_sampling_targets(
                    tier_name=str(tier_name),
                    budget_scale=float(budget_scale),
                )
                self._write_progress(
                    stage="oracle_energy_estimate_start",
                    force=True,
                    checkpoint_index=int(checkpoint_ctx.checkpoint_index),
                    tier_name=str(tier_name),
                    observable_family=str(observable_family),
                    candidate_label=(None if candidate_label is None else str(candidate_label)),
                    position_id=(None if position_id is None else int(position_id)),
                    min_total_shots=int(min_total_shots),
                    min_samples=int(min_samples),
                    budget_scale=float(budget_scale),
                )
                try:
                    result = raw_group_pool.estimate_observable(
                        oracle=oracle,
                        circuit=circuit,
                        observable=target_observable,
                        observable_family=str(observable_family),
                        candidate_label=(None if candidate_label is None else str(candidate_label)),
                        position_id=(None if position_id is None else int(position_id)),
                        min_total_shots=int(min_total_shots),
                        min_samples=int(min_samples),
                        state_key=(None if state_key is None else str(state_key)),
                    )
                    self._write_progress(
                        stage="oracle_energy_estimate_done",
                        force=True,
                        checkpoint_index=int(checkpoint_ctx.checkpoint_index),
                        tier_name=str(tier_name),
                        observable_family=str(observable_family),
                        candidate_label=(None if candidate_label is None else str(candidate_label)),
                        position_id=(None if position_id is None else int(position_id)),
                        backend="raw_group_pool",
                    )
                    return result
                except Exception as raw_exc:
                    est = oracle.evaluate(circuit, target_observable)
                    backend_info = {
                        "noise_mode": str(oracle.backend_info.noise_mode),
                        "estimator_kind": str(oracle.backend_info.estimator_kind),
                        "backend_name": oracle.backend_info.backend_name,
                        "using_fake_backend": bool(oracle.backend_info.using_fake_backend),
                        "details": {
                            **dict(oracle.backend_info.details),
                            "raw_group_pool_fallback_reason": f"{type(raw_exc).__name__}: {raw_exc}",
                        },
                    }
                    return {
                        "mean": float(est.mean),
                        "stderr": float(est.stderr),
                        "std": float(est.std),
                        "stdev": float(est.stdev),
                        "n_samples": int(est.n_samples),
                        "aggregate": str(est.aggregate),
                        "backend_info": backend_info,
                    }
            self._write_progress(
                stage="oracle_energy_estimate_start",
                force=True,
                checkpoint_index=int(checkpoint_ctx.checkpoint_index),
                tier_name=str(tier_name),
                observable_family=str(observable_family),
                candidate_label=(None if candidate_label is None else str(candidate_label)),
                position_id=(None if position_id is None else int(position_id)),
                budget_scale=float(budget_scale),
            )
            est = oracle.evaluate(circuit, target_observable)
            backend_info = {
                "noise_mode": str(oracle.backend_info.noise_mode),
                "estimator_kind": str(oracle.backend_info.estimator_kind),
                "backend_name": oracle.backend_info.backend_name,
                "using_fake_backend": bool(oracle.backend_info.using_fake_backend),
                "details": dict(oracle.backend_info.details),
            }
            result = {
                "mean": float(est.mean),
                "stderr": float(est.stderr),
                "std": float(est.std),
                "stdev": float(est.stdev),
                "n_samples": int(est.n_samples),
                "aggregate": str(est.aggregate),
                "backend_info": backend_info,
            }
            self._write_progress(
                stage="oracle_energy_estimate_done",
                force=True,
                checkpoint_index=int(checkpoint_ctx.checkpoint_index),
                tier_name=str(tier_name),
                observable_family=str(observable_family),
                candidate_label=(None if candidate_label is None else str(candidate_label)),
                position_id=(None if position_id is None else int(position_id)),
                backend="direct_oracle",
            )
            return result

        return cache.get_or_compute(value_key, compute=_compute)

    def _oracle_measured_baseline_geometry(
        self,
        *,
        checkpoint_ctx: Any,
        cache: ExactCheckpointValueCache,
        geometry_memo: DerivedGeometryMemo,
        raw_group_pool: BackendScheduledRawGroupPool | None,
        h_poly_step: Any,
        tier_name: str,
        budget_scale: float = 1.0,
        layout: AnsatzParameterLayout | None = None,
        theta_runtime: np.ndarray | Sequence[float] | None = None,
        planning_audit: MeasurementCacheAudit | None = None,
    ) -> dict[str, Any]:
        memo_key = DerivedGeometryKey(
            checkpoint_id=str(checkpoint_ctx.checkpoint_id),
            memo_family="oracle_measured_baseline_geometry",
            candidate_label=None,
            position_id=None,
        )

        def _compute() -> dict[str, Any]:
            oracle = self._oracle_for_tier(str(tier_name))
            min_total_shots, min_samples = self._oracle_sampling_targets(
                tier_name=str(tier_name),
                budget_scale=float(budget_scale),
                floor_to_base_config=True,
            )
            layout_obj = self.current_layout if layout is None else layout
            theta_arr = np.asarray(
                self.current_theta if theta_runtime is None else theta_runtime,
                dtype=float,
            ).reshape(-1)
            planning = self._planning_audit if planning_audit is None else planning_audit
            state_key = self._measurement_state_key(
                layout=layout_obj,
                theta_runtime=theta_arr,
            )
            selected_runtime_indices = self._selected_measurement_runtime_indices(
                layout=layout_obj,
            )
            measured = estimate_grouped_raw_mclachlan_geometry(
                oracle=oracle,
                raw_group_pool=raw_group_pool,
                layout=layout_obj,
                theta_runtime=theta_arr,
                psi_ref=np.asarray(self.replay_context.psi_ref, dtype=complex).reshape(-1),
                h_poly=h_poly_step,
                geom_cfg=self._measured_geometry_config(),
                observable_family_prefix="baseline_geometry",
                candidate_label=None,
                position_id=None,
                state_key=str(state_key),
                min_total_shots=int(min_total_shots),
                min_samples=int(min_samples),
                selected_runtime_indices=selected_runtime_indices,
            )
            geometry = dict(measured["geometry"])
            step_objective_value = float(measured.get("step_objective_value", 0.0))
            norm_b_sq = float(geometry["variance"])
            rho_real = float(float(geometry["epsilon_step_sq"]) / max(norm_b_sq, 1.0e-14))
            rho_num = float(max(0.0, rho_real - float(geometry["rho_miss"])))
            summary = BaselineGeometrySummary(
                energy=float(geometry["energy"]),
                variance=float(geometry["variance"]),
                epsilon_proj_sq=float(geometry["epsilon_proj_sq"]),
                epsilon_step_sq=float(geometry["epsilon_step_sq"]),
                rho_miss=float(geometry["rho_miss"]),
                rho_real=float(rho_real),
                rho_num=float(rho_num),
                step_objective_value=float(step_objective_value),
                step_gain_ratio=float(step_objective_value / max(norm_b_sq, 1.0e-14)),
                theta_dot_l2=float(np.linalg.norm(np.asarray(geometry["theta_dot_step"], dtype=float))),
                matrix_rank=int(geometry["matrix_rank"]),
                condition_number=float(geometry["condition_number"]),
                regularization_lambda=self._cfg_float("regularization_lambda"),
                solve_mode="grouped_raw_measured",
                logical_block_count=int(layout_obj.logical_parameter_count),
                runtime_parameter_count=int(layout_obj.runtime_parameter_count),
                planning_summary=dict(planning.summary()),
                exact_cache_summary=dict(cache.summary()),
            )
            return {
                **geometry,
                "rho_real": float(rho_real),
                "rho_num": float(rho_num),
                "summary": summary,
                "backend_info": dict(measured.get("backend_info", {})),
                "observable_estimates": dict(measured.get("observable_estimates", {})),
                "plan_stats": dict(measured.get("plan_stats", {})),
                "raw_group_pool_summary": dict(measured.get("raw_group_pool_summary", {})),
                "step_objective_value": float(measured.get("step_objective_value", 0.0)),
                "state_key": str(measured.get("state_key", state_key)),
            }

        value, _ = geometry_memo.get_or_compute(memo_key, compute=_compute)
        return dict(value)

    def _oracle_measured_candidate_incremental_block(
        self,
        *,
        checkpoint_ctx: Any,
        geometry_memo: DerivedGeometryMemo,
        raw_group_pool: BackendScheduledRawGroupPool | None,
        tier_name: str,
        baseline_measured: Mapping[str, Any],
        record: Mapping[str, Any],
        h_poly_step: Any,
        budget_scale: float = 1.0,
    ) -> dict[str, Any]:
        candidate_identity = str(record.get("candidate_identity", record["candidate_label"]))
        position_id = int(record["position_id"])
        memo_key = DerivedGeometryKey(
            checkpoint_id=str(checkpoint_ctx.checkpoint_id),
            memo_family="oracle_measured_candidate_incremental_block",
            candidate_label=str(candidate_identity),
            position_id=int(position_id),
        )

        def _compute() -> dict[str, Any]:
            oracle = self._oracle_for_tier(str(tier_name))
            min_total_shots, min_samples = self._oracle_sampling_targets(
                tier_name=str(tier_name),
                budget_scale=float(budget_scale),
            )
            candidate_data = dict(record["candidate_data"])
            selected_baseline_runtime_indices = self._selected_measurement_runtime_indices(
                layout=self.current_layout,
            )
            state_key = self._measurement_state_key(
                layout=candidate_data["aug_layout"],
                theta_runtime=np.asarray(candidate_data["theta_aug"], dtype=float).reshape(-1),
            )
            measured = estimate_grouped_raw_mclachlan_incremental_block(
                oracle=oracle,
                raw_group_pool=raw_group_pool,
                baseline_measured=baseline_measured,
                layout=candidate_data["aug_layout"],
                theta_runtime=np.asarray(candidate_data["theta_aug"], dtype=float).reshape(-1),
                psi_ref=np.asarray(self.replay_context.psi_ref, dtype=complex).reshape(-1),
                h_poly=h_poly_step,
                candidate_runtime_indices=tuple(candidate_data["runtime_block_indices"]),
                runtime_insert_position=int(candidate_data["runtime_insert_position"]),
                geom_cfg=self._measured_geometry_config(),
                candidate_regularization_lambda=self._cfg_float("candidate_regularization_lambda"),
                pinv_rcond=self._cfg_float("pinv_rcond"),
                observable_family_prefix="candidate_incremental_block",
                candidate_label=str(candidate_identity),
                position_id=int(position_id),
                state_key=str(state_key),
                min_total_shots=int(min_total_shots),
                min_samples=int(min_samples),
                selected_baseline_runtime_indices=selected_baseline_runtime_indices,
            )
            incremental = dict(measured["incremental_block"])
            return {
                **incremental,
                "backend_info": dict(measured.get("backend_info", {})),
                "observable_estimates": dict(measured.get("observable_estimates", {})),
                "plan_stats": dict(measured.get("plan_stats", {})),
                "raw_group_pool_summary": dict(measured.get("raw_group_pool_summary", {})),
                "state_key": str(measured.get("state_key", state_key)),
                "selected_observable_names": list(measured.get("selected_observable_names", [])),
            }

        value, _ = geometry_memo.get_or_compute(memo_key, compute=_compute)
        return dict(value)

    def _confirm_candidates_oracle_geometry(
        self,
        *,
        checkpoint_ctx: Any,
        cache: ExactCheckpointValueCache,
        geometry_memo: DerivedGeometryMemo,
        confirmed: Sequence[Mapping[str, Any]],
        raw_group_pool: BackendScheduledRawGroupPool | None,
        h_poly_step: Any,
        confirm_limit: int,
        budget_scale: float = 1.0,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str | None]:
        try:
            baseline_measured = self._oracle_measured_baseline_geometry(
                checkpoint_ctx=checkpoint_ctx,
                cache=cache,
                geometry_memo=geometry_memo,
                raw_group_pool=raw_group_pool,
                h_poly_step=h_poly_step,
                tier_name="confirm",
                budget_scale=float(budget_scale),
            )
        except Exception as exc:
            return None, [dict(rec) for rec in confirmed], str(exc)

        if float(baseline_measured["summary"].rho_miss) <= float(self.cfg.miss_threshold):
            skipped: list[dict[str, Any]] = []
            for record in confirmed:
                rec = self._clear_confirm_payload(
                    record,
                    confirm_error="skipped_due_to_measured_baseline_stay",
                    rejection_reason="measured_baseline_below_threshold",
                )
                skipped.append(rec)
            return baseline_measured, skipped, None

        ranked = sorted(
            [dict(rec) for rec in confirmed],
            key=self._confirm_rank_key,
        )
        measured_records: list[dict[str, Any]] = []
        for idx, record in enumerate(ranked):
            rec = dict(record)
            if int(idx) >= int(confirm_limit):
                rec = self._clear_confirm_payload(
                    rec,
                    confirm_error="deferred_by_refresh_pressure",
                    rejection_reason="deferred_by_refresh_pressure",
                )
                measured_records.append(rec)
                continue
            try:
                measured_candidate = self._oracle_measured_candidate_incremental_block(
                    checkpoint_ctx=checkpoint_ctx,
                    geometry_memo=geometry_memo,
                    raw_group_pool=raw_group_pool,
                    tier_name="confirm",
                    baseline_measured=baseline_measured,
                    record=rec,
                    h_poly_step=h_poly_step,
                    budget_scale=float(budget_scale),
                )
                theta_dot_aug = np.asarray(measured_candidate["theta_dot_step"], dtype=float).reshape(-1)
                theta_dot_aug_existing = np.asarray(
                    measured_candidate["theta_dot_aug_existing"],
                    dtype=float,
                ).reshape(-1)
                eta_dot = np.asarray(measured_candidate["eta_dot"], dtype=float).reshape(-1)
                directional_change_l2 = _overlap_l2(theta_dot_aug, self._previous_theta_dot)
                gain_exact = float(measured_candidate["gain_exact"])
                gain_ratio = float(measured_candidate["gain_ratio"])
                confirm_payload = self._confirm_score_payload(
                    baseline=baseline_measured,
                    B=np.asarray(measured_candidate["B"], dtype=float),
                    C=np.asarray(measured_candidate["C"], dtype=float),
                    q=np.asarray(measured_candidate["q"], dtype=float).reshape(-1),
                    w=np.asarray(measured_candidate["w"], dtype=float).reshape(-1),
                    gain_ratio=float(gain_ratio),
                    gain_exact=float(gain_exact),
                    groups_new=float(rec.get("groups_new", 0.0)),
                    directional_change_l2=directional_change_l2,
                )
                rec["gain_exact"] = float(gain_exact)
                rec["gain_ratio"] = float(gain_ratio)
                rec.update(confirm_payload)
                rec["theta_dot_aug"] = theta_dot_aug
                rec["theta_dot_aug_existing"] = theta_dot_aug_existing
                rec["eta_dot"] = eta_dot
                rec["confirm_backend_info"] = dict(measured_candidate.get("backend_info", {}))
                rec["confirm_error"] = None
                rec["candidate_summary"] = replace(
                    rec["candidate_summary"],
                    gain_exact=float(gain_exact),
                    gain_ratio=float(gain_ratio),
                    directional_change_l2=(None if directional_change_l2 is None else float(directional_change_l2)),
                    decision_metric=(
                        "measured_compressed_whitened_confirm_gain_ratio"
                        if str(getattr(self.cfg, "confirm_score_mode", "exact_gain_ratio")) == "compressed_whitened_v1"
                        else "measured_incremental_gain_ratio"
                    ),
                    oracle_estimate_kind=self._oracle_estimate_kind(),
                )
            except Exception as exc:
                return None, [dict(item) for item in confirmed], f"measured_candidate_geometry_error: {exc}"
            measured_records.append(rec)
        return baseline_measured, measured_records, None

    def _confirm_candidates_oracle(
        self,
        *,
        checkpoint_ctx: Any,
        baseline: Mapping[str, Any],
        confirmed: Sequence[Mapping[str, Any]],
        dt: float,
        oracle_cache: OracleCheckpointValueCache,
        raw_group_pool: BackendScheduledRawGroupPool | None,
        oracle_observable: Any | None,
        budget_scale: float = 1.0,
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str | None]:
        stay_theta = np.asarray(
            self.current_theta + float(dt) * np.asarray(baseline["theta_dot_step"], dtype=float),
            dtype=float,
        ).reshape(-1)
        stay_state_key = self._measurement_state_key(
            layout=self.current_layout,
            theta_runtime=stay_theta,
        )
        try:
            stay_estimate, _ = self._oracle_energy_estimate(
                checkpoint_ctx=checkpoint_ctx,
                cache=oracle_cache,
                raw_group_pool=raw_group_pool,
                tier_name="confirm",
                observable_family="stay_step_energy",
                candidate_label=None,
                position_id=None,
                layout=self.current_layout,
                theta_runtime=stay_theta,
                observable=oracle_observable,
                state_key=str(stay_state_key),
                budget_scale=float(budget_scale),
            )
        except Exception as exc:
            return [dict(rec) for rec in confirmed], None, str(exc)

        confirmed_oracle: list[dict[str, Any]] = []
        for record in confirmed:
            rec = dict(record)
            try:
                best_est: dict[str, Any] | None = None
                best_improvement_abs: float | None = None
                best_improvement_ratio: float | None = None
                best_improvement_stderr: float | None = None
                best_adjusted_noisy_improvement: float | None = None
                best_scale: float | None = None
                best_theta_dot_aug: np.ndarray | None = None
                best_theta_dot_existing: np.ndarray | None = None
                best_eta_dot: np.ndarray | None = None
                candidate_data = dict(rec["candidate_data"])
                for step_scale in self._candidate_step_scales():
                    scaled_theta_dot_aug, scaled_theta_dot_existing, scaled_eta_dot = (
                        self._scale_candidate_theta_dot(
                            candidate_data=candidate_data,
                            baseline_theta_dot=baseline["theta_dot_step"],
                            theta_dot_aug=rec["theta_dot_aug"],
                            step_scale=float(step_scale),
                        )
                    )
                    candidate_theta = np.asarray(
                        candidate_data["theta_aug"] + float(dt) * np.asarray(scaled_theta_dot_aug, dtype=float),
                        dtype=float,
                    ).reshape(-1)
                    candidate_state_key = self._measurement_state_key(
                        layout=candidate_data["aug_layout"],
                        theta_runtime=candidate_theta,
                    )
                    est, _ = self._oracle_energy_estimate(
                        checkpoint_ctx=checkpoint_ctx,
                        cache=oracle_cache,
                        raw_group_pool=raw_group_pool,
                        tier_name="confirm",
                        observable_family=f"candidate_step_energy_scale_{self._candidate_scale_tag(float(step_scale))}",
                        candidate_label=str(rec.get("candidate_identity", rec["candidate_label"])),
                        position_id=int(rec["position_id"]),
                        layout=candidate_data["aug_layout"],
                        theta_runtime=candidate_theta,
                        observable=oracle_observable,
                        state_key=str(candidate_state_key),
                        budget_scale=float(budget_scale),
                    )
                    improvement_abs = float(stay_estimate["mean"] - est["mean"])
                    improvement_ratio = float(
                        improvement_abs / max(abs(float(stay_estimate["mean"])), 1e-14)
                    )
                    improvement_stderr = float(
                        np.sqrt(float(stay_estimate["stderr"]) ** 2 + float(est["stderr"]) ** 2)
                    )
                    directional_penalty = (
                        0.0
                        if rec["candidate_summary"].directional_change_l2 is None
                        else float(rec["candidate_summary"].directional_change_l2)
                    )
                    adjusted_noisy_improvement = float(
                        improvement_ratio
                        - float(self.cfg.directional_penalty_weight) * directional_penalty
                        - float(self.cfg.measurement_penalty_weight) * float(rec.get("groups_new", 0.0))
                    )
                    choose = False
                    if best_est is None or best_improvement_abs is None:
                        choose = True
                    elif improvement_abs > float(best_improvement_abs) + 1e-12:
                        choose = True
                    elif abs(improvement_abs - float(best_improvement_abs)) <= 1e-12:
                        if float(step_scale) < float(best_scale):
                            choose = True
                    if choose:
                        best_est = dict(est)
                        best_improvement_abs = float(improvement_abs)
                        best_improvement_ratio = float(improvement_ratio)
                        best_improvement_stderr = float(improvement_stderr)
                        best_adjusted_noisy_improvement = float(adjusted_noisy_improvement)
                        best_scale = float(step_scale)
                        best_theta_dot_aug = np.asarray(scaled_theta_dot_aug, dtype=float).reshape(-1)
                        best_theta_dot_existing = np.asarray(
                            scaled_theta_dot_existing, dtype=float
                        ).reshape(-1)
                        best_eta_dot = np.asarray(scaled_eta_dot, dtype=float).reshape(-1)
                if best_est is None:
                    raise RuntimeError("no oracle candidate step-scale estimates were produced")
                rec["theta_dot_aug"] = np.asarray(best_theta_dot_aug, dtype=float).reshape(-1)
                rec["theta_dot_aug_existing"] = np.asarray(
                    best_theta_dot_existing, dtype=float
                ).reshape(-1)
                rec["eta_dot"] = np.asarray(best_eta_dot, dtype=float).reshape(-1)
                rec["candidate_step_scale"] = float(best_scale)
                rec["predicted_noisy_energy_mean"] = float(best_est["mean"])
                rec["predicted_noisy_energy_stderr"] = float(best_est["stderr"])
                rec["predicted_noisy_improvement_abs"] = float(best_improvement_abs)
                rec["predicted_noisy_improvement_ratio"] = float(best_improvement_ratio)
                rec["predicted_noisy_improvement_stderr"] = float(best_improvement_stderr)
                rec["adjusted_noisy_improvement"] = float(best_adjusted_noisy_improvement)
                rec["confirm_backend_info"] = dict(best_est.get("backend_info", {}))
                rec["confirm_error"] = None
                rec["candidate_summary"] = replace(
                    rec["candidate_summary"],
                    decision_metric="oracle_energy_improvement",
                    oracle_estimate_kind=self._oracle_estimate_kind(),
                    predicted_noisy_energy_mean=float(best_est["mean"]),
                    predicted_noisy_energy_stderr=float(best_est["stderr"]),
                    predicted_noisy_improvement_abs=float(best_improvement_abs),
                    predicted_noisy_improvement_ratio=float(best_improvement_ratio),
                    selected_step_scale=float(best_scale),
                )
            except Exception as exc:
                rec["predicted_noisy_energy_mean"] = None
                rec["predicted_noisy_energy_stderr"] = None
                rec["predicted_noisy_improvement_abs"] = None
                rec["predicted_noisy_improvement_ratio"] = None
                rec["predicted_noisy_improvement_stderr"] = None
                rec["adjusted_noisy_improvement"] = float("-inf")
                rec["candidate_step_scale"] = None
                rec["confirm_backend_info"] = None
                rec["confirm_error"] = str(exc)
            confirmed_oracle.append(rec)
        return confirmed_oracle, stay_estimate, None

    def _select_action_oracle(
        self,
        *,
        baseline: Mapping[str, Any],
        confirmed: Sequence[Mapping[str, Any]],
    ) -> tuple[str, Mapping[str, Any] | None]:
        if float(baseline["summary"].rho_miss) <= float(self.cfg.miss_threshold):
            return "stay", None
        viable = [rec for rec in confirmed if rec.get("predicted_noisy_improvement_abs") is not None]
        if not viable:
            return "stay", None
        ordered = sorted(
            viable,
            key=lambda rec: (
                -float(rec.get("adjusted_noisy_improvement", float("-inf"))),
                float(rec["candidate_summary"].position_jump_penalty),
                float(rec["candidate_summary"].compile_proxy_total),
                float(rec["candidate_summary"].groups_new),
                int(rec["candidate_summary"].candidate_pool_index),
                int(rec["candidate_summary"].position_id),
            ),
        )
        best = ordered[0]
        if float(best.get("predicted_noisy_improvement_ratio", 0.0)) < float(self.cfg.gain_ratio_threshold):
            return "stay", None
        if float(best.get("predicted_noisy_improvement_abs", 0.0)) < float(self.cfg.append_margin_abs):
            return "stay", None
        return "append_candidate", best

    def _oracle_commit_payload(
        self,
        *,
        checkpoint_ctx: Any,
        oracle_cache: OracleCheckpointValueCache,
        raw_group_pool: BackendScheduledRawGroupPool | None,
        baseline: Mapping[str, Any],
        selected: Mapping[str, Any] | None,
        action_kind: str,
        dt: float,
        oracle_observable: Any | None,
        budget_scale: float = 1.0,
    ) -> tuple[dict[str, Any], str | None]:
        stay_theta = np.asarray(
            self.current_theta + float(dt) * np.asarray(baseline["theta_dot_step"], dtype=float),
            dtype=float,
        ).reshape(-1)
        out: dict[str, Any] = {
            "stay_noisy_energy_mean": None,
            "stay_noisy_energy_stderr": None,
            "stay_noisy_backend_info": None,
            "selected_noisy_energy_mean": None,
            "selected_noisy_energy_stderr": None,
            "selected_noisy_backend_info": None,
            "selected_noisy_improvement_abs": None,
            "selected_noisy_improvement_ratio": None,
        }
        degraded_reason: str | None = None
        baseline_summary = baseline.get("summary", None)
        baseline_energy_raw = (
            None
            if baseline_summary is None
            else (
                baseline_summary.get("energy", None)
                if isinstance(baseline_summary, Mapping)
                else getattr(baseline_summary, "energy", None)
            )
        )
        baseline_has_measured_energy = bool(
            baseline.get("backend_info")
            or baseline.get("observable_estimates")
            or baseline.get("raw_group_pool_summary")
        )
        if (
            (str(action_kind) == "stay" or selected is None)
            and baseline_has_measured_energy
            and baseline_energy_raw is not None
        ):
            out["stay_noisy_energy_mean"] = float(baseline_energy_raw)
            out["stay_noisy_energy_stderr"] = None
            out["stay_noisy_backend_info"] = dict(baseline.get("backend_info", {}))
            out["selected_noisy_energy_mean"] = float(baseline_energy_raw)
            out["selected_noisy_energy_stderr"] = None
            out["selected_noisy_backend_info"] = dict(baseline.get("backend_info", {}))
            out["selected_noisy_improvement_abs"] = 0.0
            out["selected_noisy_improvement_ratio"] = 0.0
            return out, None
        stay_state_key = self._measurement_state_key(
            layout=self.current_layout,
            theta_runtime=stay_theta,
        )
        try:
            stay_est, _ = self._oracle_energy_estimate(
                checkpoint_ctx=checkpoint_ctx,
                cache=oracle_cache,
                raw_group_pool=raw_group_pool,
                tier_name="commit",
                observable_family="stay_step_energy",
                candidate_label=None,
                position_id=None,
                layout=self.current_layout,
                theta_runtime=stay_theta,
                observable=oracle_observable,
                state_key=str(stay_state_key),
                budget_scale=float(budget_scale),
            )
            out["stay_noisy_energy_mean"] = float(stay_est["mean"])
            out["stay_noisy_energy_stderr"] = float(stay_est["stderr"])
            out["stay_noisy_backend_info"] = dict(stay_est.get("backend_info", {}))
            if str(action_kind) == "stay" or selected is None:
                out["selected_noisy_energy_mean"] = float(stay_est["mean"])
                out["selected_noisy_energy_stderr"] = float(stay_est["stderr"])
                out["selected_noisy_backend_info"] = dict(stay_est.get("backend_info", {}))
                out["selected_noisy_improvement_abs"] = 0.0
                out["selected_noisy_improvement_ratio"] = 0.0
                return out, None
            selected_theta = np.asarray(
                selected["candidate_data"]["theta_aug"] + float(dt) * np.asarray(selected["theta_dot_aug"], dtype=float),
                dtype=float,
            ).reshape(-1)
            selected_state_key = self._measurement_state_key(
                layout=selected["candidate_data"]["aug_layout"],
                theta_runtime=selected_theta,
            )
            selected_est, _ = self._oracle_energy_estimate(
                checkpoint_ctx=checkpoint_ctx,
                cache=oracle_cache,
                raw_group_pool=raw_group_pool,
                tier_name="commit",
                observable_family="candidate_step_energy",
                candidate_label=str(selected.get("candidate_identity", selected["candidate_label"])),
                position_id=int(selected["position_id"]),
                layout=selected["candidate_data"]["aug_layout"],
                theta_runtime=selected_theta,
                observable=oracle_observable,
                state_key=str(selected_state_key),
                budget_scale=float(budget_scale),
            )
            improvement_abs = float(stay_est["mean"] - selected_est["mean"])
            out["selected_noisy_energy_mean"] = float(selected_est["mean"])
            out["selected_noisy_energy_stderr"] = float(selected_est["stderr"])
            out["selected_noisy_backend_info"] = dict(selected_est.get("backend_info", {}))
            out["selected_noisy_improvement_abs"] = float(improvement_abs)
            out["selected_noisy_improvement_ratio"] = float(
                improvement_abs / max(abs(float(stay_est["mean"])), 1e-14)
            )
        except Exception as exc:
            degraded_reason = str(exc)
        return out, degraded_reason

    def _oracle_commit_override_reason(
        self,
        *,
        motion: MotionSchedulerTelemetry,
        selected: Mapping[str, Any] | None,
        action_kind: str,
        oracle_commit_payload: Mapping[str, Any],
        predicted_displacement: float,
        runtime_parameter_count_before: int,
    ) -> str | None:
        if str(action_kind) != "append_candidate" or selected is None:
            return None
        improvement_abs_raw = oracle_commit_payload.get("selected_noisy_improvement_abs", None)
        if improvement_abs_raw is None:
            return None
        improvement_ratio_raw = oracle_commit_payload.get("selected_noisy_improvement_ratio", None)
        try:
            improvement_abs = float(improvement_abs_raw)
        except Exception:
            return None
        try:
            improvement_ratio = (
                None if improvement_ratio_raw is None else float(improvement_ratio_raw)
            )
        except Exception:
            improvement_ratio = None
        if not np.isfinite(improvement_abs):
            return None
        if improvement_abs >= -float(self.cfg.append_margin_abs):
            regime = str(motion.regime)
            if (
                regime == "kink"
                and int(runtime_parameter_count_before) <= 2
                and float(predicted_displacement) >= 0.08
                and improvement_ratio is not None
                and np.isfinite(improvement_ratio)
                and float(improvement_ratio) < 0.20
            ):
                return "kink_weak_margin_first_append"
            if (
                regime == "kink"
                and int(runtime_parameter_count_before) >= 3
                and float(predicted_displacement) >= 0.5
                and improvement_ratio is not None
                and np.isfinite(improvement_ratio)
                and float(improvement_ratio) < 0.30
            ):
                return "kink_large_displacement_commit"
            return None
        regime = str(motion.regime)
        if regime == "bootstrap":
            return "bootstrap_negative_noisy_commit"
        # Under driven kink motion, a negative measured commit signal is more trustworthy
        # than the exact geometry preview for deciding whether to spend a new append.
        if regime == "kink":
            return "kink_negative_noisy_commit"
        return None

    def _baseline_geometry(
        self,
        checkpoint_ctx: Any,
        cache: ExactCheckpointValueCache,
        geometry_memo: DerivedGeometryMemo,
        step_hamiltonian: StepHamiltonianArtifacts | None = None,
    ) -> dict[str, Any]:
        resolved_step = (
            self._step_hamiltonian_artifacts(
                self._projection_sample_time(
                    float(getattr(checkpoint_ctx, "time_start", 0.0)),
                    getattr(checkpoint_ctx, "time_stop", None),
                )
            )
            if step_hamiltonian is None
            else step_hamiltonian
        )
        value, _ = geometry_memo.get_or_compute(
            DerivedGeometryKey(
                checkpoint_id=str(checkpoint_ctx.checkpoint_id),
                memo_family="baseline_geometry",
                candidate_label=None,
                position_id=None,
            ),
            compute=lambda: self._compute_baseline_geometry(
                checkpoint_ctx=checkpoint_ctx,
                cache=cache,
                step_hamiltonian=resolved_step,
            ),
        )
        return dict(value)

    def _compute_baseline_geometry(
        self,
        *,
        checkpoint_ctx: Any,
        cache: ExactCheckpointValueCache,
        step_hamiltonian: StepHamiltonianArtifacts,
    ) -> dict[str, Any]:
        return self._compute_baseline_geometry_for_runtime_state(
            checkpoint_ctx=checkpoint_ctx,
            cache=cache,
            executor=self.current_executor,
            layout=self.current_layout,
            theta_runtime=self.current_theta,
            planning_audit=self._planning_audit,
            step_hamiltonian=step_hamiltonian,
        )

    def _compute_baseline_geometry_for_runtime_state(
        self,
        *,
        checkpoint_ctx: Any,
        cache: ExactCheckpointValueCache,
        executor: CompiledAnsatzExecutor,
        layout: AnsatzParameterLayout,
        theta_runtime: np.ndarray | Sequence[float],
        planning_audit: MeasurementCacheAudit,
        step_hamiltonian: StepHamiltonianArtifacts,
    ) -> dict[str, Any]:
        theta_arr = np.asarray(theta_runtime, dtype=float).reshape(-1)
        runtime_indices = tuple(range(int(layout.runtime_parameter_count)))
        psi, raw_tangents = cache.get_or_compute(
            GeometryValueKey(
                checkpoint_id=str(checkpoint_ctx.checkpoint_id),
                observable_family="baseline_runtime_tangents",
                candidate_label=None,
                position_id=None,
                runtime_indices=runtime_indices,
                group_key=None,
                grouping_mode=str(self.cfg.grouping_mode),
            ),
            tier_name="scout",
            compute=lambda: executor.prepare_state_with_runtime_tangents(
                theta_arr,
                self.replay_context.psi_ref,
                runtime_indices=runtime_indices,
            ),
        )[0]
        energy, hpsi, variance = cache.get_or_compute(
            GeometryValueKey(
                checkpoint_id=str(checkpoint_ctx.checkpoint_id),
                observable_family="baseline_h_apply",
                candidate_label=None,
                position_id=None,
                runtime_indices=tuple(),
                group_key=None,
                grouping_mode=str(self.cfg.grouping_mode),
            ),
            tier_name="scout",
            compute=lambda: self._energy_hpsi_variance(psi, compiled_h=step_hamiltonian.compiled_h),
        )[0]

        psi_vec = np.asarray(psi, dtype=complex).reshape(-1)
        tangents_matrix: np.ndarray
        if int(layout.runtime_parameter_count) <= 0:
            tangents_matrix = np.zeros((psi_vec.size, 0), dtype=complex)
        else:
            centered_cols: list[np.ndarray] = []
            for runtime_idx in runtime_indices:
                tangent = np.asarray(raw_tangents[int(runtime_idx)], dtype=complex).reshape(-1)
                centered = tangent - complex(np.vdot(psi_vec, tangent)) * psi_vec
                centered_cols.append(np.asarray(centered, dtype=complex))
            tangents_matrix = np.column_stack(centered_cols) if centered_cols else np.zeros((psi_vec.size, 0), dtype=complex)

        b_bar = -1.0j * (np.asarray(hpsi, dtype=complex).reshape(-1) - float(energy) * psi_vec)
        norm_b_sq = float(max(0.0, np.real(np.vdot(b_bar, b_bar))))
        G = np.asarray(np.real(tangents_matrix.conj().T @ tangents_matrix), dtype=float)
        f = np.asarray(np.real(tangents_matrix.conj().T @ b_bar), dtype=float).reshape(-1)

        def _geometry_payload(G_now: np.ndarray, f_now: np.ndarray) -> dict[str, Any]:
            G_eval = np.asarray(G_now, dtype=float)
            f_eval = np.asarray(f_now, dtype=float).reshape(-1)
            G_pinv = (
                np.linalg.pinv(G_eval, rcond=self._cfg_float("pinv_rcond"))
                if G_eval.size
                else np.zeros((0, 0), dtype=float)
            )
            K = np.asarray(
                G_eval
                + self._cfg_float("regularization_lambda") * np.eye(int(G_eval.shape[0])),
                dtype=float,
            )
            K_pinv = (
                np.linalg.pinv(K, rcond=self._cfg_float("pinv_rcond"))
                if K.size
                else np.zeros((0, 0), dtype=float)
            )
            theta_dot_proj = (
                np.asarray(G_pinv @ f_eval, dtype=float).reshape(-1)
                if G_eval.size
                else np.zeros(0, dtype=float)
            )
            theta_dot_step = (
                np.asarray(K_pinv @ f_eval, dtype=float).reshape(-1)
                if K.size
                else np.zeros(0, dtype=float)
            )
            step_objective_value = (
                float(max(0.0, norm_b_sq - float(np.real(np.vdot(tangents_matrix @ theta_dot_step - b_bar, tangents_matrix @ theta_dot_step - b_bar)))))
                if theta_dot_step.size
                else 0.0
            )
            epsilon_proj_sq = (
                float(max(0.0, norm_b_sq - float(f_eval @ theta_dot_proj)))
                if f_eval.size
                else float(norm_b_sq)
            )
            residual_step = np.asarray(
                tangents_matrix @ theta_dot_step - b_bar, dtype=complex
            ).reshape(-1)
            epsilon_step_sq = float(
                max(0.0, np.real(np.vdot(residual_step, residual_step)))
            )
            rho_miss = float(epsilon_proj_sq / max(norm_b_sq, 1e-14))
            rho_real = float(epsilon_step_sq / max(norm_b_sq, 1e-14))
            rho_num = float(max(0.0, rho_real - rho_miss))
            rank = (
                int(np.linalg.matrix_rank(K, tol=self._cfg_float("pinv_rcond")))
                if K.size
                else 0
            )
            cond = float(np.linalg.cond(K)) if K.size else 1.0
            theta_dot_l2 = float(np.linalg.norm(theta_dot_step))
            return {
                "G": G_eval,
                "f": f_eval,
                "K": K,
                "K_pinv": K_pinv,
                "theta_dot_proj": theta_dot_proj,
                "theta_dot_step": theta_dot_step,
                "residual_step": residual_step,
                "rho_miss": float(rho_miss),
                "rho_real": float(rho_real),
                "rho_num": float(rho_num),
                "summary": BaselineGeometrySummary(
                    energy=float(energy),
                    variance=float(variance),
                    epsilon_proj_sq=float(epsilon_proj_sq),
                    epsilon_step_sq=float(epsilon_step_sq),
                    rho_miss=float(rho_miss),
                    rho_real=float(rho_real),
                    rho_num=float(rho_num),
                    step_objective_value=float(step_objective_value),
                    step_gain_ratio=float(step_objective_value / max(norm_b_sq, 1.0e-14)),
                    theta_dot_l2=float(theta_dot_l2),
                    matrix_rank=int(rank),
                    condition_number=float(cond),
                    regularization_lambda=self._cfg_float("regularization_lambda"),
                    solve_mode="pinv_reg",
                    logical_block_count=int(layout.logical_parameter_count),
                    runtime_parameter_count=int(layout.runtime_parameter_count),
                    planning_summary=dict(planning_audit.summary()),
                    exact_cache_summary=dict(cache.summary()),
                ),
            }

        def _geometry_payload_is_finite(payload: Mapping[str, Any]) -> bool:
            for key in ("G", "f", "K", "K_pinv", "theta_dot_proj", "theta_dot_step"):
                arr = np.asarray(payload.get(key), dtype=float)
                if arr.size and (not np.all(np.isfinite(arr))):
                    return False
            summary = payload["summary"]
            return bool(
                np.isfinite(float(summary.epsilon_proj_sq))
                and np.isfinite(float(summary.epsilon_step_sq))
                and np.isfinite(float(summary.rho_miss))
                and np.isfinite(float(summary.rho_real))
                and np.isfinite(float(summary.rho_num))
                and np.isfinite(float(summary.theta_dot_l2))
                and np.isfinite(float(summary.condition_number))
            )

        baseline_payload = {
            "psi": psi_vec,
            "energy": float(energy),
            "variance": float(variance),
            "Hpsi": np.asarray(hpsi, dtype=complex).reshape(-1),
            "b_bar": b_bar,
            "norm_b_sq": float(norm_b_sq),
            "T": tangents_matrix,
            **_geometry_payload(G, f),
            "analytic_noise_applied": bool(self._analytic_noise_enabled()),
            "analytic_noise_model": str(self._analytic_noise_model),
            "analytic_noise_degraded_reason": None,
        }
        if not self._analytic_noise_enabled() or not G.size:
            return baseline_payload

        summary = baseline_payload["summary"]
        if self._analytic_noise_model == "hybrid_qpu_proxy_v1":
            scale = float(self._hybrid_noise_scale(summary))
            group_burden = float(self._planning_group_burden(summary))
            G_noise = self._add_symmetric_gaussian_noise(np.zeros_like(G, dtype=float))
            f_noise = self._add_vector_gaussian_noise(np.zeros_like(f, dtype=float))
            noisy_G = np.asarray(G + scale * G_noise, dtype=float)
            noisy_f = np.asarray(
                f
                + scale * f_noise
                + self._hybrid_observable_bias_vector(
                    psi=psi_vec,
                    energy=float(energy),
                    f=f,
                ),
                dtype=float,
            )
            if bool(self._analytic_noise_force_psd):
                noisy_G = self._force_psd_metric(noisy_G)
            noisy_payload = _geometry_payload(noisy_G, noisy_f)
            baseline_payload["analytic_noise_features"] = {
                "shots_eff": float(
                    max(
                        1.0,
                        (
                            float(self._analytic_noise_nominal_shots)
                            * float(self._analytic_noise_nominal_repeats)
                            / max(group_burden, 1.0)
                        ),
                    )
                ),
                "group_burden": float(group_burden),
                "runtime_parameter_count": int(summary.runtime_parameter_count),
                "logical_block_count": int(summary.logical_block_count),
                "resolved_scale": float(scale),
            }
        else:
            noisy_payload = _geometry_payload(
                self._add_symmetric_gaussian_noise(G),
                self._add_vector_gaussian_noise(f),
            )
        if not _geometry_payload_is_finite(noisy_payload):
            baseline_payload["analytic_noise_degraded_reason"] = (
                "analytic_noise_nonfinite_metric"
            )
            return baseline_payload

        baseline_payload.update(noisy_payload)
        return baseline_payload

    def _integrator_policy(self) -> str:
        return str(getattr(self.cfg, "integrator_policy", "euler")).strip().lower()

    def _integrator_time_fraction(self, time_start: float) -> float:
        if int(getattr(self.times, "size", 0)) <= 1:
            return 1.0
        t0 = float(self.times[0])
        t1 = float(self.times[-1])
        span = float(t1 - t0)
        if abs(span) <= 1.0e-14:
            return 1.0
        frac = (float(time_start) - t0) / span
        return float(min(1.0, max(0.0, frac)))

    def _integrator_euler_time_gate_pass(self, time_start: float) -> tuple[bool, float, float]:
        fraction = self._integrator_time_fraction(float(time_start))
        min_fraction = float(getattr(self.cfg, "integrator_euler_min_time_fraction", 0.0))
        return bool(float(fraction) >= float(min_fraction)), float(fraction), float(min_fraction)

    def _integrator_euler_observable_gate(self) -> dict[str, Any]:
        """Return whether recent controller observables are calm enough for Euler.

        This gate deliberately uses controller trajectory observables only. Exact
        benchmark rows may exist in the same trajectory for reporting, but Euler
        eligibility must be based on whether the prepared-controller observables
        have stopped changing.
        """
        window = int(max(1, int(getattr(self.cfg, "integrator_euler_observable_window", 16))))
        physical_rows = physical_trajectory_rows(self._trajectory, fallback_to_raw=False)
        rows = physical_rows[-window:]

        def _finite_scalar(row: Mapping[str, Any], *keys: str) -> float | None:
            for key in keys:
                raw_value = row.get(key, None)
                if raw_value is None:
                    continue
                try:
                    value = float(raw_value)
                except Exception:
                    continue
                if np.isfinite(value):
                    return float(value)
            return None

        def _span(values: Sequence[float]) -> float | None:
            vals = [float(value) for value in values if np.isfinite(float(value))]
            if len(vals) < 2:
                return None
            return float(max(vals) - min(vals))

        def _scalar_span(*keys: str) -> float | None:
            values: list[float] = []
            for row in rows:
                value = _finite_scalar(row, *keys)
                if value is not None:
                    values.append(float(value))
            return _span(values)

        def _site_span_max() -> float | None:
            vectors: list[list[float]] = []
            for row in rows:
                raw_values = row.get("site_occupations", None)
                if isinstance(raw_values, (str, bytes)) or not isinstance(raw_values, Sequence):
                    continue
                values: list[float] = []
                valid = True
                for raw_value in raw_values:
                    try:
                        value = float(raw_value)
                    except Exception:
                        valid = False
                        break
                    if not np.isfinite(value):
                        valid = False
                        break
                    values.append(float(value))
                if valid and values:
                    vectors.append(values)
            if len(vectors) < 2:
                return None
            width = min(len(values) for values in vectors)
            if width <= 0:
                return None
            max_span = 0.0
            for site_index in range(width):
                site_values = [float(values[site_index]) for values in vectors]
                max_span = max(max_span, float(max(site_values) - min(site_values)))
            return float(max_span)

        site_span = _site_span_max()
        primary_span = _scalar_span("primary_density", "staggered")
        energy_span = _scalar_span("energy_total", "energy_total_controller")
        checks = (
            ("integrator_euler_site_span_max", site_span),
            ("integrator_euler_primary_density_span_max", primary_span),
            ("integrator_euler_energy_span_max", energy_span),
        )
        pass_gate = True
        any_enabled = False
        for field_name, metric_value in checks:
            threshold = getattr(self.cfg, field_name, None)
            if threshold is None:
                continue
            any_enabled = True
            if metric_value is None or float(metric_value) > float(threshold):
                pass_gate = False
        return {
            "integrator_euler_observable_gate_pass": bool(pass_gate if any_enabled else True),
            "integrator_euler_site_span": site_span,
            "integrator_euler_primary_density_span": primary_span,
            "integrator_euler_energy_span": energy_span,
        }

    def _integrator_vector_diagnostics(
        self,
        theta_dot: np.ndarray | Sequence[float],
    ) -> tuple[float, float]:
        current = np.asarray(theta_dot, dtype=float).reshape(-1)
        previous = self._previous_theta_dot
        if previous is None:
            return 1.0, 0.0
        prev = np.asarray(previous, dtype=float).reshape(-1)
        size = max(int(current.size), int(prev.size))
        if size <= 0:
            return 1.0, 0.0
        current_pad = np.zeros(size, dtype=float)
        prev_pad = np.zeros(size, dtype=float)
        current_pad[: int(current.size)] = current
        prev_pad[: int(prev.size)] = prev
        current_norm = float(np.linalg.norm(current_pad))
        prev_norm = float(np.linalg.norm(prev_pad))
        if current_norm <= 1.0e-14 or prev_norm <= 1.0e-14:
            columnarity = 1.0
        else:
            columnarity = float(np.dot(current_pad, prev_pad) / (current_norm * prev_norm))
        curvature = float(
            np.linalg.norm(current_pad - prev_pad) / max(current_norm, prev_norm, 1.0e-14)
        )
        return float(columnarity), float(curvature)

    def _theta_fs_distance(
        self,
        delta_theta: np.ndarray | Sequence[float],
        *,
        baseline: Mapping[str, Any],
    ) -> float:
        delta = np.asarray(delta_theta, dtype=float).reshape(-1)
        if delta.size <= 0:
            return 0.0
        G = np.asarray(baseline.get("G", np.zeros((0, 0), dtype=float)), dtype=float)
        if G.shape == (int(delta.size), int(delta.size)):
            quad = float(delta @ G @ delta)
        else:
            quad = float(delta @ delta)
        return float(np.sqrt(max(quad, 0.0)))

    def _integrator_stage_baseline(
        self,
        *,
        checkpoint_index: int,
        stage_time: float,
        executor: CompiledAnsatzExecutor,
        layout: AnsatzParameterLayout,
        theta_runtime: np.ndarray | Sequence[float],
        planning_audit: MeasurementCacheAudit,
        scaffold_labels: Sequence[str],
    ) -> dict[str, Any]:
        theta_arr = np.asarray(theta_runtime, dtype=float).reshape(-1)
        psi_stage = executor.prepare_state(theta_arr, self.replay_context.psi_ref)
        stage_ctx = make_checkpoint_context(
            checkpoint_index=int(checkpoint_index),
            time_start=float(stage_time),
            time_stop=float(stage_time),
            scaffold_labels=[str(label) for label in scaffold_labels],
            theta=theta_arr,
            psi=psi_stage,
            logical_count=int(layout.logical_parameter_count),
            runtime_count=int(layout.runtime_parameter_count),
            resolved_family=str(self.replay_context.family_info.get("resolved", "unknown")),
            grouping_mode=str(self.cfg.grouping_mode),
            structure_locked=False,
        )
        stage_cache = ExactCheckpointValueCache(
            checkpoint_id=str(stage_ctx.checkpoint_id),
            grouping_mode=str(self.cfg.grouping_mode),
        )
        step_hamiltonian = self._step_hamiltonian_artifacts(float(stage_time))
        return self._compute_baseline_geometry_for_runtime_state(
            checkpoint_ctx=stage_ctx,
            cache=stage_cache,
            executor=executor,
            layout=layout,
            theta_runtime=theta_arr,
            planning_audit=planning_audit,
            step_hamiltonian=step_hamiltonian,
        )

    def _rk4_integrate_theta_one_step(
        self,
        *,
        checkpoint_index: int,
        time_start: float,
        time_stop: float,
        executor: CompiledAnsatzExecutor,
        layout: AnsatzParameterLayout,
        theta_runtime: np.ndarray | Sequence[float],
        planning_audit: MeasurementCacheAudit,
        scaffold_labels: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        theta0 = np.asarray(theta_runtime, dtype=float).reshape(-1)
        dt = float(time_stop) - float(time_start)
        if theta0.size <= 0 or abs(dt) <= 0.0:
            return np.asarray(theta0, dtype=float), np.zeros_like(theta0, dtype=float)
        t0 = float(time_start)
        tm = float(time_start) + 0.5 * float(dt)
        t1 = float(time_stop)
        k1 = np.asarray(
            self._integrator_stage_baseline(
                checkpoint_index=int(checkpoint_index),
                stage_time=t0,
                executor=executor,
                layout=layout,
                theta_runtime=theta0,
                planning_audit=planning_audit,
                scaffold_labels=scaffold_labels,
            )["theta_dot_step"],
            dtype=float,
        ).reshape(-1)
        k2 = np.asarray(
            self._integrator_stage_baseline(
                checkpoint_index=int(checkpoint_index),
                stage_time=tm,
                executor=executor,
                layout=layout,
                theta_runtime=theta0 + 0.5 * float(dt) * k1,
                planning_audit=planning_audit,
                scaffold_labels=scaffold_labels,
            )["theta_dot_step"],
            dtype=float,
        ).reshape(-1)
        k3 = np.asarray(
            self._integrator_stage_baseline(
                checkpoint_index=int(checkpoint_index),
                stage_time=tm,
                executor=executor,
                layout=layout,
                theta_runtime=theta0 + 0.5 * float(dt) * k2,
                planning_audit=planning_audit,
                scaffold_labels=scaffold_labels,
            )["theta_dot_step"],
            dtype=float,
        ).reshape(-1)
        k4 = np.asarray(
            self._integrator_stage_baseline(
                checkpoint_index=int(checkpoint_index),
                stage_time=t1,
                executor=executor,
                layout=layout,
                theta_runtime=theta0 + float(dt) * k3,
                planning_audit=planning_audit,
                scaffold_labels=scaffold_labels,
            )["theta_dot_step"],
            dtype=float,
        ).reshape(-1)
        theta_dot_rk4 = np.asarray((k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0, dtype=float).reshape(-1)
        theta_next = np.asarray(theta0 + float(dt) * theta_dot_rk4, dtype=float).reshape(-1)
        return theta_next, theta_dot_rk4

    def _integrate_theta_one_step(
        self,
        *,
        checkpoint_index: int,
        time_start: float,
        time_stop: float | None,
        executor: CompiledAnsatzExecutor,
        layout: AnsatzParameterLayout,
        theta_runtime: np.ndarray | Sequence[float],
        baseline: Mapping[str, Any],
        planning_audit: MeasurementCacheAudit,
        scaffold_labels: Sequence[str],
        forced_policy: str | None = None,
        euler_theta_dot: np.ndarray | Sequence[float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        theta0 = np.asarray(theta_runtime, dtype=float).reshape(-1)
        dt = 0.0 if time_stop is None else float(float(time_stop) - float(time_start))
        theta_dot_euler = np.asarray(
            baseline.get("theta_dot_step") if euler_theta_dot is None else euler_theta_dot,
            dtype=float,
        ).reshape(-1)
        policy = str(self._integrator_policy() if forced_policy is None else forced_policy).strip().lower()
        summary = baseline.get("summary", None)
        condition_number = float(getattr(summary, "condition_number", 1.0))
        rho_miss = float(getattr(summary, "rho_miss", 0.0))
        columnarity, curvature = self._integrator_vector_diagnostics(theta_dot_euler)
        euler_time_gate_pass, time_fraction, euler_min_time_fraction = (
            self._integrator_euler_time_gate_pass(float(time_start))
        )
        observable_gate = self._integrator_euler_observable_gate()
        condition_pass = bool(condition_number <= float(getattr(self.cfg, "integrator_condition_max", 1.0e10)))
        rho_miss_pass = bool(rho_miss <= float(getattr(self.cfg, "miss_threshold", 0.05)))
        theta_next_euler = np.asarray(theta0 + float(dt) * theta_dot_euler, dtype=float).reshape(-1)
        diagnostics: dict[str, Any] = {
            "integrator_policy": str(policy),
            "integrator_used": "euler",
            "integrator_columnarity": float(columnarity),
            "integrator_curvature": float(curvature),
            "integrator_euler_fs_error": None,
            "integrator_auto_policy_schema": (
                AUTO_EULER_RK4_POLICY_SCHEMA if str(policy) == "auto_euler_rk4" else None
            ),
            "integrator_auto_admit_euler": None,
            "integrator_euler_blockers": [],
            "integrator_condition_number": float(condition_number),
            "integrator_condition_pass": bool(condition_pass),
            "integrator_rho_miss_pass": bool(rho_miss_pass),
            "integrator_time_fraction": float(time_fraction),
            "integrator_euler_min_time_fraction": float(euler_min_time_fraction),
            "integrator_euler_time_gate_pass": bool(euler_time_gate_pass),
            **observable_gate,
            "integrator_error": None,
        }
        if time_stop is None or abs(float(dt)) <= 0.0:
            diagnostics["integrator_used"] = "none"
            return theta_next_euler, theta_dot_euler, diagnostics
        if policy == "euler":
            return theta_next_euler, theta_dot_euler, diagnostics
        try:
            theta_next_rk4, theta_dot_rk4 = self._rk4_integrate_theta_one_step(
                checkpoint_index=int(checkpoint_index),
                time_start=float(time_start),
                time_stop=float(time_stop),
                executor=executor,
                layout=layout,
                theta_runtime=theta0,
                planning_audit=planning_audit,
                scaffold_labels=scaffold_labels,
            )
            euler_fs_error = self._theta_fs_distance(
                theta_next_rk4 - theta_next_euler,
                baseline=baseline,
            )
            diagnostics["integrator_euler_fs_error"] = float(euler_fs_error)
        except Exception as exc:
            diagnostics["integrator_error"] = f"{type(exc).__name__}: {exc}"
            raise
        if policy == "rk4":
            diagnostics["integrator_used"] = "rk4"
            return theta_next_rk4, theta_dot_rk4, diagnostics
        geometry_gate_pass = bool(
            float(columnarity) >= float(getattr(self.cfg, "integrator_columnarity_threshold", 0.80))
            and float(curvature) <= float(getattr(self.cfg, "integrator_curvature_threshold", 0.10))
        )
        euler_error_pass = bool(
            float(diagnostics["integrator_euler_fs_error"])
            <= float(getattr(self.cfg, "integrator_euler_fs_error_threshold", 1.0e-3))
        )
        diagnostics.update(
            _auto_euler_blocker_diagnostics(
                geometry_gate_pass=bool(geometry_gate_pass),
                euler_error_pass=bool(euler_error_pass),
                condition_pass=bool(condition_pass),
                rho_miss_pass=bool(rho_miss_pass),
                euler_time_gate_pass=bool(euler_time_gate_pass),
                observable_gate_pass=bool(observable_gate["integrator_euler_observable_gate_pass"]),
            )
        )
        use_euler = bool(diagnostics["integrator_auto_admit_euler"])
        if use_euler:
            diagnostics["integrator_used"] = "euler"
            return theta_next_euler, theta_dot_euler, diagnostics
        diagnostics["integrator_used"] = "rk4"
        return theta_next_rk4, theta_dot_rk4, diagnostics

    def _strict_qpu_hh_measured_integrator_stage_baseline(
        self,
        *,
        checkpoint_index: int,
        stage_time: float,
        layout: AnsatzParameterLayout,
        theta_runtime: np.ndarray | Sequence[float],
        planning_audit: MeasurementCacheAudit,
        scaffold_labels: Sequence[str],
        tier_name: str,
        budget_scale: float,
    ) -> dict[str, Any]:
        theta_arr = np.asarray(theta_runtime, dtype=float).reshape(-1)
        labels = [str(label) for label in scaffold_labels]
        stage_ctx = make_measurement_checkpoint_context(
            checkpoint_index=int(checkpoint_index),
            time_start=float(stage_time),
            time_stop=float(stage_time),
            scaffold_labels=labels,
            theta=theta_arr,
            logical_count=int(layout.logical_parameter_count),
            runtime_count=int(layout.runtime_parameter_count),
            resolved_family=str(self.replay_context.family_info.get("resolved", "unknown")),
            grouping_mode=str(self.cfg.grouping_mode),
            structure_locked=False,
        )
        stage_cache = ExactCheckpointValueCache(
            checkpoint_id=str(stage_ctx.checkpoint_id),
            grouping_mode=str(self.cfg.grouping_mode),
        )
        stage_memo = DerivedGeometryMemo(checkpoint_id=str(stage_ctx.checkpoint_id))
        stage_raw_group_pool = (
            BackendScheduledRawGroupPool(checkpoint_id=str(stage_ctx.checkpoint_id))
            if self._oracle_base_config is not None
            and bool(controller_oracle_supports_raw_group_sampling(self._oracle_base_config))
            else None
        )
        step_hamiltonian = self._step_hamiltonian_artifacts(float(stage_time))
        return self._oracle_measured_baseline_geometry(
            checkpoint_ctx=stage_ctx,
            cache=stage_cache,
            geometry_memo=stage_memo,
            raw_group_pool=stage_raw_group_pool,
            h_poly_step=step_hamiltonian.h_poly,
            tier_name=str(tier_name),
            budget_scale=float(budget_scale),
            layout=layout,
            theta_runtime=theta_arr,
            planning_audit=planning_audit,
        )

    def _strict_qpu_hh_integrate_theta_one_step(
        self,
        *,
        checkpoint_index: int,
        time_start: float,
        time_stop: float | None,
        layout: AnsatzParameterLayout,
        theta_runtime: np.ndarray | Sequence[float],
        baseline: Mapping[str, Any],
        planning_audit: MeasurementCacheAudit,
        scaffold_labels: Sequence[str],
        tier_name: str,
        budget_scale: float,
        euler_theta_dot: np.ndarray | Sequence[float] | None = None,
        forced_policy: str | None = None,
        forced_policy_reason: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        theta0 = np.asarray(theta_runtime, dtype=float).reshape(-1)
        dt = 0.0 if time_stop is None else float(float(time_stop) - float(time_start))
        theta_dot_euler = np.asarray(
            baseline.get("theta_dot_step") if euler_theta_dot is None else euler_theta_dot,
            dtype=float,
        ).reshape(-1)
        policy = str(
            self._integrator_policy() if forced_policy is None else forced_policy
        ).strip().lower()
        summary = baseline.get("summary", None)
        condition_number = float(getattr(summary, "condition_number", 1.0))
        rho_miss = float(getattr(summary, "rho_miss", 0.0))
        columnarity, curvature = self._integrator_vector_diagnostics(theta_dot_euler)
        euler_time_gate_pass, time_fraction, euler_min_time_fraction = (
            self._integrator_euler_time_gate_pass(float(time_start))
        )
        observable_gate = self._integrator_euler_observable_gate()
        condition_pass = bool(
            condition_number <= float(getattr(self.cfg, "integrator_condition_max", 1.0e10))
        )
        rho_miss_pass = bool(rho_miss <= float(getattr(self.cfg, "miss_threshold", 0.05)))
        theta_next_euler = np.asarray(theta0 + float(dt) * theta_dot_euler, dtype=float).reshape(-1)
        diagnostics: dict[str, Any] = {
            "integrator_policy": str(policy),
            "integrator_used": "euler",
            "integrator_columnarity": float(columnarity),
            "integrator_curvature": float(curvature),
            "integrator_euler_fs_error": None,
            "integrator_auto_policy_schema": (
                AUTO_EULER_RK4_POLICY_SCHEMA if str(policy) == "auto_euler_rk4" else None
            ),
            "integrator_auto_admit_euler": None,
            "integrator_euler_blockers": [],
            "integrator_condition_number": float(condition_number),
            "integrator_condition_pass": bool(condition_pass),
            "integrator_rho_miss_pass": bool(rho_miss_pass),
            "integrator_time_fraction": float(time_fraction),
            "integrator_euler_min_time_fraction": float(euler_min_time_fraction),
            "integrator_euler_time_gate_pass": bool(euler_time_gate_pass),
            **observable_gate,
            "integrator_error": None,
            "integrator_forced_policy": (
                None if forced_policy is None else str(forced_policy)
            ),
            "integrator_forced_policy_reason": (
                None if forced_policy_reason is None else str(forced_policy_reason)
            ),
        }
        if time_stop is None or abs(float(dt)) <= 0.0:
            diagnostics["integrator_used"] = "none"
            return theta_next_euler, theta_dot_euler, diagnostics
        if policy == "euler":
            return theta_next_euler, theta_dot_euler, diagnostics
        try:
            t0 = float(time_start)
            tm = float(time_start) + 0.5 * float(dt)
            t1 = float(time_stop)
            k1 = np.asarray(
                self._strict_qpu_hh_measured_integrator_stage_baseline(
                    checkpoint_index=int(checkpoint_index),
                    stage_time=t0,
                    layout=layout,
                    theta_runtime=theta0,
                    planning_audit=planning_audit,
                    scaffold_labels=scaffold_labels,
                    tier_name=str(tier_name),
                    budget_scale=float(budget_scale),
                )["theta_dot_step"],
                dtype=float,
            ).reshape(-1)
            k2 = np.asarray(
                self._strict_qpu_hh_measured_integrator_stage_baseline(
                    checkpoint_index=int(checkpoint_index),
                    stage_time=tm,
                    layout=layout,
                    theta_runtime=theta0 + 0.5 * float(dt) * k1,
                    planning_audit=planning_audit,
                    scaffold_labels=scaffold_labels,
                    tier_name=str(tier_name),
                    budget_scale=float(budget_scale),
                )["theta_dot_step"],
                dtype=float,
            ).reshape(-1)
            k3 = np.asarray(
                self._strict_qpu_hh_measured_integrator_stage_baseline(
                    checkpoint_index=int(checkpoint_index),
                    stage_time=tm,
                    layout=layout,
                    theta_runtime=theta0 + 0.5 * float(dt) * k2,
                    planning_audit=planning_audit,
                    scaffold_labels=scaffold_labels,
                    tier_name=str(tier_name),
                    budget_scale=float(budget_scale),
                )["theta_dot_step"],
                dtype=float,
            ).reshape(-1)
            k4 = np.asarray(
                self._strict_qpu_hh_measured_integrator_stage_baseline(
                    checkpoint_index=int(checkpoint_index),
                    stage_time=t1,
                    layout=layout,
                    theta_runtime=theta0 + float(dt) * k3,
                    planning_audit=planning_audit,
                    scaffold_labels=scaffold_labels,
                    tier_name=str(tier_name),
                    budget_scale=float(budget_scale),
                )["theta_dot_step"],
                dtype=float,
            ).reshape(-1)
            theta_dot_rk4 = np.asarray((k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0, dtype=float).reshape(-1)
            theta_next_rk4 = np.asarray(theta0 + float(dt) * theta_dot_rk4, dtype=float).reshape(-1)
            euler_fs_error = self._theta_fs_distance(theta_next_rk4 - theta_next_euler, baseline=baseline)
            diagnostics["integrator_euler_fs_error"] = float(euler_fs_error)
        except Exception as exc:
            diagnostics["integrator_error"] = f"{type(exc).__name__}: {exc}"
            raise
        if policy == "rk4":
            diagnostics["integrator_used"] = "rk4"
            return theta_next_rk4, theta_dot_rk4, diagnostics
        geometry_gate_pass = bool(
            float(columnarity) >= float(getattr(self.cfg, "integrator_columnarity_threshold", 0.80))
            and float(curvature) <= float(getattr(self.cfg, "integrator_curvature_threshold", 0.10))
        )
        euler_error_pass = bool(
            float(diagnostics["integrator_euler_fs_error"])
            <= float(getattr(self.cfg, "integrator_euler_fs_error_threshold", 1.0e-3))
        )
        diagnostics.update(
            _auto_euler_blocker_diagnostics(
                geometry_gate_pass=bool(geometry_gate_pass),
                euler_error_pass=bool(euler_error_pass),
                condition_pass=bool(condition_pass),
                rho_miss_pass=bool(rho_miss_pass),
                euler_time_gate_pass=bool(euler_time_gate_pass),
                observable_gate_pass=bool(observable_gate["integrator_euler_observable_gate_pass"]),
            )
        )
        use_euler = bool(diagnostics["integrator_auto_admit_euler"])
        if use_euler:
            diagnostics["integrator_used"] = "euler"
            return theta_next_euler, theta_dot_euler, diagnostics
        diagnostics["integrator_used"] = "rk4"
        return theta_next_rk4, theta_dot_rk4, diagnostics

    def _no_advance_integrator_diagnostics(self) -> dict[str, Any]:
        return {
            "integrator_policy": str(self._integrator_policy()),
            "integrator_used": "none",
            "integrator_columnarity": None,
            "integrator_curvature": None,
            "integrator_euler_fs_error": None,
            "integrator_condition_number": None,
            "integrator_condition_pass": None,
            "integrator_geometry_gate_pass": None,
            "integrator_euler_error_pass": None,
            "integrator_auto_policy_schema": (
                AUTO_EULER_RK4_POLICY_SCHEMA
                if str(self._integrator_policy()) == "auto_euler_rk4"
                else None
            ),
            "integrator_auto_admit_euler": None,
            "integrator_euler_blockers": [],
            "integrator_rho_miss_pass": None,
            "integrator_time_fraction": None,
            "integrator_euler_min_time_fraction": float(
                getattr(self.cfg, "integrator_euler_min_time_fraction", 0.0)
            ),
            "integrator_euler_time_gate_pass": None,
            "integrator_euler_observable_gate_pass": None,
            "integrator_euler_site_span": None,
            "integrator_euler_primary_density_span": None,
            "integrator_euler_energy_span": None,
            "integrator_error": None,
        }

    def _energy_hpsi_variance(self, psi: np.ndarray, *, compiled_h: Any | None = None) -> tuple[float, np.ndarray, float]:
        psi_vec = np.asarray(psi, dtype=complex).reshape(-1)
        hpsi = apply_compiled_polynomial(psi_vec, self._compiled_h if compiled_h is None else compiled_h)
        energy = float(np.real(np.vdot(psi_vec, hpsi)))
        variance = float(max(0.0, np.real(np.vdot(hpsi, hpsi)) - energy * energy))
        return float(energy), np.asarray(hpsi, dtype=complex), float(variance)

    def _position_jump_penalty(self, position_id: int) -> float:
        if self._previous_append_position is None:
            return 0.0
        return float(
            abs(int(position_id) - int(self._previous_append_position))
            / max(int(self.current_layout.logical_parameter_count), 1)
        )

    """
    Built Math: H_flow(t_k) := H_static + H_drive(t_k), with identity
    skipped by the runtime parameter layout as a global phase.
    """
    def _driven_spinful_hamiltonian_flow_candidate(
        self,
        *,
        time_start: float | None,
        time_stop: float | None,
    ) -> tuple[AnsatzTerm, dict[str, Any]] | None:
        if (
            self._drive_config is None
            or (self._drive_coeff_provider_exyz is None and self._drive_model is None)
            or str(getattr(self, "_family_key", "")) not in _DRIVEN_HAMILTONIAN_FLOW_FAMILIES
            or str(getattr(self.cfg, "mode", "")) != "exact_v1"
            or time_start is None
            or time_stop is None
        ):
            return None
        sample_time = self._projection_sample_time(float(time_start), float(time_stop))
        step_hamiltonian = self._step_hamiltonian_artifacts(float(sample_time))
        if int(step_hamiltonian.drive_term_count) <= 0:
            return None
        return (
            AnsatzTerm(label="ham_full", polynomial=step_hamiltonian.h_poly),
            {
                "dynamic_ham_full_active": True,
                "dynamic_ham_full_source": "step_hamiltonian_artifacts",
                "dynamic_ham_full_sample_time": float(sample_time),
                "dynamic_ham_full_physical_time": float(step_hamiltonian.physical_time),
                "dynamic_ham_full_drive_term_count": int(step_hamiltonian.drive_term_count),
            },
        )

    def _candidate_pool_terms(
        self,
        *,
        baseline: Mapping[str, Any] | None = None,
        time_start: float | None = None,
        time_stop: float | None = None,
    ) -> list[tuple[int, AnsatzTerm]]:
        raw_append_family_pool = getattr(self.replay_context, "append_family_pool", None)
        append_pool_terms = list(
            self.replay_context.family_pool if raw_append_family_pool is None else raw_append_family_pool
        )
        append_info = dict(
            getattr(self.replay_context, "append_family_info", None)
            or {
                "requested": "match_replay",
                "resolved": self.replay_context.family_info.get("resolved", "unknown"),
                "resolution_source": "replay_family",
                "fallback_used": False,
                "uses_replay_pool": True,
            }
        )
        raw_append_pool_meta = getattr(self.replay_context, "append_pool_meta", None)
        append_meta = dict(
            self.replay_context.pool_meta if raw_append_pool_meta is None else raw_append_pool_meta
        )
        dynamic_ham_full_meta: dict[str, Any] = {"dynamic_ham_full_active": False}
        if not bool(getattr(self.cfg, "append_enabled", True)):
            self._last_candidate_pool_diagnostics = {
                "append_enabled": False,
                "append_disabled_reason": "checkpoint_controller_append_enabled_false",
                "replay_family_requested": str(self.replay_context.family_info.get("requested", "")),
                "replay_family_resolved": str(self.replay_context.family_info.get("resolved", "")),
                "replay_family_resolution_source": str(
                    self.replay_context.family_info.get("resolution_source", "")
                ),
                "replay_family_fallback_used": bool(
                    self.replay_context.family_info.get("fallback_used", False)
                ),
                "append_family_requested": str(append_info.get("requested", "match_replay")),
                "append_family_resolved": str(append_info.get("resolved", "")),
                "append_family_resolution_source": str(
                    append_info.get("resolution_source", "")
                ),
                "append_family_fallback_used": bool(append_info.get("fallback_used", False)),
                "append_uses_replay_pool": bool(append_info.get("uses_replay_pool", False)),
                "family_pool_sizes": {
                    "replay_family_pool_count": int(len(self.replay_context.family_pool)),
                    "append_family_pool_count": int(len(append_pool_terms)),
                    "replay_terms_count": int(len(self.replay_context.replay_terms)),
                    "current_source_label_count": int(len(self._current_source_labels())),
                    "available_candidate_count": 0,
                    "repeated_candidate_count": 0,
                    "repeated_suppressed_count": 0,
                    "repeated_allowed_count": 0,
                },
                "candidate_pool_complete": bool(append_meta.get("candidate_pool_complete", True)),
                "candidate_pool_incomplete_reason": append_meta.get("incomplete_reason", None),
                "candidate_label_samples": [],
                "current_source_label_samples": sorted(str(x) for x in self._current_source_labels())[:8],
                "repeated_label_samples": [],
                "repeat_reopen_reason": None,
                "allow_repeats": bool(self.allow_repeats),
                **dict(dynamic_ham_full_meta),
            }
            return []
        dynamic_ham_full = self._driven_spinful_hamiltonian_flow_candidate(
            time_start=time_start,
            time_stop=time_stop,
        )
        if dynamic_ham_full is not None:
            dynamic_term, dynamic_ham_full_meta = dynamic_ham_full
            for term_index, term in enumerate(append_pool_terms):
                if str(getattr(term, "label", "")) == "ham_full":
                    append_pool_terms[int(term_index)] = dynamic_term
                    break
            else:
                append_pool_terms.insert(0, dynamic_term)
                dynamic_ham_full_meta = {
                    **dict(dynamic_ham_full_meta),
                    "dynamic_ham_full_inserted_missing_static_record": True,
                }
        current_source_labels = self._current_source_labels()
        available: list[tuple[int, AnsatzTerm]] = []
        repeated: list[tuple[int, AnsatzTerm]] = []
        dynamic_ham_full_repeat_reopened = False
        for pool_index, term in enumerate(append_pool_terms):
            candidate = (int(pool_index), term)
            source_label = str(term.label)
            if source_label in current_source_labels:
                repeated.append(candidate)
                if bool(self.allow_repeats):
                    available.append(candidate)
                continue
            available.append(candidate)
        if available:
            preferred_site_index = self._exact_v1_preferred_site_index_at_time(
                baseline=baseline,
                time_start=time_start,
                time_stop=time_stop,
            )
            if preferred_site_index is not None:
                available_has_preferred_turn = any(
                    self._candidate_primary_site_index(getattr(term, "label", None)) == int(preferred_site_index)
                    and self._candidate_is_site_turn_family(getattr(term, "label", None))
                    for _, term in available
                )
                if not available_has_preferred_turn:
                    repeated_preferred_turn = [
                        (int(pool_index), term)
                        for pool_index, term in repeated
                        if self._candidate_primary_site_index(getattr(term, "label", None))
                        == int(preferred_site_index)
                        and self._candidate_is_site_turn_family(getattr(term, "label", None))
                    ]
                    if repeated_preferred_turn:
                        seen_pool_indices = {int(pool_index) for pool_index, _ in available}
                        available.extend(
                            (int(pool_index), term)
                            for pool_index, term in repeated_preferred_turn
                            if int(pool_index) not in seen_pool_indices
                        )
            returned = list(available)
            repeated_returned = {
                int(pool_index)
                for pool_index, _ in returned
                if any(int(pool_index) == int(rep_idx) for rep_idx, _rep in repeated)
            }
            self._last_candidate_pool_diagnostics = {
                "replay_family_requested": str(self.replay_context.family_info.get("requested", "")),
                "replay_family_resolved": str(self.replay_context.family_info.get("resolved", "")),
                "replay_family_resolution_source": str(
                    self.replay_context.family_info.get("resolution_source", "")
                ),
                "replay_family_fallback_used": bool(
                    self.replay_context.family_info.get("fallback_used", False)
                ),
                "append_family_requested": str(append_info.get("requested", "match_replay")),
                "append_family_resolved": str(append_info.get("resolved", "")),
                "append_family_resolution_source": str(
                    append_info.get("resolution_source", "")
                ),
                "append_family_fallback_used": bool(append_info.get("fallback_used", False)),
                "append_uses_replay_pool": bool(append_info.get("uses_replay_pool", False)),
                "family_pool_sizes": {
                    "replay_family_pool_count": int(len(self.replay_context.family_pool)),
                    "append_family_pool_count": int(len(append_pool_terms)),
                    "replay_terms_count": int(len(self.replay_context.replay_terms)),
                    "current_source_label_count": int(len(current_source_labels)),
                    "available_candidate_count": int(len(returned)),
                    "repeated_candidate_count": int(len(repeated)),
                    "repeated_suppressed_count": int(
                        max(0, len(repeated) - len(repeated_returned))
                        if not bool(self.allow_repeats)
                        else 0
                    ),
                    "repeated_allowed_count": int(len(repeated_returned)),
                },
                "candidate_pool_complete": bool(
                    append_meta.get("candidate_pool_complete", True)
                ),
                "candidate_pool_incomplete_reason": append_meta.get(
                    "incomplete_reason",
                    None,
                ),
                "candidate_label_samples": [str(term.label) for _, term in returned[:8]],
                "current_source_label_samples": sorted(str(x) for x in current_source_labels)[:8],
                "repeated_label_samples": [str(term.label) for _, term in repeated[:8]],
                "repeat_reopen_reason": (
                    (
                        "dynamic_ham_full_time_dependent"
                        if bool(dynamic_ham_full_repeat_reopened)
                        else (
                            "allow_repeats"
                            if bool(self.allow_repeats)
                            else "preferred_site_turn_reopen"
                        )
                    )
                    if repeated_returned
                    else None
                ),
                "allow_repeats": bool(self.allow_repeats),
                "dynamic_ham_full_repeat_reopened": bool(dynamic_ham_full_repeat_reopened),
                **dict(dynamic_ham_full_meta),
            }
            return returned
        if self._exact_v1_sign_reversal_repeat_reopen_active(
            baseline=baseline,
            time_start=time_start,
            time_stop=time_stop,
        ):
            self._last_candidate_pool_diagnostics = {
                "replay_family_requested": str(self.replay_context.family_info.get("requested", "")),
                "replay_family_resolved": str(self.replay_context.family_info.get("resolved", "")),
                "replay_family_resolution_source": str(
                    self.replay_context.family_info.get("resolution_source", "")
                ),
                "replay_family_fallback_used": bool(
                    self.replay_context.family_info.get("fallback_used", False)
                ),
                "append_family_requested": str(append_info.get("requested", "match_replay")),
                "append_family_resolved": str(append_info.get("resolved", "")),
                "append_family_resolution_source": str(
                    append_info.get("resolution_source", "")
                ),
                "append_family_fallback_used": bool(append_info.get("fallback_used", False)),
                "append_uses_replay_pool": bool(append_info.get("uses_replay_pool", False)),
                "family_pool_sizes": {
                    "replay_family_pool_count": int(len(self.replay_context.family_pool)),
                    "append_family_pool_count": int(len(append_pool_terms)),
                    "replay_terms_count": int(len(self.replay_context.replay_terms)),
                    "current_source_label_count": int(len(current_source_labels)),
                    "available_candidate_count": int(len(repeated)),
                    "repeated_candidate_count": int(len(repeated)),
                    "repeated_suppressed_count": 0,
                    "repeated_allowed_count": int(len(repeated)),
                },
                "candidate_pool_complete": bool(
                    append_meta.get("candidate_pool_complete", True)
                ),
                "candidate_pool_incomplete_reason": append_meta.get(
                    "incomplete_reason",
                    None,
                ),
                "candidate_label_samples": [str(term.label) for _, term in repeated[:8]],
                "current_source_label_samples": sorted(str(x) for x in current_source_labels)[:8],
                "repeated_label_samples": [str(term.label) for _, term in repeated[:8]],
                "repeat_reopen_reason": "sign_reversal_window",
                "allow_repeats": bool(self.allow_repeats),
            "dynamic_ham_full_repeat_reopened": bool(dynamic_ham_full_repeat_reopened),
            **dict(dynamic_ham_full_meta),
            }
            return repeated
        self._last_candidate_pool_diagnostics = {
            "replay_family_requested": str(self.replay_context.family_info.get("requested", "")),
            "replay_family_resolved": str(self.replay_context.family_info.get("resolved", "")),
            "replay_family_resolution_source": str(
                self.replay_context.family_info.get("resolution_source", "")
            ),
            "replay_family_fallback_used": bool(
                self.replay_context.family_info.get("fallback_used", False)
            ),
            "append_family_requested": str(append_info.get("requested", "match_replay")),
            "append_family_resolved": str(append_info.get("resolved", "")),
            "append_family_resolution_source": str(
                append_info.get("resolution_source", "")
            ),
            "append_family_fallback_used": bool(append_info.get("fallback_used", False)),
            "append_uses_replay_pool": bool(append_info.get("uses_replay_pool", False)),
            "family_pool_sizes": {
                "replay_family_pool_count": int(len(self.replay_context.family_pool)),
                "append_family_pool_count": int(len(append_pool_terms)),
                "replay_terms_count": int(len(self.replay_context.replay_terms)),
                "current_source_label_count": int(len(current_source_labels)),
                "available_candidate_count": 0,
                "repeated_candidate_count": int(len(repeated)),
                "repeated_suppressed_count": int(len(repeated)),
                "repeated_allowed_count": 0,
            },
            "candidate_pool_complete": bool(append_meta.get("candidate_pool_complete", True)),
            "candidate_pool_incomplete_reason": append_meta.get("incomplete_reason", None),
            "candidate_label_samples": [],
            "current_source_label_samples": sorted(str(x) for x in current_source_labels)[:8],
            "repeated_label_samples": [str(term.label) for _, term in repeated[:8]],
            "repeat_reopen_reason": None,
            "allow_repeats": bool(self.allow_repeats),
            **dict(dynamic_ham_full_meta),
        }
        return []

    def _candidate_positions(self) -> list[int]:
        if not bool(getattr(self.cfg, "append_enabled", True)):
            return []
        logical_count = int(self.current_layout.logical_parameter_count)
        active_window = range(max(0, logical_count - self._cfg_int("active_window_size")), logical_count)
        return allowed_positions(
            n_params=int(logical_count),
            append_position=int(logical_count),
            active_window_indices=active_window,
            max_positions=self._cfg_int("max_probe_positions"),
        )

    def _candidate_executor_data(
        self,
        *,
        checkpoint_ctx: Any,
        cache: ExactCheckpointValueCache,
        geometry_memo: DerivedGeometryMemo,
        candidate_term: AnsatzTerm,
        candidate_pool_index: int,
        position_id: int,
    ) -> dict[str, Any]:
        memo_label = f"{candidate_term.label}__pool{int(candidate_pool_index)}"
        value, _ = geometry_memo.get_or_compute(
            DerivedGeometryKey(
                checkpoint_id=str(checkpoint_ctx.checkpoint_id),
                memo_family="candidate_executor_data",
                candidate_label=str(memo_label),
                position_id=int(position_id),
            ),
            compute=lambda: self._compute_candidate_executor_data(
                checkpoint_ctx=checkpoint_ctx,
                cache=cache,
                candidate_term=candidate_term,
                candidate_pool_index=int(candidate_pool_index),
                position_id=int(position_id),
            ),
        )
        return dict(value)

    def _compute_candidate_executor_data(
        self,
        *,
        checkpoint_ctx: Any,
        cache: ExactCheckpointValueCache,
        candidate_term: AnsatzTerm,
        candidate_pool_index: int,
        position_id: int,
    ) -> dict[str, Any]:
        candidate_identity = f"{candidate_term.label}__pool{int(candidate_pool_index)}"
        unique_label = (
            f"{candidate_term.label}__pool{int(candidate_pool_index)}"
            f"__append{self._append_counter}_p{int(position_id)}"
        )
        candidate_carrier = _build_candidate_carrier(
            candidate_term,
            logical_index=int(position_id),
            unique_label=str(unique_label),
            template_layout=self.current_layout,
            candidate_pool_index=int(candidate_pool_index),
        )
        aug_terms = list(self.current_terms)
        aug_terms.insert(int(position_id), candidate_carrier)
        aug_layout = _layout_from_carriers(aug_terms, template=self.current_layout)
        runtime_pos = int(runtime_insert_position(self.current_layout, int(position_id)))
        theta_aug = _insert_theta_block(
            self.current_theta,
            runtime_position=int(runtime_pos),
            width=int(len(candidate_carrier.runtime_specs)),
        )
        aug_executor = self._build_executor(aug_terms, aug_layout)
        block_indices = tuple(range(int(runtime_pos), int(runtime_pos + len(candidate_carrier.runtime_specs))))
        value, _ = cache.get_or_compute(
            GeometryValueKey(
                checkpoint_id=str(checkpoint_ctx.checkpoint_id),
                observable_family="candidate_insert_tangent_block",
                candidate_label=str(candidate_identity),
                position_id=int(position_id),
                runtime_indices=block_indices,
                group_key=None,
                grouping_mode=str(self.cfg.grouping_mode),
            ),
            tier_name="scout",
            compute=lambda: aug_executor.prepare_state_with_runtime_tangents(
                theta_aug,
                self.replay_context.psi_ref,
                runtime_indices=block_indices,
            ),
        )
        aug_psi, raw_tangents = value
        return {
            "candidate_carrier": candidate_carrier,
            "aug_terms": aug_terms,
            "aug_layout": aug_layout,
            "aug_executor": aug_executor,
            "theta_aug": theta_aug,
            "runtime_insert_position": int(runtime_pos),
            "runtime_block_indices": [int(x) for x in block_indices],
            "aug_psi": np.asarray(aug_psi, dtype=complex).reshape(-1),
            "raw_tangents": {int(k): np.asarray(v, dtype=complex).reshape(-1) for k, v in raw_tangents.items()},
        }

    def _candidate_primary_site_index(self, candidate_label: str | None) -> int | None:
        if candidate_label in {None, ""}:
            return None
        match = re.search(r"site=(\d+)", str(candidate_label))
        if match is None:
            return None
        return int(match.group(1))

    def _candidate_is_site_turn_family(self, candidate_label: str | None) -> bool:
        if candidate_label in {None, ""}:
            return False
        label = str(candidate_label)
        return bool(
            label.startswith("paop_full:paop_disp(")
            or label.startswith("paop_full:paop_cloud_p(")
        )

    def _exact_v1_preferred_site_index_at_time(
        self,
        *,
        baseline: Mapping[str, Any] | None,
        time_start: float | None,
        time_stop: float | None,
    ) -> int | None:
        if not self._exact_v1_guarded_turn_window_ranking_active() or baseline is None:
            return None
        if time_start is None or time_stop is None:
            return None
        theta_dot_step = np.asarray(baseline.get("theta_dot_step", ()), dtype=float).reshape(-1)
        if theta_dot_step.size != int(self.current_theta.size):
            return None
        theta_next = np.asarray(
            self.current_theta + float(time_stop - time_start) * theta_dot_step,
            dtype=float,
        ).reshape(-1)
        psi_next = np.asarray(
            self.current_executor.prepare_state(theta_next, self.replay_context.psi_ref),
            dtype=complex,
        ).reshape(-1)
        projected_obs = self._observable_snapshot(psi_next)
        projected_primary_density = float(self._primary_density_value_from_snapshot(projected_obs))
        if (not np.isfinite(projected_primary_density)) or (
            abs(float(projected_primary_density)) < float(self._exact_v1_d_shape_turn_window_abs_activation())
        ):
            return None
        site_values = np.asarray(
            projected_obs.get("site_occupations", projected_obs.get("n_site", [])),
            dtype=float,
        ).reshape(-1)
        if site_values.size < 2 or not np.all(np.isfinite(site_values)):
            return None
        return int(np.argmax(site_values))

    def _exact_v1_shortlist_preferred_site_index(
        self,
        *,
        checkpoint_ctx: Any,
        baseline: Mapping[str, Any] | None,
    ) -> int | None:
        return self._exact_v1_preferred_site_index_at_time(
            baseline=baseline,
            time_start=getattr(checkpoint_ctx, "time_start", None),
            time_stop=getattr(checkpoint_ctx, "time_stop", None),
        )

    def _inject_preferred_site_shortlist_record(
        self,
        *,
        records: Sequence[Mapping[str, Any]],
        shortlist: Sequence[Mapping[str, Any]],
        preferred_site_index: int | None,
    ) -> list[dict[str, Any]]:
        shortlist_records_local = [dict(item) for item in shortlist]
        if preferred_site_index is None or not shortlist_records_local:
            return shortlist_records_local

        def _record_key(item: Mapping[str, Any]) -> tuple[str, int]:
            return (
                str(item.get("candidate_identity", item.get("candidate_label"))),
                int(item.get("position_id", -1)),
            )

        def _best_record(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
            if not candidates:
                return None
            ordered = sorted(
                (dict(item) for item in candidates),
                key=lambda item: (
                    float(item.get("scout_score", float("-inf"))),
                    float(item.get("simple_score", float("-inf"))),
                ),
                reverse=True,
            )
            return ordered[0] if ordered else None

        preferred_records = [
            dict(item)
            for item in records
            if self._candidate_primary_site_index(item.get("candidate_label")) == int(preferred_site_index)
        ]
        if not preferred_records:
            return shortlist_records_local

        preferred_turn_records = [
            dict(item)
            for item in preferred_records
            if self._candidate_is_site_turn_family(item.get("candidate_label"))
        ]
        shortlist_has_preferred_turn = any(
            self._candidate_primary_site_index(item.get("candidate_label")) == int(preferred_site_index)
            and self._candidate_is_site_turn_family(item.get("candidate_label"))
            for item in shortlist_records_local
        )
        preferred: dict[str, Any] | None = None
        if not shortlist_has_preferred_turn:
            preferred = _best_record(preferred_turn_records)
        if preferred is None and any(
            self._candidate_primary_site_index(item.get("candidate_label")) == int(preferred_site_index)
            for item in shortlist_records_local
        ):
            return shortlist_records_local
        if preferred is None:
            preferred = _best_record(preferred_records)
        if preferred is None:
            return shortlist_records_local
        preferred_key = _record_key(preferred)
        if any(_record_key(item) == preferred_key for item in shortlist_records_local):
            return shortlist_records_local
        shortlist_records_local.sort(
            key=lambda item: (
                float(item.get("scout_score", float("-inf"))),
                float(item.get("simple_score", float("-inf"))),
            ),
            reverse=True,
        )
        shortlist_records_local[-1] = preferred
        return shortlist_records_local

    def _inject_spinful_lattice_hamiltonian_flow_shortlist_record(
        self,
        *,
        records: Sequence[Mapping[str, Any]],
        shortlist: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        shortlist_records_local = [dict(item) for item in shortlist]
        if (
            str(getattr(self, "_family_key", "")) not in _STATIC_SPINFUL_LATTICE_HAMILTONIAN_FLOW_FAMILIES
            or str(getattr(self.cfg, "mode", "")) != "exact_v1"
            or not shortlist_records_local
        ):
            return shortlist_records_local

        def _record_key(item: Mapping[str, Any]) -> tuple[str, int]:
            return (
                str(item.get("candidate_identity", item.get("candidate_label"))),
                int(item.get("position_id", -1)),
            )

        flow_records = [
            dict(item)
            for item in records
            if str(item.get("candidate_label", "")) == "ham_full"
        ]
        if not flow_records:
            return shortlist_records_local
        preferred = max(
            flow_records,
            key=lambda item: (
                float(item.get("scout_gain_ratio", 0.0)),
                float(item.get("scout_score", float("-inf"))),
                -float(item.get("position_jump_penalty", 0.0)),
            ),
        )
        preferred_key = _record_key(preferred)
        if any(_record_key(item) == preferred_key for item in shortlist_records_local):
            return shortlist_records_local
        shortlist_records_local.sort(
            key=lambda item: (
                float(item.get("scout_score", float("-inf"))),
                float(item.get("simple_score", float("-inf"))),
            ),
            reverse=True,
        )
        shortlist_records_local[-1] = dict(preferred)
        return shortlist_records_local

    def _scout_candidates_with_records(
        self,
        *,
        checkpoint_ctx: Any,
        cache: ExactCheckpointValueCache,
        geometry_memo: DerivedGeometryMemo,
        baseline: Mapping[str, Any],
        predicted_displacement: float,
        shortlist_cfg: FullScoreConfig | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        records: list[dict[str, Any]] = []
        current_terms_window = self.current_terms[
            max(0, len(self.current_terms) - self._cfg_int("active_window_size")) :
        ]
        for candidate_pool_index, candidate_term in self._candidate_pool_terms(
            baseline=baseline,
            time_start=(None if checkpoint_ctx.time_start is None else float(checkpoint_ctx.time_start)),
            time_stop=(None if checkpoint_ctx.time_stop is None else float(checkpoint_ctx.time_stop)),
        ):
            for position_id in self._candidate_positions():
                candidate_data = self._candidate_executor_data(
                    checkpoint_ctx=checkpoint_ctx,
                    cache=cache,
                    geometry_memo=geometry_memo,
                    candidate_term=candidate_term,
                    candidate_pool_index=int(candidate_pool_index),
                    position_id=int(position_id),
                )
                centered_cols: list[np.ndarray] = []
                for runtime_idx in candidate_data["runtime_block_indices"]:
                    tangent = np.asarray(candidate_data["raw_tangents"][int(runtime_idx)], dtype=complex)
                    centered = tangent - complex(np.vdot(baseline["psi"], tangent)) * np.asarray(baseline["psi"], dtype=complex)
                    centered_cols.append(np.asarray(centered, dtype=complex))
                u_block = np.column_stack(centered_cols) if centered_cols else np.zeros((baseline["psi"].size, 0), dtype=complex)
                residual_overlap_vec = np.asarray(np.real(u_block.conj().T @ baseline["residual_step"]), dtype=float).reshape(-1)
                residual_overlap_l2 = float(np.linalg.norm(residual_overlap_vec))
                C = np.asarray(np.real(u_block.conj().T @ u_block), dtype=float)
                C_reg = np.asarray(
                    C + self._cfg_float("candidate_regularization_lambda") * np.eye(int(C.shape[0])),
                    dtype=float,
                )
                C_reg_pinv = (
                    np.linalg.pinv(C_reg, rcond=self._cfg_float("pinv_rcond"))
                    if C_reg.size
                    else np.zeros((0, 0), dtype=float)
                )
                scout_lower_gain = (
                    float(max(0.0, float(residual_overlap_vec @ C_reg_pinv @ residual_overlap_vec)))
                    if residual_overlap_vec.size
                    else 0.0
                )
                scout_gain_ratio = float(
                    scout_lower_gain / max(float(baseline["norm_b_sq"]), 1e-14)
                )
                planning_stats = planning_stats_for_term(candidate_term, self._planning_audit)
                compile_est = self._compile_oracle.estimate(
                    candidate_term_count=max(1, len(candidate_data["runtime_block_indices"])),
                    position_id=int(position_id),
                    append_position=int(self.current_layout.logical_parameter_count),
                    refit_active_count=max(0, int(self.current_layout.logical_parameter_count) - int(position_id)),
                    candidate_term=candidate_term,
                )
                novelty = None
                if current_terms_window:
                    try:
                        novelty_info = self._novelty_oracle.estimate(
                            psi_state=np.asarray(baseline["psi"], dtype=complex),
                            candidate_label=str(candidate_term.label),
                            candidate_term=candidate_term,
                            window_terms=[_carrier_to_term(term) for term in current_terms_window],
                            window_labels=[str(term.label) for term in current_terms_window],
                            compiled_cache=self._compiled_poly_cache,
                            pauli_action_cache=self._pauli_action_cache,
                            novelty_eps=1e-6,
                        )
                        novelty = float(novelty_info.get("novelty", 0.0))
                    except Exception:
                        novelty = None
                position_jump_penalty = self._position_jump_penalty(int(position_id))
                temporal_prior_bonus = float(
                    self._temporal_ledger.candidate_probe_bonus(
                        candidate_identity=f"{candidate_term.label}__pool{int(candidate_pool_index)}",
                        position_id=int(position_id),
                        predicted_displacement=float(predicted_displacement),
                    )
                )
                legacy_simple_score = float(
                    residual_overlap_l2
                    + float(temporal_prior_bonus)
                    - float(self.cfg.compile_penalty_weight) * float(compile_est.proxy_total)
                    - float(self.cfg.measurement_penalty_weight) * float(planning_stats.groups_new)
                    - float(self.cfg.directional_penalty_weight) * float(position_jump_penalty)
                )
                scout_score = float(
                    scout_gain_ratio
                    + float(temporal_prior_bonus)
                    - float(self.cfg.compile_penalty_weight) * float(compile_est.proxy_total)
                    - float(self.cfg.measurement_penalty_weight) * float(planning_stats.groups_new)
                    - float(self.cfg.directional_penalty_weight) * float(position_jump_penalty)
                )
                records.append(
                    {
                        "candidate_label": str(candidate_term.label),
                        "candidate_identity": f"{candidate_term.label}__pool{int(candidate_pool_index)}",
                        "candidate_pool_index": int(candidate_pool_index),
                        "position_id": int(position_id),
                        "runtime_insert_position": int(candidate_data["runtime_insert_position"]),
                        "runtime_block_indices": list(candidate_data["runtime_block_indices"]),
                        "residual_overlap_l2": float(residual_overlap_l2),
                        "compile_proxy_total": float(compile_est.proxy_total),
                        "groups_new": float(planning_stats.groups_new),
                        "novelty": novelty,
                        "position_jump_penalty": float(position_jump_penalty),
                        "temporal_prior_bonus": float(temporal_prior_bonus),
                        "scout_lower_gain": float(scout_lower_gain),
                        "scout_gain_ratio": float(scout_gain_ratio),
                        "scout_score": float(scout_score),
                        "scout_score_kind": "shared_baseline_lower_gain_ratio_minus_penalties",
                        "simple_score": float(legacy_simple_score),
                        "candidate_data": candidate_data,
                        "candidate_term": candidate_term,
                    }
                )
        shortlist = shortlist_records(
            records,
            cfg=(self._shortlist_cfg if shortlist_cfg is None else shortlist_cfg),
            score_key="scout_score",
            tie_break_score_key="scout_score",
        )
        shortlisted = self._inject_preferred_site_shortlist_record(
            records=records,
            shortlist=shortlist,
            preferred_site_index=self._exact_v1_shortlist_preferred_site_index(
                checkpoint_ctx=checkpoint_ctx,
                baseline=baseline,
            ),
        )
        shortlisted = self._inject_spinful_lattice_hamiltonian_flow_shortlist_record(
            records=records,
            shortlist=shortlisted,
        )
        return shortlisted, [dict(item) for item in records]

    def _scout_candidates(
        self,
        *,
        checkpoint_ctx: Any,
        cache: ExactCheckpointValueCache,
        geometry_memo: DerivedGeometryMemo,
        baseline: Mapping[str, Any],
        predicted_displacement: float,
        shortlist_cfg: FullScoreConfig | None = None,
    ) -> list[dict[str, Any]]:
        shortlist, _records = self._scout_candidates_with_records(
            checkpoint_ctx=checkpoint_ctx,
            cache=cache,
            geometry_memo=geometry_memo,
            baseline=baseline,
            predicted_displacement=float(predicted_displacement),
            shortlist_cfg=shortlist_cfg,
        )
        self._last_scout_records = [dict(item) for item in _records]
        return shortlist

    def _compressed_confirm_gain(
        self,
        *,
        baseline: Mapping[str, Any],
        B: np.ndarray,
        C: np.ndarray,
        q: np.ndarray,
        fallback_w: np.ndarray,
    ) -> tuple[float, float, int, int]:
        K = np.asarray(baseline.get("K", np.zeros((0, 0), dtype=float)), dtype=float)
        f_vec = np.asarray(baseline.get("f", np.zeros(0, dtype=float)), dtype=float).reshape(-1)
        q_vec = np.asarray(q, dtype=float).reshape(-1)
        fallback = np.asarray(fallback_w, dtype=float).reshape(-1)
        B_mat = np.asarray(B, dtype=float)
        C_mat = np.asarray(C, dtype=float)
        if str(getattr(self.cfg, "confirm_score_mode", "exact_gain_ratio")) != "compressed_whitened_v1":
            gain = float(max(0.0, fallback @ (np.linalg.pinv(C_mat + self._cfg_float("candidate_regularization_lambda") * np.eye(int(C_mat.shape[0])), rcond=self._cfg_float("pinv_rcond")) @ fallback))) if C_mat.size else 0.0
            ratio = float(gain / max(float(baseline.get("norm_b_sq", 0.0)), 1.0e-14))
            return float(gain), float(ratio), 0, 0
        if K.size == 0 or q_vec.size == 0:
            return 0.0, 0.0, 0, 0
        evals, evecs = np.linalg.eigh(K)
        if evals.size == 0:
            return 0.0, 0.0, 0, 0
        tol = max(1.0e-12, self._cfg_float("pinv_rcond") * float(np.max(np.abs(evals))))
        support = np.flatnonzero(evals > tol)
        if support.size <= 0:
            S_tilde = np.asarray(
                C_mat + self._cfg_float("candidate_regularization_lambda") * np.eye(int(C_mat.shape[0])),
                dtype=float,
            )
            gain = float(max(0.0, fallback @ (np.linalg.pinv(S_tilde, rcond=self._cfg_float("pinv_rcond")) @ fallback))) if fallback.size else 0.0
            ratio = float(gain / max(float(baseline.get("norm_b_sq", 0.0)), 1.0e-14))
            return float(gain), float(ratio), 0, 0
        V = np.asarray(evecs[:, support], dtype=float)
        sigma_inv = 1.0 / np.sqrt(np.asarray(evals[support], dtype=float))
        z = np.asarray(sigma_inv * (V.T @ f_vec), dtype=float).reshape(-1)
        Gamma = np.asarray((sigma_inv[:, None]) * (V.T @ B_mat), dtype=float)
        w_vec = np.asarray(q_vec - Gamma.T @ z, dtype=float).reshape(-1)
        rank = int(Gamma.shape[0])
        if rank <= 0:
            S_tilde = np.asarray(
                C_mat + self._cfg_float("candidate_regularization_lambda") * np.eye(int(C_mat.shape[0])),
                dtype=float,
            )
            gain = float(max(0.0, w_vec @ (np.linalg.pinv(S_tilde, rcond=self._cfg_float("pinv_rcond")) @ w_vec))) if w_vec.size else 0.0
            ratio = float(gain / max(float(baseline.get("norm_b_sq", 0.0)), 1.0e-14))
            return float(gain), float(ratio), 0, 0
        requested = int(np.ceil(float(self.cfg.confirm_compress_fraction) * float(rank)))
        modes_used = max(int(self.cfg.confirm_compress_min_modes), requested)
        max_modes = int(self.cfg.confirm_compress_max_modes)
        if max_modes > 0:
            modes_used = min(int(modes_used), int(max_modes))
        modes_used = min(int(rank), max(0, int(modes_used)))
        row_norms = np.linalg.norm(Gamma, axis=1)
        selected_rows = np.argsort(-row_norms, kind="mergesort")[: int(modes_used)] if modes_used > 0 else np.asarray([], dtype=int)
        Gamma_selected = np.asarray(Gamma[selected_rows, :], dtype=float) if selected_rows.size > 0 else np.zeros((0, int(q_vec.size)), dtype=float)
        S_tilde = np.asarray(
            C_mat
            + self._cfg_float("candidate_regularization_lambda") * np.eye(int(C_mat.shape[0]))
            - Gamma_selected.T @ Gamma_selected,
            dtype=float,
        )
        S_tilde_pinv = np.linalg.pinv(S_tilde, rcond=self._cfg_float("pinv_rcond")) if S_tilde.size else np.zeros((0, 0), dtype=float)
        gain = float(max(0.0, w_vec @ (S_tilde_pinv @ w_vec))) if w_vec.size else 0.0
        ratio = float(gain / max(float(baseline.get("norm_b_sq", 0.0)), 1.0e-14))
        return float(gain), float(ratio), int(modes_used), int(rank)

    def _confirm_score_payload(
        self,
        *,
        baseline: Mapping[str, Any],
        B: np.ndarray,
        C: np.ndarray,
        q: np.ndarray,
        w: np.ndarray,
        gain_ratio: float,
        gain_exact: float,
        groups_new: float,
        directional_change_l2: float | None,
    ) -> dict[str, Any]:
        directional_penalty = 0.0 if directional_change_l2 is None else float(directional_change_l2)
        directional_penalty_value = float(
            float(self.cfg.directional_penalty_weight) * directional_penalty
        )
        measurement_penalty_value = float(
            float(self.cfg.measurement_penalty_weight) * float(groups_new)
        )
        adjusted_gain = float(
            float(gain_ratio)
            - directional_penalty_value
            - measurement_penalty_value
        )
        mode = str(getattr(self.cfg, "confirm_score_mode", "exact_gain_ratio"))
        gain_ratio_gate = bool(float(gain_ratio) >= float(self.cfg.gain_ratio_threshold))
        gain_exact_gate = bool(float(gain_exact) >= float(self.cfg.append_margin_abs))
        score_threshold = float(self._exact_v1_live_confirm_score_threshold())
        if mode != "compressed_whitened_v1":
            score_gate = bool(float(adjusted_gain) >= float(score_threshold))
            if not gain_ratio_gate:
                gate_reason = "gain_ratio_below_threshold"
            elif not gain_exact_gate:
                gate_reason = "gain_exact_below_threshold"
            elif not score_gate:
                gate_reason = "confirm_score_below_threshold"
            else:
                gate_reason = None
            return {
                "adjusted_gain": float(adjusted_gain),
                "confirm_score": float(adjusted_gain),
                "confirm_score_kind": "geometry_gain_ratio_minus_penalties",
                "confirm_compress_modes_used": 0,
                "confirm_support_rank": 0,
                "confirm_gain_ratio_raw": float(gain_ratio),
                "confirm_gain_exact_raw": float(gain_exact),
                "confirm_compressed_gain_ratio": None,
                "confirm_compressed_gain_exact": None,
                "confirm_directional_change_l2": (
                    None if directional_change_l2 is None else float(directional_change_l2)
                ),
                "confirm_directional_penalty_value": float(directional_penalty_value),
                "confirm_groups_new": float(groups_new),
                "confirm_measurement_penalty_value": float(measurement_penalty_value),
                "confirm_score_threshold": float(score_threshold),
                "confirm_gain_ratio_threshold": float(self.cfg.gain_ratio_threshold),
                "confirm_gain_exact_threshold": float(self.cfg.append_margin_abs),
                "confirm_gain_ratio_gate": bool(gain_ratio_gate),
                "confirm_gain_exact_gate": bool(gain_exact_gate),
                "confirm_score_gate": bool(score_gate),
                "confirm_gate_passed": bool(gain_ratio_gate and gain_exact_gate and score_gate),
                "confirm_gate_reason": gate_reason,
            }
        compressed_gain, compressed_ratio, modes_used, support_rank = self._compressed_confirm_gain(
            baseline=baseline,
            B=np.asarray(B, dtype=float),
            C=np.asarray(C, dtype=float),
            q=np.asarray(q, dtype=float).reshape(-1),
            fallback_w=np.asarray(w, dtype=float).reshape(-1),
        )
        confirm_score = float(
            float(compressed_ratio)
            - directional_penalty_value
            - measurement_penalty_value
        )
        score_gate = bool(float(confirm_score) >= float(score_threshold))
        if not gain_ratio_gate:
            gate_reason = "gain_ratio_below_threshold"
        elif not gain_exact_gate:
            gate_reason = "gain_exact_below_threshold"
        elif not score_gate:
            gate_reason = "confirm_score_below_threshold"
        else:
            gate_reason = None
        return {
            "adjusted_gain": float(adjusted_gain),
            "confirm_score": float(confirm_score),
            "confirm_score_kind": "compressed_whitened_lower_gain_ratio_minus_penalties",
            "confirm_compress_modes_used": int(modes_used),
            "confirm_support_rank": int(support_rank),
            "confirm_gain_ratio_raw": float(gain_ratio),
            "confirm_gain_exact_raw": float(gain_exact),
            "confirm_compressed_gain_ratio": float(compressed_ratio),
            "confirm_compressed_gain_exact": float(compressed_gain),
            "confirm_directional_change_l2": (
                None if directional_change_l2 is None else float(directional_change_l2)
            ),
            "confirm_directional_penalty_value": float(directional_penalty_value),
            "confirm_groups_new": float(groups_new),
            "confirm_measurement_penalty_value": float(measurement_penalty_value),
            "confirm_score_threshold": float(score_threshold),
            "confirm_gain_ratio_threshold": float(self.cfg.gain_ratio_threshold),
            "confirm_gain_exact_threshold": float(self.cfg.append_margin_abs),
            "confirm_gain_ratio_gate": bool(gain_ratio_gate),
            "confirm_gain_exact_gate": bool(gain_exact_gate),
            "confirm_score_gate": bool(score_gate),
            "confirm_gate_passed": bool(gain_ratio_gate and gain_exact_gate and score_gate),
            "confirm_gate_reason": gate_reason,
        }

    def _clear_confirm_payload(
        self,
        record: Mapping[str, Any],
        *,
        confirm_error: str,
        rejection_reason: str,
    ) -> dict[str, Any]:
        rec = dict(record)
        rec["gain_exact"] = None
        rec["gain_ratio"] = None
        rec["adjusted_gain"] = float("-inf")
        rec["confirm_score"] = None
        rec["confirm_score_kind"] = "not_confirmed"
        rec["confirm_compress_modes_used"] = 0
        rec["confirm_support_rank"] = 0
        rec["confirm_gain_ratio_raw"] = None
        rec["confirm_gain_exact_raw"] = None
        rec["confirm_compressed_gain_ratio"] = None
        rec["confirm_compressed_gain_exact"] = None
        rec["confirm_directional_change_l2"] = None
        rec["confirm_directional_penalty_value"] = None
        rec["confirm_groups_new"] = rec.get("groups_new", None)
        rec["confirm_measurement_penalty_value"] = None
        rec["confirm_score_threshold"] = float(self._exact_v1_live_confirm_score_threshold())
        rec["confirm_gain_ratio_threshold"] = float(self.cfg.gain_ratio_threshold)
        rec["confirm_gain_exact_threshold"] = float(self.cfg.append_margin_abs)
        rec["confirm_gain_ratio_gate"] = False
        rec["confirm_gain_exact_gate"] = False
        rec["confirm_score_gate"] = False
        rec["confirm_gate_passed"] = False
        rec["confirm_gate_reason"] = str(rejection_reason)
        rec["confirm_backend_info"] = None
        rec["confirm_error"] = str(confirm_error)
        rec["candidate_summary"] = replace(
            rec["candidate_summary"],
            gain_exact=None,
            gain_ratio=None,
            admissible=False,
            rejection_reason=str(rejection_reason),
            decision_metric="not_confirmed",
            oracle_estimate_kind=None,
        )
        return rec

    def _confirm_rank_key(self, rec: Mapping[str, Any]) -> tuple[float, float, float, float, int, int]:
        raw_score = rec.get("confirm_score", rec.get("adjusted_gain", float("-inf")))
        score = float("-inf") if raw_score is None else float(raw_score)
        summary = rec["candidate_summary"]
        return (
            -score,
            float(summary.position_jump_penalty),
            float(summary.compile_proxy_total),
            float(summary.groups_new),
            int(summary.candidate_pool_index),
            int(summary.position_id),
        )

    def _passes_exact_confirm_thresholds(self, rec: Mapping[str, Any]) -> bool:
        gain_ratio = rec.get("gain_ratio")
        gain_exact = rec.get("gain_exact")
        if gain_ratio is None or gain_exact is None:
            return False
        if float(gain_ratio) < float(self.cfg.gain_ratio_threshold):
            return False
        if float(gain_exact) < float(self.cfg.append_margin_abs):
            return False
        return True

    def _finite_float_or_none(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if np.isfinite(out) else None

    def _repair_retry_admission_policy(self) -> str:
        return str(getattr(self.cfg, "repair_retry_admission_policy", "strict")).strip().lower()

    def _repair_retry_rescue_attempt_policy(self) -> str:
        return str(getattr(self.cfg, "repair_retry_rescue_attempt", "terminal_attempt_only")).strip().lower()

    def _repair_retry_rescue_candidate_rejection_reason(
        self,
        rec: Mapping[str, Any],
        *,
        min_gain_ratio: float,
    ) -> str | None:
        summary = rec.get("candidate_summary", None)
        if summary is None:
            return "candidate_summary_missing"
        if bool(getattr(summary, "admissible", True)) is False:
            reason = getattr(summary, "rejection_reason", None)
            return "candidate_summary_not_admissible" if reason in {None, ""} else str(reason)
        if "candidate_data" not in rec or not isinstance(rec.get("candidate_data"), Mapping):
            return "candidate_data_missing"
        if rec.get("theta_dot_aug", None) is None:
            return "theta_dot_aug_missing"
        gain_ratio = self._finite_float_or_none(rec.get("gain_ratio"))
        if gain_ratio is None:
            return "gain_ratio_missing"
        gain_exact = self._finite_float_or_none(rec.get("gain_exact"))
        if gain_exact is None:
            return "gain_exact_missing"
        raw_score = rec.get("confirm_score", None)
        if raw_score is None:
            raw_score = rec.get("adjusted_gain", None)
        score = self._finite_float_or_none(raw_score)
        if score is None:
            return "candidate_score_missing"
        if float(gain_ratio) < float(min_gain_ratio):
            return "gain_ratio_below_rescue_min"
        if not bool(self.allow_repeats):
            identity = str(rec.get("candidate_identity", rec.get("candidate_label", "")))
            label = str(rec.get("candidate_label", ""))
            current_labels = {str(item) for item in self._current_scaffold_labels()}
            current_source_labels = {str(item) for item in self._current_source_labels()}
            if identity in current_labels or label in current_source_labels:
                return "repeat_candidate"
        return None

    def _best_confirmed_candidate_for_repair_rescue(
        self,
        confirmed_candidates: Sequence[Mapping[str, Any]],
        *,
        min_gain_ratio: float,
    ) -> tuple[dict[str, Any] | None, str | None]:
        rejection_counts: dict[str, int] = {}
        for rec in self._sorted_confirmed_by_gain(confirmed_candidates):
            reason = self._repair_retry_rescue_candidate_rejection_reason(
                rec,
                min_gain_ratio=float(min_gain_ratio),
            )
            if reason is None:
                return dict(rec), None
            rejection_counts[str(reason)] = int(rejection_counts.get(str(reason), 0)) + 1
        if not confirmed_candidates:
            return None, "no_confirmed_candidates"
        if rejection_counts:
            reason = max(rejection_counts.items(), key=lambda item: (int(item[1]), str(item[0])))[0]
            return None, str(reason)
        return None, "no_rescue_candidate"

    def _forecast_veto_reason_for_repair_rescue(
        self,
        *,
        checkpoint_index: int,
        baseline: Mapping[str, Any],
        candidate: Mapping[str, Any],
        dt: float,
        time_stop: float,
        stay_forecast: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
        try:
            anchor_predicted_displacement = self._predicted_displacement(
                dt=float(dt),
                baseline=baseline,
            )
            if stay_forecast is None:
                stay_theta_forecast = np.asarray(
                    self.current_theta + float(dt) * np.asarray(baseline["theta_dot_step"], dtype=float),
                    dtype=float,
                ).reshape(-1)
                stay_payload = self._local_projective_forecast_rollout(
                    checkpoint_index=int(checkpoint_index),
                    time_stop=float(time_stop),
                    executor=self.current_executor,
                    layout=self.current_layout,
                    theta_runtime_start=stay_theta_forecast,
                    theta_dot_step=np.asarray(baseline["theta_dot_step"], dtype=float).reshape(-1),
                    planning_audit=self._planning_audit,
                    scaffold_labels=self._current_scaffold_labels(),
                    immediate_gain_ratio=float(getattr(baseline["summary"], "step_gain_ratio", 0.0)),
                    anchor_summary=baseline["summary"],
                    anchor_predicted_displacement=float(anchor_predicted_displacement),
                )[0]
            else:
                stay_payload = dict(stay_forecast)
            if str(self.cfg.mode) == "exact_v1":
                scaled_candidate, selected_payload = self._select_exact_v1_candidate_step_scale(
                    checkpoint_index=int(checkpoint_index),
                    baseline_theta_dot=np.asarray(baseline["theta_dot_step"], dtype=float).reshape(-1),
                    selected=candidate,
                    dt=float(dt),
                    time_stop=float(time_stop),
                    anchor_summary=baseline["summary"],
                    anchor_predicted_displacement=float(anchor_predicted_displacement),
                )
                candidate_out = dict(scaled_candidate)
            else:
                selected_theta_forecast = np.asarray(
                    candidate["candidate_data"]["theta_aug"]
                    + float(dt) * np.asarray(candidate["theta_dot_aug"], dtype=float),
                    dtype=float,
                ).reshape(-1)
                selected_payload = self._local_projective_forecast_rollout(
                    checkpoint_index=int(checkpoint_index),
                    time_stop=float(time_stop),
                    executor=candidate["candidate_data"]["aug_executor"],
                    layout=candidate["candidate_data"]["aug_layout"],
                    theta_runtime_start=selected_theta_forecast,
                    theta_dot_step=np.asarray(candidate["theta_dot_aug"], dtype=float).reshape(-1),
                    planning_audit=self._build_planning_audit_for_terms(candidate["candidate_data"]["aug_terms"]),
                    scaffold_labels=[str(carrier.label) for carrier in candidate["candidate_data"]["aug_terms"]],
                    immediate_gain_ratio=float(candidate.get("gain_ratio", 0.0)),
                    anchor_summary=baseline["summary"],
                    anchor_predicted_displacement=float(anchor_predicted_displacement),
                )[0]
                candidate_out = dict(candidate)
            if str(self.cfg.mode) == "exact_v1":
                try:
                    exact_v1_veto_reason = self._exact_v1_forecast_override_reason(
                        stay_forecast=stay_payload,
                        selected_forecast=selected_payload,
                        action_kind="append_candidate",
                        selected=candidate_out,
                    )
                except KeyError as exc:
                    missing_key = "" if not exc.args else str(exc.args[0])
                    if missing_key not in {
                        "fidelity_exact_next",
                        "abs_doublon_error_next",
                        "site_occupations_abs_error_max_next",
                        "abs_energy_total_error_next",
                    }:
                        raise
                    exact_v1_veto_reason = None
                if exact_v1_veto_reason is not None:
                    return dict(candidate_out), dict(selected_payload), str(exact_v1_veto_reason)
            veto_reason = self._local_forecast_override_reason(
                stay_forecast=stay_payload,
                selected_forecast=selected_payload,
                selected=candidate_out,
            )
            if veto_reason is not None:
                return dict(candidate_out), dict(selected_payload), str(veto_reason)
            return dict(candidate_out), dict(selected_payload), None
        except Exception as exc:
            return None, None, f"local_forecast_error: {type(exc).__name__}: {exc}"

    def _select_repair_rescue_candidate(
        self,
        *,
        confirmed_candidates: Sequence[Mapping[str, Any]],
        scout_records: Sequence[Mapping[str, Any]],
        baseline: Mapping[str, Any],
        repair_attempt: RepairAttemptState,
        controller_lane: str,
        proposed_action_kind: str,
        proposed_selected: Mapping[str, Any] | None,
        checkpoint_index: int,
        dt: float,
        time_stop: float | None,
        stay_forecast: Mapping[str, Any] | None = None,
    ) -> tuple[str | None, Mapping[str, Any] | None, str | None, Mapping[str, Any] | None]:
        del scout_records
        if self._repair_retry_admission_policy() == "strict":
            return None, None, "rescue_policy_strict", None
        if self._repair_retry_admission_policy() != "rescue_best_confirmed_append_v1":
            return None, None, "unsupported_rescue_policy", None
        if normalize_high_miss_no_admit_policy(getattr(self.cfg, "high_miss_no_admit_policy", None)) != "repair_retry":
            return None, None, "not_repair_retry", None
        if str(controller_lane) != "append":
            return None, None, "not_append_lane", None
        if not (str(proposed_action_kind) == "stay" or proposed_selected is None):
            return None, None, "strict_selection_admitted", None
        if self._repair_retry_rescue_attempt_policy() != "terminal_attempt_only":
            return None, None, "unsupported_rescue_attempt_policy", None
        if repair_attempt.max_attempts is None or int(repair_attempt.attempt_index) < int(repair_attempt.max_attempts):
            return None, None, "not_terminal_retry_attempt", None
        if time_stop is None:
            return None, None, "terminal_checkpoint_has_no_append_step", None
        min_gain_ratio = float(getattr(self.cfg, "repair_retry_rescue_min_gain_ratio", 0.0))
        candidate, reason = self._best_confirmed_candidate_for_repair_rescue(
            confirmed_candidates,
            min_gain_ratio=float(min_gain_ratio),
        )
        if candidate is None:
            return None, None, reason, None
        candidate, forecast_payload, veto_reason = self._forecast_veto_reason_for_repair_rescue(
            checkpoint_index=int(checkpoint_index),
            baseline=baseline,
            candidate=candidate,
            dt=float(dt),
            time_stop=float(time_stop),
            stay_forecast=stay_forecast,
        )
        if veto_reason is not None:
            return None, None, str(veto_reason), forecast_payload
        if candidate is None:
            return None, None, "rescue_candidate_forecast_failed", forecast_payload
        candidate_out = dict(candidate)
        candidate_out["repair_rescue_admission_reason"] = "repair_retry_rescue_best_confirmed_append_v1"
        return (
            "append_candidate",
            candidate_out,
            "repair_retry_rescue_best_confirmed_append_v1",
            forecast_payload,
        )

    def _repair_no_admit_diagnostics(
        self,
        *,
        controller_lane: str,
        repair_attempt: RepairAttemptState,
        scout_records: Sequence[Mapping[str, Any]],
        confirmed_candidates: Sequence[Mapping[str, Any]],
        proposed_action_kind: str,
        proposed_selected: Mapping[str, Any] | None,
        strict_no_admit_reason: str | None,
        forecast_veto_reason: str | None,
        no_admit_resolution: str,
        no_admit_resolution_advances_time: bool,
        high_miss_no_admit_soft_fallback: bool = False,
        soft_fallback_reason: str | None = None,
        soft_fallback_warning: str | None = None,
    ) -> dict[str, Any]:
        del proposed_selected
        best_candidate = None
        if confirmed_candidates:
            best_candidate = self._sorted_confirmed_by_gain(confirmed_candidates)[0]
        admissible_count = int(
            sum(1 for rec in confirmed_candidates if self._passes_exact_confirm_thresholds(rec))
        )
        best_gain_ratio = (
            None
            if best_candidate is None
            else self._finite_float_or_none(best_candidate.get("gain_ratio"))
        )
        best_score = (
            None
            if best_candidate is None
            else self._finite_float_or_none(best_candidate.get("confirm_score", best_candidate.get("adjusted_gain")))
        )
        reason = strict_no_admit_reason
        if reason in {None, ""}:
            if not confirmed_candidates:
                reason = "no_confirmed_candidates"
            elif str(proposed_action_kind) == "stay":
                reason = "strict_selected_stay"
            else:
                reason = "strict_no_selected_candidate"
        return {
            "controller_lane": str(controller_lane),
            "repair_attempt_index": int(repair_attempt.attempt_index),
            "repair_escalation_kind": repair_attempt.escalation_kind,
            "scout_candidate_count": int(len(scout_records)),
            "confirmed_candidate_count": int(len(confirmed_candidates)),
            "admissible_candidate_count": int(admissible_count),
            "best_candidate_label": (
                None if best_candidate is None else str(best_candidate.get("candidate_label"))
            ),
            "best_candidate_identity": (
                None
                if best_candidate is None
                else str(best_candidate.get("candidate_identity", best_candidate.get("candidate_label")))
            ),
            "best_candidate_gain_ratio": best_gain_ratio,
            "best_candidate_score": best_score,
            "strict_no_admit_reason": str(reason),
            "forecast_veto_reason": (None if forecast_veto_reason in {None, ""} else str(forecast_veto_reason)),
            "high_miss_no_admit_policy": str(getattr(self.cfg, "high_miss_no_admit_policy", HIGH_MISS_NO_ADMIT_POLICY_DEFAULT)),
            "no_admit_resolution": str(no_admit_resolution),
            "no_admit_resolution_advances_time": bool(no_admit_resolution_advances_time),
            "high_miss_no_admit_soft_fallback": bool(high_miss_no_admit_soft_fallback),
            "high_miss_no_admit_soft_fallback_reason": (
                None if soft_fallback_reason in {None, ""} else str(soft_fallback_reason)
            ),
            "soft_fallback_warning": (
                None if soft_fallback_warning in {None, ""} else str(soft_fallback_warning)
            ),
            "selection_policy": str(getattr(self.cfg, "oracle_selection_policy", "measured_gain_commit_veto")),
            "repair_retry_admission_policy": self._repair_retry_admission_policy(),
            "thresholds": {
                "gain_ratio_threshold": float(getattr(self.cfg, "gain_ratio_threshold", 0.0)),
                "append_margin_abs": float(getattr(self.cfg, "append_margin_abs", 0.0)),
                "repair_retry_rescue_min_gain_ratio": float(
                    getattr(self.cfg, "repair_retry_rescue_min_gain_ratio", 0.0)
                ),
                "miss_threshold": float(getattr(self.cfg, "miss_threshold", 0.0)),
                "miss_abs_threshold": float(getattr(self.cfg, "miss_abs_threshold", 0.0)),
                "miss_persistence_window": int(getattr(self.cfg, "miss_persistence_window", 1)),
                "miss_persistence_count": int(getattr(self.cfg, "miss_persistence_count", 1)),
                "shortlist_size": int(getattr(self._active_cfg(), "shortlist_size", 0)),
                "shortlist_fraction": float(getattr(self._active_cfg(), "shortlist_fraction", 0.0)),
                "max_probe_positions": int(getattr(self._active_cfg(), "max_probe_positions", 0)),
            },
        }

    def _exact_v1_live_confirm_score_threshold(self) -> float:
        return 0.0

    def _exact_v1_live_append_gate_failure_reason(
        self,
        rec: Mapping[str, Any],
    ) -> str | None:
        if not self._passes_exact_confirm_thresholds(rec):
            gain_ratio = rec.get("gain_ratio")
            gain_exact = rec.get("gain_exact")
            if gain_ratio is None:
                return "gain_ratio_missing"
            if gain_exact is None:
                return "gain_exact_missing"
            gain_ratio_value = float(gain_ratio)
            gain_exact_value = float(gain_exact)
            if not np.isfinite(gain_ratio_value):
                return "gain_ratio_missing"
            if not np.isfinite(gain_exact_value):
                return "gain_exact_missing"
            if float(gain_ratio_value) < float(self.cfg.gain_ratio_threshold):
                return "gain_ratio_below_threshold"
            if float(gain_exact_value) < float(self.cfg.append_margin_abs):
                return "gain_exact_below_threshold"
            return "gain_gate_failed"
        confirm_score = rec.get("confirm_score")
        if confirm_score is None:
            return "confirm_score_missing"
        confirm_score_value = float(confirm_score)
        if not np.isfinite(confirm_score_value):
            return "confirm_score_missing"
        if float(confirm_score_value) < float(self._exact_v1_live_confirm_score_threshold()):
            return "confirm_score_below_threshold"
        return None

    def _exact_v1_near_miss_confirm_thresholds(self, rec: Mapping[str, Any]) -> bool:
        gain_ratio = rec.get("gain_ratio")
        gain_exact = rec.get("gain_exact")
        if gain_ratio is None or gain_exact is None:
            return False
        gain_ratio_threshold = max(0.0, float(self.cfg.gain_ratio_threshold))
        gain_exact_threshold = max(0.0, float(self.cfg.append_margin_abs))
        ratio_floor = 0.5 * float(gain_ratio_threshold)
        exact_floor = 0.5 * float(gain_exact_threshold)
        return bool(
            float(gain_ratio) >= float(ratio_floor) - 1.0e-12
            and float(gain_exact) >= float(exact_floor) - 1.0e-12
        )

    def _exact_v1_below_floor_probe_limit(
        self,
        *,
        stay_forecast: Mapping[str, Any] | None = None,
    ) -> int:
        if int(self._exact_v1_append_lane_stall_streak) < int(
            self._exact_v1_below_floor_probe_stall_threshold(stay_forecast=stay_forecast)
        ):
            return 0
        return 1

    def _exact_v1_d_shape_outside_turn_below_floor_probe_stall_threshold(self) -> int:
        return max(
            0,
            int(
                getattr(
                    self.cfg,
                    "exact_v1_d_shape_outside_turn_below_floor_probe_stall_threshold",
                    0,
                )
            ),
        )

    def _exact_v1_below_floor_probe_stall_threshold(
        self,
        *,
        stay_forecast: Mapping[str, Any] | None = None,
    ) -> int:
        threshold = 3
        if (
            stay_forecast is not None
            and self._exact_v1_d_shape_barrier_ranking_active()
            and not self._exact_v1_d_shape_turn_window_active(stay_forecast=stay_forecast)
        ):
            override = self._exact_v1_d_shape_outside_turn_below_floor_probe_stall_threshold()
            if int(override) > 0:
                threshold = max(int(threshold), int(override))
        return int(threshold)

    def _exact_v1_repeat_reopen_mode(self) -> str:
        return str(getattr(self.cfg, "exact_v1_repeat_reopen_mode", "off")).strip().lower()

    def _exact_v1_repeat_reopen_stall_threshold(self) -> int:
        return max(1, int(self._exact_v1_below_floor_probe_stall_threshold()))

    def _exact_v1_sign_reversal_repeat_reopen_active(
        self,
        *,
        baseline: Mapping[str, Any] | None,
        time_start: float | None,
        time_stop: float | None,
    ) -> bool:
        if bool(self.allow_repeats):
            return False
        if str(self.cfg.mode) != "exact_v1":
            return False
        if self._exact_v1_repeat_reopen_mode() != "sign_reversal_window":
            return False
        if baseline is None or time_start is None or time_stop is None:
            return False
        if int(self._exact_v1_append_lane_stall_streak) < int(self._exact_v1_repeat_reopen_stall_threshold()):
            return False
        psi_current = np.asarray(baseline.get("psi", ()), dtype=complex).reshape(-1)
        theta_dot_step = np.asarray(baseline.get("theta_dot_step", ()), dtype=float).reshape(-1)
        if psi_current.size == 0 or theta_dot_step.size != int(self.current_theta.size):
            return False
        controller_density = float(
            self._primary_density_value_from_snapshot(
                self._observable_snapshot(psi_current)
            )
        )
        theta_next = np.asarray(
            self.current_theta + float(time_stop - time_start) * theta_dot_step,
            dtype=float,
        ).reshape(-1)
        psi_next = np.asarray(
            self.current_executor.prepare_state(theta_next, self.replay_context.psi_ref),
            dtype=complex,
        ).reshape(-1)
        projected_next = float(
            self._primary_density_value_from_snapshot(
                self._observable_snapshot(psi_next)
            )
        )
        if not all(np.isfinite(float(value)) for value in (controller_density, projected_next)):
            return False

        sign_eps = 5.0e-2
        gap_floor = max(5.0e-2, 2.0 * float(self.cfg.miss_threshold))

        def _sign_bucket(value: float) -> int:
            if float(value) > float(sign_eps):
                return 1
            if float(value) < -float(sign_eps):
                return -1
            return 0

        ctrl_sign = _sign_bucket(controller_density)
        projected_next_sign = _sign_bucket(projected_next)
        sign_flip_next = (
            int(ctrl_sign) != 0
            and int(projected_next_sign) != 0
            and int(ctrl_sign) != int(projected_next_sign)
        )
        return bool(
            sign_flip_next and abs(float(projected_next) - float(controller_density)) >= float(gap_floor)
        )

    def _exact_v1_below_floor_energy_safe_window(
        self,
        *,
        stay_forecast: Mapping[str, Any],
        selected_forecast: Mapping[str, Any],
    ) -> tuple[bool, str | None]:
        stay_next_energy_error = float(stay_forecast.get("abs_energy_total_error_next", float("inf")))
        if float(stay_next_energy_error) > 1.0e-2:
            allow_turn_escape = False
            allow_d_shape_escape = False
            if bool(getattr(self.cfg, "exact_v1_below_floor_energy_safe_turn_escape", False)):
                stay_turn_error = self._exact_v1_site_turn_error_total(stay_forecast)
                selected_turn_error = self._exact_v1_site_turn_error_total(selected_forecast)
                selected_next_energy_error = float(
                    selected_forecast.get("abs_energy_total_error_next", float("inf"))
                )
                if (
                    stay_turn_error is not None
                    and selected_turn_error is not None
                    and self._exact_v1_turn_escape_density_failure_active(
                        stay_forecast=stay_forecast,
                        selected_forecast=selected_forecast,
                    )
                    and float(self._forecast_tracking_score(forecast=selected_forecast))
                    < float(self._forecast_tracking_score(forecast=stay_forecast)) - 1.0e-12
                    and float(selected_turn_error) < float(stay_turn_error) - 1.0e-12
                    and float(selected_next_energy_error) <= float(stay_next_energy_error) + 1.0e-12
                ):
                    allow_turn_escape = True
            if self._exact_v1_below_floor_d_shape_escape_active(
                stay_forecast=stay_forecast,
                selected_forecast=selected_forecast,
            ):
                allow_d_shape_escape = True
            if not (allow_turn_escape or allow_d_shape_escape):
                return False, "outside_energy_safe_window"
        shape_caps: tuple[tuple[str, float, str], ...] = (
            (
                "tracking_energy_slope_abs_error_mean",
                2.5e-2,
                "fails_energy_slope_window",
            ),
            (
                "tracking_energy_curvature_abs_error_mean",
                2.5e-2,
                "fails_energy_curvature_window",
            ),
            (
                "tracking_energy_excursion_under_response_mean",
                1.0e-2,
                "fails_energy_excursion_under_window",
            ),
            (
                "tracking_energy_excursion_over_response_mean",
                1.0e-2,
                "fails_energy_excursion_over_window",
            ),
        )
        for key, tolerance, reason in shape_caps:
            if key not in stay_forecast or key not in selected_forecast:
                continue
            stay_value = float(stay_forecast.get(key, 0.0))
            selected_value = float(selected_forecast.get(key, 0.0))
            if not np.isfinite(stay_value) or not np.isfinite(selected_value):
                return False, str(reason)
            if float(selected_value) > float(stay_value) + float(tolerance):
                return False, str(reason)
        return True, None

    def _confirm_candidates(
        self,
        *,
        checkpoint_ctx: Any,
        cache: ExactCheckpointValueCache,
        geometry_memo: DerivedGeometryMemo,
        baseline: Mapping[str, Any],
        shortlist: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        confirmed: list[dict[str, Any]] = []
        for record in shortlist:
            memo_label = f"{record['candidate_label']}__pool{int(record['candidate_pool_index'])}"
            block_value, _ = geometry_memo.get_or_compute(
                DerivedGeometryKey(
                    checkpoint_id=str(checkpoint_ctx.checkpoint_id),
                    memo_family="candidate_incremental_block",
                    candidate_label=str(memo_label),
                    position_id=int(record["position_id"]),
                ),
                compute=lambda rec=record: self._compute_candidate_incremental_block(
                    checkpoint_ctx=checkpoint_ctx,
                    cache=cache,
                    geometry_memo=geometry_memo,
                    baseline=baseline,
                    candidate_term=rec["candidate_term"],
                    candidate_pool_index=int(rec["candidate_pool_index"]),
                    position_id=int(rec["position_id"]),
                ),
            )
            candidate_data = dict(block_value["candidate_data"])
            gain_exact = float(block_value["gain_exact"])
            gain_ratio = float(block_value["gain_ratio"])
            theta_dot_aug_existing = np.asarray(block_value["theta_dot_aug_existing"], dtype=float).reshape(-1)
            theta_dot_aug = np.asarray(block_value["theta_dot_aug"], dtype=float).reshape(-1)
            eta_dot = np.asarray(block_value["eta_dot"], dtype=float).reshape(-1)
            runtime_pos = int(candidate_data["runtime_insert_position"])
            directional_change_l2 = _overlap_l2(theta_dot_aug, self._previous_theta_dot)
            confirm_payload = self._confirm_score_payload(
                baseline=baseline,
                B=np.asarray(block_value["B"], dtype=float),
                C=np.asarray(block_value["C"], dtype=float),
                q=np.asarray(block_value["q"], dtype=float).reshape(-1),
                w=np.asarray(block_value["w"], dtype=float).reshape(-1),
                gain_ratio=float(gain_ratio),
                gain_exact=float(gain_exact),
                groups_new=float(record["groups_new"]),
                directional_change_l2=directional_change_l2,
            )
            candidate_summary = CandidateProbeSummary(
                candidate_label=str(record["candidate_label"]),
                candidate_pool_index=int(record["candidate_pool_index"]),
                position_id=int(record["position_id"]),
                runtime_insert_position=int(runtime_pos),
                runtime_block_indices=list(candidate_data["runtime_block_indices"]),
                residual_overlap_l2=float(record["residual_overlap_l2"]),
                gain_exact=float(gain_exact),
                gain_ratio=float(gain_ratio),
                compile_proxy_total=float(record["compile_proxy_total"]),
                groups_new=float(record["groups_new"]),
                novelty=(None if record.get("novelty") is None else float(record["novelty"])),
                position_jump_penalty=float(record["position_jump_penalty"]),
                directional_change_l2=(None if directional_change_l2 is None else float(directional_change_l2)),
                tier_reached="confirm",
                admissible=True,
                rejection_reason=None,
                decision_metric=(
                    "compressed_whitened_confirm_gain_ratio"
                    if str(getattr(self.cfg, "confirm_score_mode", "exact_gain_ratio")) == "compressed_whitened_v1"
                    else "gain_ratio"
                ),
                oracle_estimate_kind=self._oracle_estimate_kind(),
                temporal_prior_bonus=float(record.get("temporal_prior_bonus", 0.0)),
            )
            confirmed.append(
                {
                    **dict(record),
                    "gain_exact": float(gain_exact),
                    "gain_ratio": float(gain_ratio),
                    **dict(confirm_payload),
                    "theta_dot_aug": theta_dot_aug,
                    "theta_dot_aug_existing": theta_dot_aug_existing,
                    "eta_dot": eta_dot,
                    "candidate_summary": candidate_summary,
                }
            )
        return confirmed

    def _compute_candidate_incremental_block(
        self,
        *,
        checkpoint_ctx: Any,
        cache: ExactCheckpointValueCache,
        geometry_memo: DerivedGeometryMemo,
        baseline: Mapping[str, Any],
        candidate_term: AnsatzTerm,
        candidate_pool_index: int,
        position_id: int,
    ) -> dict[str, Any]:
        candidate_data = self._candidate_executor_data(
            checkpoint_ctx=checkpoint_ctx,
            cache=cache,
            geometry_memo=geometry_memo,
            candidate_term=candidate_term,
            candidate_pool_index=int(candidate_pool_index),
            position_id=int(position_id),
        )
        T = np.asarray(baseline["T"], dtype=complex)
        b_bar = np.asarray(baseline["b_bar"], dtype=complex)
        K_pinv = np.asarray(baseline["K_pinv"], dtype=float)
        theta_dot_step = np.asarray(baseline["theta_dot_step"], dtype=float)
        norm_b_sq = float(baseline["norm_b_sq"])
        candidate_tangents = [
            np.asarray(candidate_data["raw_tangents"][idx], dtype=complex)
            - complex(np.vdot(baseline["psi"], candidate_data["raw_tangents"][idx])) * np.asarray(baseline["psi"], dtype=complex)
            for idx in candidate_data["runtime_block_indices"]
        ]
        U = np.column_stack(candidate_tangents) if candidate_tangents else np.zeros((baseline["psi"].size, 0), dtype=complex)
        B = np.asarray(np.real(T.conj().T @ U), dtype=float)
        C = np.asarray(np.real(U.conj().T @ U), dtype=float)
        q = np.asarray(np.real(U.conj().T @ b_bar), dtype=float).reshape(-1)
        S = np.asarray(
            C
            + self._cfg_float("candidate_regularization_lambda") * np.eye(int(C.shape[0]))
            - B.T @ K_pinv @ B,
            dtype=float,
        )
        S_pinv = np.linalg.pinv(S, rcond=self._cfg_float("pinv_rcond")) if S.size else np.zeros((0, 0), dtype=float)
        w = np.asarray(q - B.T @ theta_dot_step, dtype=float).reshape(-1)
        eta_dot = np.asarray(S_pinv @ w, dtype=float).reshape(-1) if S.size else np.zeros(0, dtype=float)
        gain_exact = float(max(0.0, float(w @ eta_dot))) if w.size else 0.0
        gain_ratio = float(gain_exact / max(norm_b_sq, 1e-14))
        theta_dot_aug_existing = np.asarray(theta_dot_step - K_pinv @ B @ eta_dot, dtype=float).reshape(-1)
        runtime_pos = int(candidate_data["runtime_insert_position"])
        theta_dot_aug = np.concatenate(
            [
                theta_dot_aug_existing[:runtime_pos],
                eta_dot,
                theta_dot_aug_existing[runtime_pos:],
            ]
        )
        return {
            "candidate_data": candidate_data,
            "B": B,
            "C": C,
            "q": q,
            "S": S,
            "w": w,
            "eta_dot": eta_dot,
            "gain_exact": float(gain_exact),
            "gain_ratio": float(gain_ratio),
            "theta_dot_aug_existing": theta_dot_aug_existing,
            "theta_dot_aug": theta_dot_aug,
        }


    def _prune_no_harm_guard_reason(
        self,
        *,
        checkpoint_index: int,
        time_value: float,
        time_stop: float | None,
        baseline: Mapping[str, Any],
        reduced_baseline: Mapping[str, Any],
        reduced_state: Mapping[str, Any],
    ) -> tuple[str | None, dict[str, Any]]:
        """Verify a nominated prune over the next segment before accepting it.

        This is the code-side form of the Math 17A prune segment residual gate:
        acceptance is allowed only when the reduced scaffold is no worse than
        stay under the same one-step Euler/RK4 policy and local projective
        forecast scoring.  The check uses prepared-state observables and
        McLachlan geometry only; exact ED target/reference data is not an input.
        """
        enabled = bool(getattr(self.cfg, "prune_no_harm_guard_enabled", True))
        diagnostics: dict[str, Any] = {
            "prune_no_harm_guard_enabled": bool(enabled),
            "prune_no_harm_mode": "local_projective_same_integrator_v1",
            "prune_no_harm_uses_exact_reference": False,
        }
        if not enabled:
            diagnostics["prune_no_harm_skipped_reason"] = "disabled"
            return None, diagnostics
        if time_stop is None:
            diagnostics["prune_no_harm_skipped_reason"] = "terminal_checkpoint"
            return "prune_no_harm_unverifiable", diagnostics
        dt = float(time_stop) - float(time_value)
        if not np.isfinite(dt) or dt <= 0.0:
            diagnostics["prune_no_harm_skipped_reason"] = "nonpositive_dt"
            return "prune_no_harm_unverifiable", diagnostics
        try:
            stay_theta_next, stay_theta_dot, stay_integrator = self._integrate_theta_one_step(
                checkpoint_index=int(checkpoint_index),
                time_start=float(time_value),
                time_stop=float(time_stop),
                executor=self.current_executor,
                layout=self.current_layout,
                theta_runtime=self.current_theta,
                baseline=baseline,
                planning_audit=self._planning_audit,
                scaffold_labels=self._current_scaffold_labels(),
            )
            prune_theta_next, prune_theta_dot, prune_integrator = self._integrate_theta_one_step(
                checkpoint_index=int(checkpoint_index),
                time_start=float(time_value),
                time_stop=float(time_stop),
                executor=reduced_state["reduced_executor"],
                layout=reduced_state["reduced_layout"],
                theta_runtime=np.asarray(reduced_state["reduced_theta"], dtype=float).reshape(-1),
                baseline=reduced_baseline,
                planning_audit=reduced_state["reduced_planning_audit"],
                scaffold_labels=[str(carrier.label) for carrier in reduced_state["reduced_terms"]],
            )
            stay_forecast, _, stay_score = self._local_projective_forecast_rollout(
                checkpoint_index=int(checkpoint_index),
                time_stop=float(time_stop),
                executor=self.current_executor,
                layout=self.current_layout,
                theta_runtime_start=np.asarray(stay_theta_next, dtype=float).reshape(-1),
                theta_dot_step=np.asarray(stay_theta_dot, dtype=float).reshape(-1),
                planning_audit=self._planning_audit,
                scaffold_labels=self._current_scaffold_labels(),
                immediate_gain_ratio=float(getattr(baseline["summary"], "step_gain_ratio", 0.0)),
                anchor_summary=baseline["summary"],
                anchor_predicted_displacement=self._predicted_displacement(dt=float(dt), baseline=baseline),
            )
            prune_forecast, _, prune_score = self._local_projective_forecast_rollout(
                checkpoint_index=int(checkpoint_index),
                time_stop=float(time_stop),
                executor=reduced_state["reduced_executor"],
                layout=reduced_state["reduced_layout"],
                theta_runtime_start=np.asarray(prune_theta_next, dtype=float).reshape(-1),
                theta_dot_step=np.asarray(prune_theta_dot, dtype=float).reshape(-1),
                planning_audit=reduced_state["reduced_planning_audit"],
                scaffold_labels=[str(carrier.label) for carrier in reduced_state["reduced_terms"]],
                immediate_gain_ratio=float(getattr(reduced_baseline["summary"], "step_gain_ratio", 0.0)),
                anchor_summary=reduced_baseline["summary"],
                anchor_predicted_displacement=self._predicted_displacement(dt=float(dt), baseline=reduced_baseline),
            )
        except Exception as exc:
            diagnostics["prune_no_harm_error"] = f"{type(exc).__name__}: {exc}"
            return "prune_no_harm_verification_error", diagnostics

        def _ff(value: Any) -> float | None:
            try:
                out = float(value)
            except (TypeError, ValueError):
                return None
            return out if np.isfinite(out) else None

        stay_score_f = float(stay_score)
        prune_score_f = float(prune_score)
        score_delta = float(prune_score_f - stay_score_f)
        stay_step_residual = _ff(stay_forecast.get("step_residual_ratio_next"))
        prune_step_residual = _ff(prune_forecast.get("step_residual_ratio_next"))
        step_residual_delta = (
            None
            if stay_step_residual is None or prune_step_residual is None
            else float(prune_step_residual - stay_step_residual)
        )
        stay_rho = _ff(stay_forecast.get("rho_miss_next"))
        prune_rho = _ff(prune_forecast.get("rho_miss_next"))
        rho_delta = None if stay_rho is None or prune_rho is None else float(prune_rho - stay_rho)
        diagnostics.update(
            {
                "prune_no_harm_stay_score": float(stay_score_f),
                "prune_no_harm_prune_score": float(prune_score_f),
                "prune_no_harm_score_delta": float(score_delta),
                "prune_no_harm_score_increase_tol": float(
                    getattr(self.cfg, "prune_no_harm_score_increase_tol", 0.0)
                ),
                "prune_no_harm_stay_step_residual_ratio": stay_step_residual,
                "prune_no_harm_prune_step_residual_ratio": prune_step_residual,
                "prune_no_harm_step_residual_ratio_delta": step_residual_delta,
                "prune_no_harm_step_residual_ratio_increase_tol": float(
                    getattr(self.cfg, "prune_no_harm_step_residual_ratio_increase_tol", 1.0e-6)
                ),
                "prune_no_harm_stay_rho_miss_next": stay_rho,
                "prune_no_harm_prune_rho_miss_next": prune_rho,
                "prune_no_harm_rho_miss_next_delta": rho_delta,
                "prune_no_harm_rho_miss_increase_tol": float(
                    getattr(self.cfg, "prune_safe_miss_increase_tol", 0.0)
                ),
                "prune_no_harm_stay_integrator": str(stay_integrator.get("integrator_used", "unknown")),
                "prune_no_harm_prune_integrator": str(prune_integrator.get("integrator_used", "unknown")),
            }
        )
        if not np.isfinite(stay_score_f) or not np.isfinite(prune_score_f):
            return "prune_no_harm_nonfinite_score", diagnostics
        if score_delta > float(getattr(self.cfg, "prune_no_harm_score_increase_tol", 0.0)):
            return "prune_no_harm_score_increase_above_tol", diagnostics
        if step_residual_delta is not None and step_residual_delta > float(
            getattr(self.cfg, "prune_no_harm_step_residual_ratio_increase_tol", 1.0e-6)
        ):
            return "prune_no_harm_step_residual_increase_above_tol", diagnostics
        if rho_delta is not None and rho_delta > float(
            getattr(self.cfg, "prune_safe_miss_increase_tol", 0.0)
        ):
            return "prune_no_harm_rho_miss_increase_above_tol", diagnostics
        return None, diagnostics

    def _select_prune_action(
        self,
        *,
        checkpoint_index: int,
        time_value: float,
        time_stop: float | None,
        baseline: Mapping[str, Any],
        step_hamiltonian: StepHamiltonianArtifacts,
        prune_candidates: Sequence[Mapping[str, Any]],
    ) -> tuple[str, Mapping[str, Any] | None, Mapping[str, Any] | None, list[dict[str, Any]], str | None]:
        evaluated: list[dict[str, Any]] = [dict(row) for row in prune_candidates]
        if not evaluated:
            return "stay", None, None, [], None
        theta_tol = float(getattr(self.cfg, "prune_theta_block_tol", 0.0))
        loss_tol = float(getattr(self.cfg, "prune_loss_threshold", float("inf")))
        recoverability_mode = self._recoverability_prune_enabled()
        rejection_reason_out: str | None = None
        proposed_out: dict[str, Any] | None = None
        for idx, raw in enumerate(list(evaluated)):
            proposed = dict(raw)
            if float(proposed.get("cached_prune_loss", 0.0)) > float(loss_tol):
                self._block_cooldown[str(proposed["candidate_label"])] = int(self.cfg.prune_cooldown_steps)
                proposed["prune_accept"] = False
                proposed["post_prune_state_jump_l2"] = None
                proposed["prune_delta_rho_miss"] = None
                proposed["prune_rejection_reason"] = "cached_prune_loss_above_tol"
                evaluated[idx] = proposed
                if proposed_out is None:
                    proposed_out = dict(proposed)
                    rejection_reason_out = "prune_rejected_cached_prune_loss_above_tol"
                continue
            if (not recoverability_mode) and float(proposed.get("theta_block_norm", 0.0)) > float(theta_tol):
                self._block_cooldown[str(proposed["candidate_label"])] = int(self.cfg.prune_cooldown_steps)
                proposed["prune_accept"] = False
                proposed["post_prune_state_jump_l2"] = None
                proposed["prune_delta_rho_miss"] = None
                proposed["prune_rejection_reason"] = "theta_block_above_tol"
                evaluated[idx] = proposed
                if proposed_out is None:
                    proposed_out = dict(proposed)
                    rejection_reason_out = "prune_rejected_theta_block_above_tol"
                continue

            reduced_state = self._build_pruned_runtime_state(
                logical_index=int(proposed["position_id"]),
                baseline=baseline,
            )
            projection_diagnostics = dict(reduced_state.get("prune_projection_diagnostics", {}))
            proposed.update(projection_diagnostics)
            baseline_psi = np.asarray(baseline["psi"], dtype=complex).reshape(-1)
            reduced_psi_aligned = self._phase_aligned_state(
                target=baseline_psi,
                state=np.asarray(reduced_state["reduced_psi"], dtype=complex).reshape(-1),
            )
            state_jump_l2 = float(np.linalg.norm(reduced_psi_aligned - baseline_psi))
            proposed["post_prune_state_jump_l2"] = float(state_jump_l2)
            proposed["prune_projected_state_jump_l2"] = float(state_jump_l2)
            if projection_diagnostics.get("prune_ray_distance") is not None:
                proposed["prune_ray_distance"] = float(projection_diagnostics["prune_ray_distance"])
            if projection_diagnostics.get("prune_projection_objective") is not None:
                proposed["prune_projection_objective"] = float(projection_diagnostics["prune_projection_objective"])
            configured_state_jump_tol = float(getattr(self.cfg, "prune_state_jump_l2_tol", 0.0))
            state_jump_hard_cap = float(getattr(self.cfg, "prune_state_jump_l2_hard_cap", 1.0e-2))
            effective_state_jump_tol = (
                float(configured_state_jump_tol)
                if float(state_jump_hard_cap) <= 0.0
                else float(min(float(configured_state_jump_tol), float(state_jump_hard_cap)))
            )
            proposed["prune_state_jump_l2_effective_tol"] = float(effective_state_jump_tol)
            if float(state_jump_l2) > float(effective_state_jump_tol):
                self._block_cooldown[str(proposed["candidate_label"])] = int(self.cfg.prune_cooldown_steps)
                proposed["prune_accept"] = False
                proposed["prune_delta_rho_miss"] = None
                proposed["prune_rejection_reason"] = "state_jump_above_tol"
                evaluated[idx] = proposed
                if proposed_out is None:
                    proposed_out = dict(proposed)
                    rejection_reason_out = "prune_rejected_state_jump_above_tol"
                continue

            reduced_ctx = make_checkpoint_context(
                checkpoint_index=int(checkpoint_index),
                time_start=float(time_value),
                time_stop=(None if time_stop is None else float(time_stop)),
                scaffold_labels=[str(carrier.label) for carrier in reduced_state["reduced_terms"]],
                theta=np.asarray(reduced_state["reduced_theta"], dtype=float).reshape(-1),
                psi=np.asarray(reduced_state["reduced_psi"], dtype=complex).reshape(-1),
                logical_count=int(reduced_state["reduced_layout"].logical_parameter_count),
                runtime_count=int(reduced_state["reduced_layout"].runtime_parameter_count),
                resolved_family=str(self.replay_context.family_info.get("resolved", "unknown")),
                grouping_mode=str(self.cfg.grouping_mode),
                structure_locked=False,
            )
            reduced_cache = ExactCheckpointValueCache(
                checkpoint_id=str(reduced_ctx.checkpoint_id),
                grouping_mode=str(self.cfg.grouping_mode),
            )
            reduced_baseline = self._compute_baseline_geometry_for_runtime_state(
                checkpoint_ctx=reduced_ctx,
                cache=reduced_cache,
                executor=reduced_state["reduced_executor"],
                layout=reduced_state["reduced_layout"],
                theta_runtime=np.asarray(reduced_state["reduced_theta"], dtype=float).reshape(-1),
                planning_audit=reduced_state["reduced_planning_audit"],
                step_hamiltonian=step_hamiltonian,
            )
            delta_rho = float(reduced_baseline["summary"].rho_miss) - float(baseline["summary"].rho_miss)
            proposed["prune_delta_rho_miss"] = float(delta_rho)
            proposed["prune_differential_miss"] = float(delta_rho)
            miss_increase_tol = (
                float(getattr(self.cfg, "prune_differential_miss_tol", 1.0e-2))
                if recoverability_mode
                else float(getattr(self.cfg, "prune_safe_miss_increase_tol", 0.0))
            )
            if float(delta_rho) > float(miss_increase_tol):
                self._block_cooldown[str(proposed["candidate_label"])] = int(self.cfg.prune_cooldown_steps)
                proposed["prune_accept"] = False
                proposed["pruned_baseline"] = reduced_baseline
                proposed["prune_rejection_reason"] = (
                    "differential_miss_above_tol" if recoverability_mode else "rho_miss_increase_above_tol"
                )
                evaluated[idx] = proposed
                if proposed_out is None:
                    proposed_out = dict(proposed)
                    rejection_reason_out = (
                        "prune_rejected_differential_miss_above_tol"
                        if recoverability_mode
                        else "prune_rejected_rho_miss_increase_above_tol"
                    )
                continue

            no_harm_reason, no_harm_diagnostics = self._prune_no_harm_guard_reason(
                checkpoint_index=int(checkpoint_index),
                time_value=float(time_value),
                time_stop=time_stop,
                baseline=baseline,
                reduced_baseline=reduced_baseline,
                reduced_state=reduced_state,
            )
            proposed["prune_no_harm_diagnostics"] = dict(no_harm_diagnostics)
            proposed.update(dict(no_harm_diagnostics))
            proposed["prune_shadow_score"] = (
                None
                if no_harm_diagnostics.get("prune_no_harm_score_delta") is None
                else float(no_harm_diagnostics.get("prune_no_harm_score_delta"))
            )
            proposed["prune_shadow_mode"] = "local_projective_same_integrator_v1"
            proposed["prune_shadow_uses_exact_reference"] = False
            if no_harm_reason is not None:
                self._block_cooldown[str(proposed["candidate_label"])] = int(self.cfg.prune_cooldown_steps)
                proposed["prune_accept"] = False
                proposed["pruned_baseline"] = reduced_baseline
                proposed["reduced_state"] = reduced_state
                proposed["prune_rejection_reason"] = str(no_harm_reason)
                evaluated[idx] = proposed
                if proposed_out is None:
                    proposed_out = dict(proposed)
                    rejection_reason_out = f"prune_rejected_{no_harm_reason}"
                continue

            if recoverability_mode:
                persistence_key = self._prune_persistence_key(proposed)
                persistence_count, persistence_required, persistence_passed = self._update_prune_persistence(
                    key=str(persistence_key),
                    passed=True,
                )
                proposed["prune_persistence_key"] = str(persistence_key)
                proposed["prune_persistence_count"] = int(persistence_count)
                proposed["prune_persistence_required"] = int(persistence_required)
                proposed["prune_persistence_passed"] = bool(persistence_passed)
                if not persistence_passed:
                    proposed["prune_accept"] = False
                    proposed["pruned_baseline"] = reduced_baseline
                    proposed["reduced_state"] = reduced_state
                    proposed["prune_rejection_reason"] = "prune_persistence_pending"
                    evaluated[idx] = proposed
                    if proposed_out is None:
                        proposed_out = dict(proposed)
                        rejection_reason_out = "prune_rejected_persistence_pending"
                    continue
                self._prune_persistence_history.pop(str(persistence_key), None)
            else:
                proposed["prune_persistence_count"] = None
                proposed["prune_persistence_required"] = None
                proposed["prune_persistence_passed"] = None

            proposed["prune_accept"] = True
            proposed["pruned_baseline"] = reduced_baseline
            proposed["reduced_state"] = reduced_state
            proposed["prune_rejection_reason"] = None
            evaluated[idx] = proposed
            return "prune_coordinate", proposed, proposed, evaluated, None

        return "stay", None, proposed_out, evaluated, rejection_reason_out

    def _select_action(
        self,
        *,
        baseline: Mapping[str, Any],
        confirmed: Sequence[Mapping[str, Any]],
    ) -> tuple[str, Mapping[str, Any] | None]:
        if float(baseline["summary"].rho_miss) <= float(self.cfg.miss_threshold):
            return "stay", None
        if not confirmed:
            return "stay", None
        ordered = sorted(confirmed, key=self._confirm_rank_key)
        for record in ordered:
            if self._passes_exact_confirm_thresholds(record):
                return "append_candidate", record
        return "stay", None

    def _exact_v1_evaluate_confirmed_candidate(
        self,
        *,
        checkpoint_index: int | None,
        baseline_theta_dot: np.ndarray | Sequence[float],
        record: Mapping[str, Any],
        rank_index: int,
        dt: float,
        time_stop: float,
        stay_forecast_payload: Mapping[str, Any] | None,
        diag_enabled: bool,
        stay_diag_summary: Mapping[str, Any] | None,
        anchor_summary: BaselineGeometrySummary | None = None,
        anchor_predicted_displacement: float | None = None,
        motion: MotionSchedulerTelemetry | None = None,
    ) -> dict[str, Any]:
        del rank_index
        rejection_reason = self._exact_v1_live_append_gate_failure_reason(record)
        passes_exact_gate = self._passes_exact_confirm_thresholds(record)
        scaled_record = dict(record)
        forecast: dict[str, Any] = {}
        score = float("inf")
        admission_reason = "live_local_gates_passed"
        if rejection_reason is None:
            scaled_record, forecast = self._select_exact_v1_candidate_step_scale(
                checkpoint_index=checkpoint_index,
                baseline_theta_dot=np.asarray(baseline_theta_dot, dtype=float).reshape(-1),
                selected=record,
                dt=float(dt),
                time_stop=float(time_stop),
                anchor_summary=anchor_summary,
                anchor_predicted_displacement=anchor_predicted_displacement,
            )
            score = float(self._forecast_score_total(forecast))
            no_harm_reason, no_harm_diagnostics = self._append_no_harm_guard_reason(
                stay_forecast=stay_forecast_payload,
                selected_forecast=forecast,
                selected=scaled_record,
                motion=motion,
            )
            scaled_record["append_no_harm_diagnostics"] = dict(no_harm_diagnostics)
            scaled_record["append_no_harm_veto_reason"] = no_harm_reason
            if no_harm_reason is not None:
                rejection_reason = str(no_harm_reason)
                self._last_append_no_harm_diagnostics = dict(no_harm_diagnostics)
        admitted = rejection_reason is None
        candidate_diag_entry = (
            None
            if not diag_enabled or not admitted
            else self._exact_v1_postcross_candidate_compare_entry(
                record=scaled_record,
                forecast=forecast,
                score_total=float(score),
                stay_summary=stay_diag_summary,
                admitted=True,
                admission_reason=str(admission_reason),
                rejection_reason=None,
            )
        )
        return {
            "scaled_record": dict(scaled_record),
            "forecast": dict(forecast),
            "score": float(score),
            "admitted": bool(admitted),
            "admission_reason": str(admission_reason),
            "rejection_reason": (None if rejection_reason is None else str(rejection_reason)),
            "passes_exact_gate": bool(passes_exact_gate),
            "near_miss_gate": False,
            "below_floor_probe": False,
            "protected_horizon_admit": False,
            "protected_horizon_reason": None,
            "append_no_harm_diagnostics": dict(
                scaled_record.get("append_no_harm_diagnostics", {})
            ),
            "candidate_diag_entry": (
                None if candidate_diag_entry is None else dict(candidate_diag_entry)
            ),
        }

    def _select_action_exact_v1(
        self,
        *,
        checkpoint_index: int | None = None,
        baseline: Mapping[str, Any],
        confirmed: Sequence[Mapping[str, Any]],
        dt: float,
        time_stop: float,
        stay_forecast: Mapping[str, Any] | None = None,
        motion: MotionSchedulerTelemetry | None = None,
    ) -> tuple[str, Mapping[str, Any] | None]:
        self._last_exact_v1_selection_reason = None
        self._last_exact_v1_postcross_compare_diag = None
        self._last_append_no_harm_diagnostics = None
        diag_enabled = bool(self._exact_v1_postcross_compare_diag_enabled())
        anchor_predicted_displacement = self._predicted_displacement(
            dt=float(dt),
            baseline=baseline,
        )
        stay_forecast_payload = (
            dict(stay_forecast)
            if stay_forecast is not None
            else self._local_projective_forecast_rollout(
                checkpoint_index=checkpoint_index,
                time_stop=float(time_stop),
                executor=self.current_executor,
                layout=self.current_layout,
                theta_runtime_start=np.asarray(
                    self.current_theta + float(dt) * np.asarray(baseline["theta_dot_step"], dtype=float),
                    dtype=float,
                ).reshape(-1),
                theta_dot_step=np.asarray(baseline["theta_dot_step"], dtype=float).reshape(-1),
                planning_audit=self._planning_audit,
                scaffold_labels=self._current_scaffold_labels(),
                immediate_gain_ratio=float(getattr(baseline["summary"], "step_gain_ratio", 0.0)),
                anchor_summary=baseline["summary"],
                anchor_predicted_displacement=float(anchor_predicted_displacement),
            )[0]
        )
        stay_score = float(self._forecast_score_total(stay_forecast_payload))
        stay_diag_summary = (
            None
            if not diag_enabled
            else self._forecast_postcross_compare_summary(
                forecast=stay_forecast_payload,
                score_total=float(stay_score),
            )
        )
        if float(baseline["summary"].rho_miss) <= float(self.cfg.miss_threshold):
            self._last_exact_v1_selection_reason = "below_miss_threshold"
            if diag_enabled:
                self._last_exact_v1_postcross_compare_diag = {
                    "weight": float(self._exact_forecast_density_postcross_wrong_sign_weight()),
                    "evaluated_count": 0,
                    "admitted_count": 0,
                    "postcross_active_evaluated_count": 0,
                    "postcross_active_admitted_count": 0,
                    "postcross_active_rejected_count": 0,
                    "stay": stay_diag_summary,
                    "selected_pre_override": None,
                    "runner_up_compare": None,
                }
            return "stay", None
        if not confirmed:
            self._last_exact_v1_selection_reason = "no_confirmed_candidates"
            if diag_enabled:
                self._last_exact_v1_postcross_compare_diag = {
                    "weight": float(self._exact_forecast_density_postcross_wrong_sign_weight()),
                    "evaluated_count": 0,
                    "admitted_count": 0,
                    "postcross_active_evaluated_count": 0,
                    "postcross_active_admitted_count": 0,
                    "postcross_active_rejected_count": 0,
                    "stay": stay_diag_summary,
                    "selected_pre_override": None,
                    "runner_up_compare": None,
                }
            return "stay", None
        best_record: dict[str, Any] | None = None
        best_score: float | None = None
        best_diag_entry: dict[str, Any] | None = None
        diag_entries: list[dict[str, Any]] = []
        rejection_counts: dict[str, int] = {}
        evaluated_count = 0
        admitted_count = 0
        postcross_active_evaluated_count = 0
        postcross_active_admitted_count = 0
        for rank_index, record in enumerate(self._sorted_confirmed_by_gain(confirmed)):
            evaluation = self._exact_v1_evaluate_confirmed_candidate(
                checkpoint_index=checkpoint_index,
                baseline_theta_dot=np.asarray(baseline["theta_dot_step"], dtype=float).reshape(-1),
                record=record,
                rank_index=int(rank_index),
                dt=float(dt),
                time_stop=float(time_stop),
                stay_forecast_payload=stay_forecast_payload,
                diag_enabled=bool(diag_enabled),
                stay_diag_summary=stay_diag_summary,
                anchor_summary=baseline["summary"],
                anchor_predicted_displacement=float(anchor_predicted_displacement),
                motion=motion,
            )
            scaled_record = dict(evaluation["scaled_record"])
            score = float(evaluation["score"])
            admission_reason = str(evaluation["admission_reason"])
            rejection_reason = evaluation["rejection_reason"]
            evaluated_count += 1
            candidate_diag_entry = evaluation["candidate_diag_entry"]
            if candidate_diag_entry is not None and bool(candidate_diag_entry.get("postcross_active")):
                postcross_active_evaluated_count += 1
            if not bool(evaluation["admitted"]):
                if rejection_reason is not None:
                    rejection_counts[str(rejection_reason)] = (
                        int(rejection_counts.get(str(rejection_reason), 0)) + 1
                    )
                if candidate_diag_entry is not None:
                    diag_entries.append(dict(candidate_diag_entry))
                continue
            admitted_count += 1
            if candidate_diag_entry is not None:
                if bool(candidate_diag_entry.get("postcross_active")):
                    postcross_active_admitted_count += 1
                diag_entries.append(dict(candidate_diag_entry))
            if best_record is None or best_score is None or float(score) < float(best_score) - 1.0e-12:
                best_record = dict(scaled_record)
                best_record["forecast_payload"] = dict(evaluation["forecast"])
                best_record["exact_confirm_passed"] = bool(evaluation["passes_exact_gate"])
                best_record["exact_confirm_near_miss_admitted"] = bool(evaluation["near_miss_gate"])
                best_record["exact_confirm_below_floor_probed"] = bool(evaluation["below_floor_probe"])
                best_record["exact_v1_admission_reason"] = str(admission_reason)
                if isinstance(evaluation.get("append_no_harm_diagnostics"), Mapping):
                    best_record["append_no_harm_diagnostics"] = dict(
                        evaluation["append_no_harm_diagnostics"]
                    )
                best_score = float(score)
                best_diag_entry = (
                    None if candidate_diag_entry is None else dict(candidate_diag_entry)
                )
        if diag_enabled:
            runner_up_entry: dict[str, Any] | None = None
            if best_diag_entry is None:
                if diag_entries:
                    runner_up_entry = min(
                        diag_entries,
                        key=lambda item: (
                            float(item.get("tracking_score_total", float("inf"))),
                            str(item.get("candidate_identity", "")),
                        ),
                    )
            else:
                selected_identity = (
                    str(best_diag_entry.get("candidate_identity", "")),
                    int(best_diag_entry.get("position_id", -1)),
                )
                non_selected_entries = [
                    item
                    for item in diag_entries
                    if (
                        str(item.get("candidate_identity", "")),
                        int(item.get("position_id", -1)),
                    )
                    != selected_identity
                ]
                if non_selected_entries:
                    runner_up_entry = min(
                        non_selected_entries,
                        key=lambda item: (
                            float(item.get("tracking_score_total", float("inf"))),
                            str(item.get("candidate_identity", "")),
                        ),
                    )
            self._last_exact_v1_postcross_compare_diag = {
                "weight": float(self._exact_forecast_density_postcross_wrong_sign_weight()),
                "evaluated_count": int(evaluated_count),
                "admitted_count": int(admitted_count),
                "postcross_active_evaluated_count": int(postcross_active_evaluated_count),
                "postcross_active_admitted_count": int(postcross_active_admitted_count),
                "postcross_active_rejected_count": int(
                    max(0, int(postcross_active_evaluated_count) - int(postcross_active_admitted_count))
                ),
                "stay": stay_diag_summary,
                "selected_pre_override": best_diag_entry,
                "runner_up_compare": runner_up_entry,
            }
        if best_record is None or best_score is None:
            if rejection_counts:
                self._last_exact_v1_selection_reason = max(
                    rejection_counts.items(),
                    key=lambda item: (int(item[1]), str(item[0])),
                )[0]
            else:
                self._last_exact_v1_selection_reason = "no_admitted_candidates"
            return "stay", None
        local_override_reason = self._local_forecast_override_reason(
            stay_forecast=stay_forecast_payload,
            selected_forecast=best_record["forecast_payload"],
            selected=best_record,
            motion=motion,
        )
        if local_override_reason is not None:
            self._last_exact_v1_selection_reason = str(local_override_reason)
            return "stay", None
        self._last_exact_v1_selection_reason = str(
            best_record.get("exact_v1_admission_reason", "append_selected")
        )
        return "append_candidate", best_record

    def _sorted_confirmed_by_gain(
        self,
        confirmed: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        return sorted([dict(rec) for rec in confirmed], key=self._confirm_rank_key)

    def _oracle_confirm_limit_with_selection_policy(
        self,
        *,
        confirmed_count: int,
        refresh_pressure: str,
        motion: MotionSchedulerTelemetry,
    ) -> int:
        base_limit = self._oracle_confirm_limit_for_motion(
            confirmed_count=int(confirmed_count),
            refresh_pressure=str(refresh_pressure),
            motion=motion,
        )
        if int(confirmed_count) <= 0:
            return 0
        if str(self.cfg.oracle_selection_policy) != "measured_topk_oracle_energy":
            return int(base_limit)
        if self._oracle_base_config is None:
            return int(base_limit)
        noise_mode = str(getattr(self._oracle_base_config, "noise_mode", ""))
        if noise_mode not in {"shots", "ideal"}:
            return int(base_limit)
        floor_limit = min(int(confirmed_count), 3)
        return int(min(int(confirmed_count), max(int(base_limit), int(floor_limit))))

    def _controller_state_payload(self) -> dict[str, Any]:
        return {
            "logical_block_count": int(self.current_layout.logical_parameter_count),
            "runtime_parameter_count": int(self.current_layout.runtime_parameter_count),
            "labels": self._current_scaffold_labels(),
        }

    def debug_probe_exact_v1(
        self,
        *,
        probe_checkpoints: Sequence[int],
        force_stay_checkpoints: Sequence[int] = (),
        candidate_rank_limit: int = 4,
        baseline_variant_limit: int = 8,
        reference_payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if str(self.cfg.mode) != "exact_v1":
            raise ValueError("debug_probe_exact_v1 requires cfg.mode='exact_v1'")
        checkpoint_ids = sorted({int(x) for x in probe_checkpoints})
        if not checkpoint_ids:
            raise ValueError("probe_checkpoints must be non-empty")
        if int(checkpoint_ids[-1]) >= int(len(self.times)):
            raise ValueError("probe checkpoint out of range")
        force_stay_ids = sorted({int(x) for x in force_stay_checkpoints})
        force_stay_set = set(force_stay_ids)
        if any(int(x) < 0 or int(x) >= int(len(self.times)) for x in force_stay_set):
            raise ValueError("force_stay checkpoint out of range")
        reference_rows_by_checkpoint: dict[int, Mapping[str, Any]] = {}
        if isinstance(reference_payload, Mapping):
            for row in reference_payload.get("trajectory", []) or []:
                if isinstance(row, Mapping) and "checkpoint_index" in row:
                    reference_rows_by_checkpoint[int(row["checkpoint_index"])] = row

        def _float_equal(lhs: Any, rhs: Any, tol: float = 1.0e-9) -> bool:
            if lhs is None or rhs is None:
                return lhs is rhs
            try:
                lhs_f = float(lhs)
                rhs_f = float(rhs)
            except (TypeError, ValueError):
                return lhs == rhs
            if not np.isfinite(lhs_f) or not np.isfinite(rhs_f):
                return lhs_f == rhs_f
            return bool(abs(lhs_f - rhs_f) <= float(tol))

        def _candidate_key(item: Mapping[str, Any] | None) -> tuple[str, int] | None:
            if item is None:
                return None
            return (
                str(item.get("candidate_identity", item.get("candidate_label", ""))),
                int(item.get("position_id", -1)),
            )

        def _candidate_stage_of_death_rows(
            *,
            scout_records: Sequence[Mapping[str, Any]],
            shortlist_records: Sequence[Mapping[str, Any]],
            confirmed_records: Sequence[Mapping[str, Any]],
            evaluated_candidate_entries: Sequence[Mapping[str, Any]],
            proposed_selected_record: Mapping[str, Any] | None,
            final_selected_record: Mapping[str, Any] | None,
            final_action_kind: str,
            decision_override_reason_local: str | None,
            forced_stay_applied_local: bool,
        ) -> list[dict[str, Any]]:
            shortlist_keys = {
                key for item in shortlist_records if (key := _candidate_key(item)) is not None
            }
            confirmed_keys = {
                key for item in confirmed_records if (key := _candidate_key(item)) is not None
            }
            evaluated_by_key = {
                key: dict(item)
                for item in evaluated_candidate_entries
                if (key := _candidate_key(item)) is not None
            }
            proposed_key = _candidate_key(proposed_selected_record)
            final_key = _candidate_key(final_selected_record)
            out: list[dict[str, Any]] = []
            for raw in scout_records:
                key = _candidate_key(raw)
                if key is None:
                    continue
                evaluated = evaluated_by_key.get(key)
                stage = "not_shortlisted"
                stage_reason = None
                if key == final_key and str(final_action_kind) == "append_candidate":
                    stage = "committed"
                elif forced_stay_applied_local and key == proposed_key:
                    stage = "forced_stay_after_provisional_selection"
                elif decision_override_reason_local is not None and key == proposed_key:
                    stage = "overridden_after_provisional_selection"
                    stage_reason = str(decision_override_reason_local)
                elif key not in shortlist_keys:
                    stage = "not_shortlisted"
                elif key not in confirmed_keys:
                    stage = "not_confirmed"
                elif evaluated is None:
                    stage = "confirmed_missing_evaluation"
                elif not bool(evaluated.get("admitted")):
                    stage = "rejected_after_confirm"
                    stage_reason = (
                        None
                        if evaluated.get("rejection_reason") is None
                        else str(evaluated.get("rejection_reason"))
                    )
                elif key == proposed_key:
                    stage = "provisional_winner_not_committed"
                else:
                    stage = "admitted_not_selected"
                out.append(
                    {
                        "candidate_label": str(raw.get("candidate_label")),
                        "candidate_identity": str(
                            raw.get("candidate_identity", raw.get("candidate_label", ""))
                        ),
                        "candidate_pool_index": int(raw.get("candidate_pool_index", -1)),
                        "position_id": int(raw.get("position_id", -1)),
                        "raw_pool": True,
                        "shortlisted": bool(key in shortlist_keys),
                        "confirmed": bool(key in confirmed_keys),
                        "admitted": bool(False if evaluated is None else evaluated.get("admitted")),
                        "provisional_winner": bool(key == proposed_key),
                        "final_committed": bool(
                            key == final_key and str(final_action_kind) == "append_candidate"
                        ),
                        "stage_of_death": str(stage),
                        "stage_reason": stage_reason,
                        "scout_score": float(raw.get("scout_score", float("-inf"))),
                        "scout_gain_ratio": float(raw.get("scout_gain_ratio", 0.0)),
                        "confirm_gain_ratio": (
                            None
                            if evaluated is None
                            else float(evaluated.get("gain_ratio", raw.get("gain_ratio", 0.0)))
                        ),
                        "tracking_score_total": (
                            None
                            if evaluated is None
                            else float(evaluated.get("tracking_score_total", float("nan")))
                        ),
                        "tracking_score_delta_vs_stay": (
                            None
                            if evaluated is None
                            else (
                                None
                                if evaluated.get("tracking_score_delta_vs_stay") is None
                                else float(evaluated.get("tracking_score_delta_vs_stay"))
                            )
                        ),
                        "admission_reason": (
                            None
                            if evaluated is None or evaluated.get("admission_reason") is None
                            else str(evaluated.get("admission_reason"))
                        ),
                        "rejection_reason": (
                            None
                            if evaluated is None or evaluated.get("rejection_reason") is None
                            else str(evaluated.get("rejection_reason"))
                        ),
                    }
                )
            out.sort(
                key=lambda item: (
                    str(item.get("stage_of_death", "")),
                    (
                        float("inf")
                        if item.get("tracking_score_total") in {None, ""}
                        else float(item.get("tracking_score_total"))
                    ),
                    float(item.get("scout_score", float("-inf"))) * -1.0,
                    str(item.get("candidate_identity", "")),
                )
            )
            return out

        def _reference_anchor_stage_rows(
            *,
            reference_row: Mapping[str, Any] | None,
            candidate_stage_rows: Sequence[Mapping[str, Any]],
        ) -> list[dict[str, Any]]:
            if reference_row is None:
                return []
            anchor_labels: list[str] = []
            for key in ("candidate_label", "proposed_candidate_label"):
                raw_label = reference_row.get(key)
                if raw_label in {None, ""}:
                    continue
                label = str(raw_label)
                if label not in anchor_labels:
                    anchor_labels.append(label)
            out: list[dict[str, Any]] = []
            for label in anchor_labels:
                matches = [
                    dict(item)
                    for item in candidate_stage_rows
                    if str(item.get("candidate_label", "")) == str(label)
                ]
                out.append(
                    {
                        "anchor_label": str(label),
                        "present_in_raw_pool": bool(matches),
                        "matching_candidates": matches,
                    }
                )
            return out

        probe_results: list[dict[str, Any]] = []
        max_checkpoint = int(checkpoint_ids[-1])
        for checkpoint_index, time_value in enumerate(self.times):
            if int(checkpoint_index) > int(max_checkpoint):
                break
            time_stop = None
            if int(checkpoint_index) + 1 < int(len(self.times)):
                time_stop = float(self.times[int(checkpoint_index) + 1])
            step_sample_time = self._projection_sample_time(float(time_value), time_stop)
            step_hamiltonian = self._step_hamiltonian_artifacts(float(step_sample_time))
            psi_current = self.current_executor.prepare_state(self.current_theta, self.replay_context.psi_ref)
            checkpoint_ctx = make_checkpoint_context(
                checkpoint_index=int(checkpoint_index),
                time_start=float(time_value),
                time_stop=(None if time_stop is None else float(time_stop)),
                scaffold_labels=self._current_scaffold_labels(),
                theta=self.current_theta,
                psi=psi_current,
                logical_count=int(self.current_layout.logical_parameter_count),
                runtime_count=int(self.current_layout.runtime_parameter_count),
                resolved_family=str(self.replay_context.family_info.get("resolved", "unknown")),
                grouping_mode=str(self.cfg.grouping_mode),
                structure_locked=False,
            )
            cache = ExactCheckpointValueCache(
                checkpoint_id=str(checkpoint_ctx.checkpoint_id),
                grouping_mode=str(self.cfg.grouping_mode),
            )
            geometry_memo = DerivedGeometryMemo(
                checkpoint_id=str(checkpoint_ctx.checkpoint_id),
            )
            baseline_exact = self._baseline_geometry(
                checkpoint_ctx,
                cache,
                geometry_memo,
                step_hamiltonian=step_hamiltonian,
            )
            baseline_for_decision = baseline_exact
            degraded_reason: str | None = None
            baseline_step_scale: float | None = None
            baseline_blend_weight: float | None = None
            baseline_gain_scale: float | None = None
            baseline_proposal_kind: str | None = None
            baseline_step_forecast: dict[str, Any] | None = None
            baseline_variants: list[dict[str, Any]] = []
            dt = 0.0 if time_stop is None else float(time_stop - float(time_value))
            if (
                time_stop is not None
                and bool(self._drive_aligned_density_active)
            ):
                try:
                    (
                        scaled_theta_dot,
                        baseline_step_scale,
                        baseline_blend_weight,
                        baseline_gain_scale,
                        baseline_step_forecast,
                    ) = self._select_exact_v1_baseline_step_scale(
                        checkpoint_index=int(checkpoint_index),
                        baseline_theta_dot=np.asarray(
                            baseline_exact["theta_dot_step"], dtype=float
                        ).reshape(-1),
                        baseline=baseline_exact,
                        dt=float(dt),
                        time_stop=float(time_stop),
                        debug_variants=baseline_variants,
                    )
                    baseline_for_decision = dict(baseline_exact)
                    baseline_for_decision["theta_dot_step"] = np.asarray(
                        scaled_theta_dot, dtype=float
                    ).reshape(-1)
                    baseline_proposal_kind = (
                        None
                        if baseline_step_forecast is None
                        else baseline_step_forecast.get("baseline_proposal_kind")
                    )
                except Exception as exc:
                    degraded_reason = f"exact_baseline_step_scale_error: {type(exc).__name__}: {exc}"
            predicted_displacement = self._predicted_displacement(
                dt=float(dt),
                baseline=baseline_for_decision,
            )
            motion_telemetry = self._motion_telemetry(
                theta_dot=np.asarray(baseline_for_decision["theta_dot_step"], dtype=float).reshape(-1),
                predicted_displacement=float(predicted_displacement),
            )
            self._decrement_prune_cooldowns()
            self._record_prune_histories(baseline=baseline_for_decision)
            if degraded_reason is not None:
                controller_lane, controller_lane_reason = "stay", str(degraded_reason)
            else:
                controller_lane, controller_lane_reason = self._controller_lane(
                    time_stop=time_stop,
                    baseline=baseline_exact,
                    prune_candidates_available=False,
                    prune_reason="exact_rho_miss_below_threshold",
                )

            scout_records: list[dict[str, Any]] = []
            shortlist: list[dict[str, Any]] = []
            if time_stop is not None and str(controller_lane) == "append":
                shortlist, scout_records = self._scout_candidates_with_records(
                    checkpoint_ctx=checkpoint_ctx,
                    cache=cache,
                    geometry_memo=geometry_memo,
                    baseline=baseline_exact,
                    predicted_displacement=float(predicted_displacement),
                    shortlist_cfg=self._shortlist_cfg_for_motion(motion_telemetry),
                )
            confirmed = (
                self._confirm_candidates(
                    checkpoint_ctx=checkpoint_ctx,
                    cache=cache,
                    geometry_memo=geometry_memo,
                    baseline=baseline_exact,
                    shortlist=shortlist,
                )
                if shortlist
                else []
            )
            if time_stop is None:
                action_kind, selected = "stay", None
            else:
                action_kind, selected = self._select_action_exact_v1(
                    checkpoint_index=int(checkpoint_index),
                    baseline=baseline_for_decision,
                    confirmed=confirmed,
                    dt=float(dt),
                    time_stop=float(time_stop),
                    stay_forecast=baseline_step_forecast,
                    motion=motion_telemetry,
                )
            proposed_action_kind = str(action_kind)
            proposed_selected = selected
            forecast_stay: dict[str, Any] | None = None
            forecast_selected: dict[str, Any] | None = None
            exact_forecast_error: str | None = None
            decision_override_reason: str | None = None
            if time_stop is not None:
                try:
                    if baseline_step_forecast is None:
                        stay_theta_forecast = np.asarray(
                            self.current_theta
                            + float(dt) * np.asarray(baseline_for_decision["theta_dot_step"], dtype=float),
                            dtype=float,
                        ).reshape(-1)
                        forecast_stay = self._exact_step_forecast(
                            time_stop=float(time_stop),
                            executor=self.current_executor,
                            theta_runtime=stay_theta_forecast,
                        )
                    else:
                        forecast_stay = dict(baseline_step_forecast)
                    if str(proposed_action_kind) == "append_candidate" and proposed_selected is not None:
                        proposed_selected, forecast_selected = self._select_exact_v1_candidate_step_scale(
                            checkpoint_index=int(checkpoint_index),
                            baseline_theta_dot=np.asarray(
                                baseline_for_decision["theta_dot_step"], dtype=float
                            ).reshape(-1),
                            selected=proposed_selected,
                            dt=float(dt),
                            time_stop=float(time_stop),
                            anchor_summary=baseline_for_decision["summary"],
                            anchor_predicted_displacement=float(
                                self._predicted_displacement(dt=float(dt), baseline=baseline_for_decision)
                            ),
                            motion=motion_telemetry,
                        )
                except Exception as exc:
                    exact_forecast_error = f"{type(exc).__name__}: {exc}"
                    forecast_stay = None
                    forecast_selected = None
            if (
                decision_override_reason is None
                and forecast_stay is not None
                and forecast_selected is not None
            ):
                forecast_override_reason = self._local_forecast_override_reason(
                    stay_forecast=forecast_stay,
                    selected_forecast=forecast_selected,
                    selected=proposed_selected,
                    motion=motion_telemetry,
                )
                if forecast_override_reason is not None:
                    decision_override_reason = str(forecast_override_reason)
            if decision_override_reason is not None:
                action_kind, selected = "stay", None
            else:
                action_kind, selected = str(proposed_action_kind), proposed_selected

            forced_stay_applied = False
            if int(checkpoint_index) in force_stay_set and str(action_kind) == "append_candidate":
                forced_stay_applied = True
                action_kind, selected = "stay", None

            if int(checkpoint_index) in checkpoint_ids:
                stay_summary = (
                    None
                    if forecast_stay is None
                    else self._forecast_postcross_compare_summary(forecast=forecast_stay)
                )
                evaluated_candidate_entries: list[dict[str, Any]] = []
                candidate_entries: list[dict[str, Any]] = []
                if time_stop is not None:
                    for rank_index, record in enumerate(self._sorted_confirmed_by_gain(confirmed)):
                        evaluation = self._exact_v1_evaluate_confirmed_candidate(
                            checkpoint_index=int(checkpoint_index),
                            baseline_theta_dot=np.asarray(
                                baseline_for_decision["theta_dot_step"], dtype=float
                            ).reshape(-1),
                            record=record,
                            rank_index=int(rank_index),
                            dt=float(dt),
                            time_stop=float(time_stop),
                            stay_forecast_payload=forecast_stay,
                            diag_enabled=True,
                            stay_diag_summary=stay_summary,
                            anchor_summary=baseline_for_decision["summary"],
                            anchor_predicted_displacement=float(
                                self._predicted_displacement(dt=float(dt), baseline=baseline_for_decision)
                            ),
                        )
                        entry = {
                            "rank": int(rank_index),
                            "admitted": bool(evaluation["admitted"]),
                            "admission_reason": str(evaluation["admission_reason"]),
                            "rejection_reason": evaluation["rejection_reason"],
                            "passes_exact_gate": bool(evaluation["passes_exact_gate"]),
                            "near_miss_gate": bool(evaluation["near_miss_gate"]),
                            "below_floor_probe": bool(evaluation["below_floor_probe"]),
                            "protected_horizon_admit": bool(evaluation["protected_horizon_admit"]),
                            "append_no_harm_diagnostics": dict(
                                evaluation.get("append_no_harm_diagnostics", {})
                            ),
                        }
                        candidate_diag_entry = evaluation["candidate_diag_entry"]
                        if candidate_diag_entry is not None:
                            entry.update(candidate_diag_entry)
                        evaluated_candidate_entries.append(dict(entry))
                        if int(rank_index) < max(0, int(candidate_rank_limit)):
                            candidate_entries.append(entry)
                chosen_baseline_summary = (
                    None
                    if forecast_stay is None
                    else {
                        "step_scale": (
                            None if baseline_step_scale is None else float(baseline_step_scale)
                        ),
                        "blend_weight": (
                            None if baseline_blend_weight is None else float(baseline_blend_weight)
                        ),
                        "gain_scale": (
                            None if baseline_gain_scale is None else float(baseline_gain_scale)
                        ),
                        "proposal_kind": (
                            None if baseline_proposal_kind is None else str(baseline_proposal_kind)
                        ),
                        **self._forecast_postcross_compare_summary(forecast=forecast_stay),
                    }
                )
                baseline_variants_sorted = sorted(
                    [dict(item) for item in baseline_variants],
                    key=lambda item: (
                        float(item.get("tracking_score_total", float("inf"))),
                        float(item.get("gain_scale", 1.0)),
                        float(item.get("step_scale", float("inf"))),
                    ),
                )[: max(0, int(baseline_variant_limit))]
                reference_row = reference_rows_by_checkpoint.get(int(checkpoint_index))
                reference_parity = None
                if reference_row is not None:
                    fields = {
                        "proposed_action_kind": (
                            reference_row.get("proposed_action_kind"),
                            str(proposed_action_kind),
                        ),
                        "exact_v1_selection_reason": (
                            reference_row.get("exact_v1_selection_reason"),
                            getattr(self, "_last_exact_v1_selection_reason", None),
                        ),
                        "candidate_label": (
                            reference_row.get("candidate_label"),
                            (None if selected is None else str(selected.get("candidate_label"))),
                        ),
                        "proposed_candidate_label": (
                            reference_row.get("proposed_candidate_label"),
                            (
                                None
                                if proposed_selected is None
                                else str(proposed_selected.get("candidate_label"))
                            ),
                        ),
                        "baseline_step_scale": (
                            reference_row.get("baseline_step_scale"),
                            baseline_step_scale,
                        ),
                        "selected_step_scale": (
                            reference_row.get("selected_step_scale"),
                            (
                                None
                                if proposed_selected is None
                                or proposed_selected.get("candidate_step_scale") is None
                                else float(proposed_selected["candidate_step_scale"])
                            ),
                        ),
                        "decision_override_reason": (
                            reference_row.get("decision_override_reason"),
                            decision_override_reason,
                        ),
                    }
                    comparisons: dict[str, Any] = {}
                    for field_name, (expected, actual) in fields.items():
                        matches = (
                            _float_equal(expected, actual)
                            if field_name in {"baseline_step_scale", "selected_step_scale"}
                            else expected == actual
                        )
                        comparisons[str(field_name)] = {
                            "expected": expected,
                            "actual": actual,
                            "matches": bool(matches),
                        }
                    reference_parity = {
                        "reference_checkpoint_found": True,
                        "all_match": bool(all(item["matches"] for item in comparisons.values())),
                        "fields": comparisons,
                    }
                candidate_stage_rows = _candidate_stage_of_death_rows(
                    scout_records=scout_records,
                    shortlist_records=shortlist,
                    confirmed_records=confirmed,
                    evaluated_candidate_entries=evaluated_candidate_entries,
                    proposed_selected_record=proposed_selected,
                    final_selected_record=selected,
                    final_action_kind=str(action_kind),
                    decision_override_reason_local=decision_override_reason,
                    forced_stay_applied_local=bool(forced_stay_applied),
                )
                probe_results.append(
                    {
                        "checkpoint_index": int(checkpoint_index),
                        "time": float(time_value),
                        "physical_time": float(step_hamiltonian.physical_time),
                        "controller_lane": str(controller_lane),
                        "controller_lane_reason": str(controller_lane_reason),
                        "rho_miss": float(baseline_for_decision["summary"].rho_miss),
                        "logical_block_count": int(self.current_layout.logical_parameter_count),
                        "runtime_parameter_count": int(self.current_layout.runtime_parameter_count),
                        "guarded_commit_surface_mode": self._exact_v1_guarded_commit_surface_mode(),
                        "raw_candidate_count": int(len(scout_records)),
                        "shortlist_count": int(len(shortlist)),
                        "confirmed_count": int(len(confirmed)),
                        "candidate_pool_diagnostics": {
                            **dict(self._last_candidate_pool_diagnostics),
                            "raw_scout_record_count": int(len(scout_records)),
                            "shortlisted_candidate_count": int(len(shortlist)),
                            "confirmed_candidate_count": int(len(confirmed)),
                        },
                        "selection_reason": getattr(self, "_last_exact_v1_selection_reason", None),
                        "proposed_action_kind": str(proposed_action_kind),
                        "final_action_kind": str(action_kind),
                        "forced_stay_applied": bool(forced_stay_applied),
                        "decision_override_reason": decision_override_reason,
                        "exact_forecast_error": exact_forecast_error,
                        "stay": stay_summary,
                        "baseline_chosen": chosen_baseline_summary,
                        "baseline_variants": baseline_variants_sorted,
                        "candidates": candidate_entries,
                        "candidate_stage_of_death": candidate_stage_rows,
                        "reference_anchor_stage_of_death": _reference_anchor_stage_rows(
                            reference_row=reference_row,
                            candidate_stage_rows=candidate_stage_rows,
                        ),
                        "reference_parity": reference_parity,
                    }
                )

            if str(self.cfg.mode) == "exact_v1":
                if str(controller_lane) == "append" and str(proposed_action_kind) == "stay":
                    self._exact_v1_append_lane_stall_streak = int(self._exact_v1_append_lane_stall_streak) + 1
                else:
                    self._exact_v1_append_lane_stall_streak = 0

            if str(action_kind) == "append_candidate" and selected is not None:
                candidate_data = dict(selected["candidate_data"])
                self.current_terms = list(candidate_data["aug_terms"])
                self.current_layout = candidate_data["aug_layout"]
                self.current_executor = candidate_data["aug_executor"]
                self.current_theta = np.asarray(
                    candidate_data["theta_aug"]
                    + float(dt) * np.asarray(selected["theta_dot_aug"], dtype=float),
                    dtype=float,
                ).reshape(-1)
                self._append_counter += 1
                selected_position_id = int(selected["position_id"])
                self._previous_append_position = int(selected_position_id)
                self._planning_audit.commit(planning_group_keys_for_term(selected["candidate_term"]))
                appended_carrier = selected["candidate_data"].get("candidate_carrier")
                appended_label = str(
                    selected["candidate_label"]
                    if appended_carrier is None
                    else getattr(appended_carrier, "label", selected["candidate_label"])
                )
                self._block_birth_checkpoint[appended_label] = int(checkpoint_index)
                self._block_cooldown[appended_label] = 0
                self._block_burden[appended_label] = float(selected["candidate_summary"].compile_proxy_total)
                self._block_origin[appended_label] = "append"
                self._block_motion_history.setdefault(appended_label, [])
                self._block_fit_history.setdefault(appended_label, [])
                self._record_theta_dot_history(
                    np.asarray(selected["theta_dot_aug"], dtype=float).reshape(-1)
                )
            else:
                self.current_theta = np.asarray(
                    self.current_theta
                    + float(dt) * np.asarray(baseline_for_decision["theta_dot_step"], dtype=float),
                    dtype=float,
                ).reshape(-1)
                self._record_theta_dot_history(
                    np.asarray(baseline_for_decision["theta_dot_step"], dtype=float).reshape(-1)
                )
            self._set_previous_block_theta_snapshot()

        return {
            "mode": "exact_v1_debug_probe",
            "probe_checkpoints": [int(x) for x in checkpoint_ids],
            "force_stay_checkpoints": [int(x) for x in force_stay_ids],
            "candidate_rank_limit": int(candidate_rank_limit),
            "baseline_variant_limit": int(baseline_variant_limit),
            "checkpoints": probe_results,
        }

    def _strict_qpu_hh_candidate_data(
        self,
        *,
        candidate_term: AnsatzTerm,
        candidate_pool_index: int,
        position_id: int,
    ) -> dict[str, Any]:
        unique_label = (
            f"{candidate_term.label}__pool{int(candidate_pool_index)}"
            f"__append{self._append_counter}_p{int(position_id)}"
        )
        candidate_carrier = _build_candidate_carrier(
            candidate_term,
            logical_index=int(position_id),
            unique_label=str(unique_label),
            template_layout=self.current_layout,
            candidate_pool_index=int(candidate_pool_index),
        )
        aug_terms = list(self.current_terms)
        aug_terms.insert(int(position_id), candidate_carrier)
        aug_layout = _layout_from_carriers(aug_terms, template=self.current_layout)
        runtime_pos = int(runtime_insert_position(self.current_layout, int(position_id)))
        theta_aug = _insert_theta_block(
            self.current_theta,
            runtime_position=int(runtime_pos),
            width=int(len(candidate_carrier.runtime_specs)),
        )
        return {
            "candidate_carrier": candidate_carrier,
            "aug_terms": aug_terms,
            "aug_layout": aug_layout,
            "aug_executor": self._build_executor(aug_terms, aug_layout),
            "theta_aug": theta_aug,
            "runtime_insert_position": int(runtime_pos),
            "runtime_block_indices": [
                int(x)
                for x in range(
                    int(runtime_pos),
                    int(runtime_pos + len(candidate_carrier.runtime_specs)),
                )
            ],
        }

    def _strict_qpu_hh_shortlist(
        self,
        *,
        checkpoint_ctx: Any,
        predicted_displacement: float,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        del checkpoint_ctx
        raw_append_family_pool = getattr(self.replay_context, "append_family_pool", None)
        append_pool_terms = list(
            self.replay_context.family_pool
            if raw_append_family_pool is None
            else raw_append_family_pool
        )
        append_info = dict(
            getattr(self.replay_context, "append_family_info", None)
            or {
                "requested": "match_replay",
                "resolved": self.replay_context.family_info.get("resolved", "unknown"),
                "resolution_source": "replay_family",
                "fallback_used": False,
                "uses_replay_pool": True,
            }
        )
        raw_append_pool_meta = getattr(self.replay_context, "append_pool_meta", None)
        append_meta = dict(
            self.replay_context.pool_meta if raw_append_pool_meta is None else raw_append_pool_meta
        )
        current_source_labels = self._current_source_labels()
        records: list[dict[str, Any]] = []
        repeated_count = 0
        for candidate_pool_index, candidate_term in enumerate(append_pool_terms):
            source_label = str(candidate_term.label)
            if source_label in current_source_labels and not bool(self.allow_repeats):
                repeated_count += 1
                continue
            for position_id in self._candidate_positions():
                candidate_data = self._strict_qpu_hh_candidate_data(
                    candidate_term=candidate_term,
                    candidate_pool_index=int(candidate_pool_index),
                    position_id=int(position_id),
                )
                planning_stats = planning_stats_for_term(candidate_term, self._planning_audit)
                compile_est = self._compile_oracle.estimate(
                    candidate_term_count=max(
                        1, len(candidate_data["runtime_block_indices"])
                    ),
                    position_id=int(position_id),
                    append_position=int(self.current_layout.logical_parameter_count),
                    refit_active_count=max(
                        0,
                        int(self.current_layout.logical_parameter_count)
                        - int(position_id),
                    ),
                    candidate_term=candidate_term,
                )
                position_jump_penalty = self._position_jump_penalty(int(position_id))
                temporal_prior_bonus = float(
                    self._temporal_ledger.candidate_probe_bonus(
                        candidate_identity=(
                            f"{candidate_term.label}__pool{int(candidate_pool_index)}"
                        ),
                        position_id=int(position_id),
                        predicted_displacement=float(predicted_displacement),
                    )
                )
                strict_score = float(
                    float(temporal_prior_bonus)
                    - float(self.cfg.compile_penalty_weight) * float(compile_est.proxy_total)
                    - float(self.cfg.measurement_penalty_weight)
                    * float(planning_stats.groups_new)
                    - float(self.cfg.directional_penalty_weight)
                    * float(position_jump_penalty)
                )
                candidate_summary = CandidateProbeSummary(
                    candidate_label=str(candidate_term.label),
                    candidate_pool_index=int(candidate_pool_index),
                    position_id=int(position_id),
                    runtime_insert_position=int(candidate_data["runtime_insert_position"]),
                    runtime_block_indices=list(candidate_data["runtime_block_indices"]),
                    residual_overlap_l2=0.0,
                    gain_exact=None,
                    gain_ratio=None,
                    compile_proxy_total=float(compile_est.proxy_total),
                    groups_new=float(planning_stats.groups_new),
                    novelty=None,
                    position_jump_penalty=float(position_jump_penalty),
                    directional_change_l2=None,
                    tier_reached="scout",
                    admissible=True,
                    rejection_reason=None,
                    decision_metric="strict_structure_cost",
                    oracle_estimate_kind=self._oracle_estimate_kind(),
                    temporal_prior_bonus=float(temporal_prior_bonus),
                )
                records.append(
                    {
                        "candidate_label": str(candidate_term.label),
                        "candidate_identity": (
                            f"{candidate_term.label}__pool{int(candidate_pool_index)}"
                        ),
                        "candidate_pool_index": int(candidate_pool_index),
                        "position_id": int(position_id),
                        "runtime_insert_position": int(
                            candidate_data["runtime_insert_position"]
                        ),
                        "runtime_block_indices": list(
                            candidate_data["runtime_block_indices"]
                        ),
                        "residual_overlap_l2": 0.0,
                        "compile_proxy_total": float(compile_est.proxy_total),
                        "groups_new": float(planning_stats.groups_new),
                        "novelty": None,
                        "position_jump_penalty": float(position_jump_penalty),
                        "temporal_prior_bonus": float(temporal_prior_bonus),
                        "strict_structure_score": float(strict_score),
                        "scout_score": float(strict_score),
                        "scout_score_kind": "strict_structure_cost_minus_penalties",
                        "simple_score": float(strict_score),
                        "adjusted_gain": float(strict_score),
                        "confirm_score": float(strict_score),
                        "confirm_score_kind": "strict_structure_cost_prefilter",
                        "candidate_data": candidate_data,
                        "candidate_term": candidate_term,
                        "candidate_summary": candidate_summary,
                    }
                )
        self._last_candidate_pool_diagnostics = {
            "replay_family_requested": str(self.replay_context.family_info.get("requested", "")),
            "replay_family_resolved": str(self.replay_context.family_info.get("resolved", "")),
            "replay_family_resolution_source": str(
                self.replay_context.family_info.get("resolution_source", "")
            ),
            "replay_family_fallback_used": bool(
                self.replay_context.family_info.get("fallback_used", False)
            ),
            "append_family_requested": str(append_info.get("requested", "match_replay")),
            "append_family_resolved": str(append_info.get("resolved", "")),
            "append_family_resolution_source": str(
                append_info.get("resolution_source", "")
            ),
            "append_family_fallback_used": bool(append_info.get("fallback_used", False)),
            "append_uses_replay_pool": bool(append_info.get("uses_replay_pool", False)),
            "family_pool_sizes": {
                "replay_family_pool_count": int(len(self.replay_context.family_pool)),
                "append_family_pool_count": int(len(append_pool_terms)),
                "replay_terms_count": int(len(self.replay_context.replay_terms)),
                "current_source_label_count": int(len(current_source_labels)),
                "available_candidate_count": int(len(records)),
                "repeated_candidate_count": int(repeated_count),
                "repeated_suppressed_count": int(repeated_count),
                "repeated_allowed_count": 0,
            },
            "candidate_pool_complete": bool(
                append_meta.get("candidate_pool_complete", True)
            ),
            "candidate_pool_incomplete_reason": append_meta.get(
                "incomplete_reason",
                None,
            ),
            "candidate_label_samples": [
                str(item["candidate_label"]) for item in records[:8]
            ],
            "current_source_label_samples": sorted(str(x) for x in current_source_labels)[:8],
            "repeat_reopen_reason": None,
            "allow_repeats": bool(self.allow_repeats),
            "strict_qpu_faithful": True,
            "strict_qpu_hh": bool(self.strict_qpu_hh),
            "strict_qpu_family": str(self._family_key),
        }
        if not records:
            return [], []
        ordered = sorted(
            records,
            key=lambda rec: (
                -float(rec.get("strict_structure_score", float("-inf"))),
                float(rec["candidate_summary"].position_jump_penalty),
                float(rec["candidate_summary"].compile_proxy_total),
                float(rec["candidate_summary"].groups_new),
                int(rec["candidate_summary"].candidate_pool_index),
                int(rec["candidate_summary"].position_id),
            ),
        )
        fraction = float(getattr(self.cfg, "shortlist_fraction", 1.0))
        fraction_limit = (
            len(ordered)
            if fraction <= 0.0 or fraction >= 1.0
            else max(1, int(np.ceil(float(len(ordered)) * float(fraction))))
        )
        size_limit = max(1, int(getattr(self.cfg, "shortlist_size", 1)))
        limit = min(len(ordered), max(1, min(int(size_limit), int(fraction_limit))))
        return [dict(item) for item in ordered[:limit]], [dict(item) for item in records]

    def _strict_qpu_hh_summary(
        self,
        *,
        strict_fail_closed: bool,
        strict_fail_closed_reason: str | None,
        early_stop_reason: str | None,
        status: str | None = None,
    ) -> dict[str, Any]:
        physical_rows = physical_trajectory_rows(self._trajectory)
        final_row = physical_rows[-1] if physical_rows else {}
        full_horizon_fields = full_horizon_completion_fields(
            self._trajectory,
            expected_t_final=float(self.times[-1]) if len(self.times) else 0.0,
            expected_row_count=int(len(self.times)),
            early_stop_reason=early_stop_reason,
            stable_early_stop_accepted=is_successful_stable_early_stop_reason(early_stop_reason),
        )
        raw_exact_decision_checkpoints = int(
            sum(1 for row in self._ledger if str(row.get("decision_backend")) == "exact")
        )
        raw_oracle_decision_checkpoints = int(
            sum(1 for row in self._ledger if str(row.get("decision_backend")) == "oracle")
        )
        raw_ideal_observable_decision_checkpoints = int(
            sum(1 for row in self._ledger if str(row.get("decision_backend")) == "ideal_observable")
        )
        oracle_attempted_checkpoints = int(
            sum(1 for row in self._ledger if bool(row.get("oracle_attempted", False)))
        )
        append_count = int(
            sum(1 for row in self._ledger if str(row.get("action_kind")) == "append_candidate")
        )
        stay_count = int(
            sum(1 for row in self._ledger if str(row.get("action_kind")) == "stay")
        )
        integrator_used_values = [
            str(row.get("integrator_used"))
            for row in self._ledger
            if row.get("integrator_used", None) not in {None, ""}
        ]
        contract_report = strict_qpu_faithful_decision_contract(
            summary={
                "reference_mode": self._reference_mode(),
                "reference_enabled": False,
                "exact_decision_checkpoints": int(raw_exact_decision_checkpoints),
                "oracle_decision_checkpoints": int(raw_oracle_decision_checkpoints),
                "ideal_observable_decision_checkpoints": int(raw_ideal_observable_decision_checkpoints),
            },
            reference={
                "reference_mode": self._reference_mode(),
                "reference_enabled": False,
            },
            decision_rows=self._ledger,
        )
        contract_violations = [str(item) for item in contract_report.get("violations", [])]
        contract_passed = bool(contract_report.get("passed", False))
        exact_decision_checkpoints = int(contract_report.get("exact_decision_checkpoints", 0))
        oracle_decision_checkpoints = int(contract_report.get("oracle_decision_checkpoints", 0))
        ideal_observable_decision_checkpoints = int(
            contract_report.get("ideal_observable_decision_checkpoints", 0)
        )
        contract_fail_closed = bool(not contract_passed)
        effective_strict_fail_closed = bool(strict_fail_closed or contract_fail_closed)
        effective_strict_fail_closed_reason = strict_fail_closed_reason
        if contract_fail_closed and effective_strict_fail_closed_reason in {None, ""}:
            effective_strict_fail_closed_reason = (
                "strict_decision_contract_violation: " + "; ".join(contract_violations)
            )
        effective_status = str(status) if status is not None else None
        if contract_fail_closed:
            effective_status = "strict_fail_closed"
        passed = bool(not effective_strict_fail_closed and contract_passed)
        executed_backends = (
            sorted({str(row.get("decision_backend", "oracle")) for row in self._ledger})
            if self._ledger
            else []
        )
        decision_data_flow_counts = dict(
            contract_report.get("decision_data_flow_counts", {})
        )
        executed_data_flows = sorted(str(key) for key in decision_data_flow_counts)
        decision_data_flow = (
            "unknown"
            if not executed_data_flows
            else (executed_data_flows[0] if len(executed_data_flows) == 1 else "mixed")
        )
        return {
            "mode": str(self.cfg.mode),
            "reference_mode": self._reference_mode(),
            "reference_enabled": False,
            "controller_reference_mode": self._reference_mode(),
            "controller_reference_enabled": False,
            "controller_exact_input_mode": self._reference_mode(),
            "requested_decision_backend": (
                "ideal_observable" if str(self.cfg.mode) == "observable_v1" else "oracle"
            ),
            "decision_backend": (
                "none"
                if not executed_backends
                else (executed_backends[0] if len(executed_backends) == 1 else "mixed")
            ),
            "executed_decision_backends": list(executed_backends),
            "decision_data_flow": str(decision_data_flow),
            "uses_reference_for_decision": bool(
                contract_report.get("uses_reference_for_decision", False)
            ),
            "uses_future_exact_forecast_for_decision": bool(
                contract_report.get("uses_future_exact_forecast_for_decision", False)
            ),
            "uses_statevector_as_ideal_observable_estimator": bool(
                contract_report.get(
                    "uses_statevector_as_ideal_observable_estimator", False
                )
            ),
            "strict_measurement_oracle_certified": bool(
                contract_report.get("strict_measurement_oracle_certified", False)
            ),
            "decision_path_kind": STRICT_QPU_FAITHFUL_DECISION_PATH_KIND,
            "strict_qpu_faithful": True,
            "strict_qpu_hh": bool(self.strict_qpu_hh),
            "strict_qpu_family": str(self._family_key),
            "strict_state_prep_contract": dict(
                getattr(self, "_strict_state_prep_contract", {})
            ),
            "strict_fail_closed": bool(effective_strict_fail_closed),
            "strict_fail_closed_reason": effective_strict_fail_closed_reason,
            "qpu_faithful_decisions_expected": True,
            "qpu_faithful_decisions_passed": bool(passed),
            "strict_decision_contract_passed": bool(contract_passed),
            "strict_decision_contract_violations": list(contract_violations),
            "status": (
                str(effective_status)
                if effective_status is not None
                else ("strict_fail_closed" if effective_strict_fail_closed else "completed")
            ),
            "decision_noise_mode": (
                "ideal"
                if str(self.cfg.mode) == "observable_v1"
                else (
                    None
                    if self._oracle_base_config is None
                    else str(self._oracle_base_config.noise_mode)
                )
            ),
            "oracle_estimate_kind": (
                None if oracle_attempted_checkpoints <= 0 else self._oracle_estimate_kind()
            ),
            "ideal_observable_decision_checkpoints": int(ideal_observable_decision_checkpoints),
            "oracle_selection_policy": str(self.cfg.oracle_selection_policy),
            "confirm_score_mode": str(getattr(self.cfg, "confirm_score_mode", "exact_gain_ratio")),
            "prune_mode": str(getattr(self.cfg, "prune_mode", "off")),
            "high_miss_no_admit_policy": str(
                getattr(
                    self.cfg,
                    "high_miss_no_admit_policy",
                    HIGH_MISS_NO_ADMIT_POLICY_DEFAULT,
                )
            ),
            "integrator_policy": str(self._integrator_policy()),
            "integrator_used_values": sorted(set(integrator_used_values)),
            "integrator_euler_count": int(
                sum(1 for value in integrator_used_values if str(value) == "euler")
            ),
            "integrator_rk4_count": int(
                sum(1 for value in integrator_used_values if str(value) == "rk4")
            ),
            "append_no_harm_guard_enabled": False,
            "append_count": int(append_count),
            "prune_count": 0,
            "repair_count": 0,
            "stay_count": int(stay_count),
            **trajectory_repair_counts(self._trajectory),
            **high_miss_no_admit_soft_fallback_counts(self._ledger),
            **high_miss_no_admit_diagnostic_counts(self._ledger),
            **full_horizon_fields,
            "exact_decision_checkpoints": int(exact_decision_checkpoints),
            "oracle_decision_checkpoints": int(oracle_decision_checkpoints),
            "ideal_observable_decision_checkpoints": int(ideal_observable_decision_checkpoints),
            "oracle_attempted_checkpoints": int(oracle_attempted_checkpoints),
            "degraded_checkpoints": int(
                sum(1 for row in self._ledger if row.get("degraded_reason") not in {None, ""})
            ),
            "final_logical_block_count": int(self.current_layout.logical_parameter_count),
            "final_runtime_parameter_count": int(self.current_layout.runtime_parameter_count),
            "final_fidelity_exact": None,
            "final_abs_energy_total_error": None,
            "final_staggered": self._finite_float_or_none(final_row.get("staggered", None)),
            "final_staggered_exact": None,
            "final_abs_staggered_error": None,
            "final_doublon": self._finite_float_or_none(final_row.get("doublon", None)),
            "final_doublon_exact": None,
            "final_abs_doublon_error": None,
            "final_site_occupations": list(final_row.get("site_occupations", [])),
            "final_site_occupations_exact": None,
            "final_site_occupations_abs_error_max": None,
            **summary_fields_from_row(final_row),
            "planning_audit": dict(self._planning_audit.summary()),
            "temporal_measurement_ledger": dict(self._temporal_ledger.summary()),
            "early_stop_reason": early_stop_reason,
            "early_stop_checkpoint_index": (
                None if not self._ledger else int(self._ledger[-1].get("checkpoint_index", 0))
            )
            if early_stop_reason is not None
            else None,
            "early_stop_time": (
                None if not self._ledger else float(self._ledger[-1].get("time", 0.0))
            )
            if early_stop_reason is not None
            else None,
        }

    def _strict_qpu_hh_reference(self) -> dict[str, Any]:
        return {
            "reference_mode": self._reference_mode(),
            "reference_enabled": False,
            "controller_exact_input_mode": self._reference_mode(),
            "uses_reference_for_decision": False,
            "uses_future_exact_forecast_for_decision": False,
            "kind": None,
            "initial_state": "stage_result.psi_final",
            "times": [float(x) for x in self.times.tolist()],
            "drive_profile": None if self._drive_profile is None else dict(self._drive_profile),
            "reference_method": None,
            "reference_steps_multiplier": None,
            "projection_time_sampling": (
                None
                if self._drive_config is None
                else str(self._drive_config.drive_time_sampling)
            ),
            "geometry_sample_time_policy": "measurement_only",
        }

    def _run_strict_qpu_hh(self, *, checkpoint_observer: Any | None = None) -> ControllerRunArtifacts:
        if checkpoint_observer is not None:
            raise ValueError("strict_qpu_faithful forbids checkpoint_observer")
        self._run_wallclock_start = time.perf_counter()
        early_stop_reason: str | None = None

        def _finish(
            *,
            strict_fail_closed: bool,
            strict_fail_closed_reason: str | None,
            status: str | None = None,
        ) -> ControllerRunArtifacts:
            summary = self._strict_qpu_hh_summary(
                strict_fail_closed=bool(strict_fail_closed),
                strict_fail_closed_reason=strict_fail_closed_reason,
                early_stop_reason=early_stop_reason,
                status=status,
            )
            reference = self._strict_qpu_hh_reference()
            final_status = str(summary.get("status", "completed"))
            self._write_progress(
                stage="run_complete",
                force=True,
                status=final_status,
                summary=summary,
            )
            self._write_partial_payload(
                status=final_status,
                stage="run_complete",
                summary=summary,
            )
            return ControllerRunArtifacts(
                trajectory=[dict(row) for row in self._trajectory],
                ledger=[dict(row) for row in self._ledger],
                summary=summary,
                reference=reference,
            )

        try:
            self._write_progress(
                stage="run_start",
                force=True,
                strict_qpu_faithful=True,
                strict_qpu_hh=bool(self.strict_qpu_hh),
                strict_qpu_family=str(self._family_key),
            )
            self._write_partial_payload(stage="run_start")
            for checkpoint_index in range(int(len(self.times))):
                time_value = float(self.times[int(checkpoint_index)])
                time_stop = (
                    None
                    if int(checkpoint_index) + 1 >= int(len(self.times))
                    else float(self.times[int(checkpoint_index) + 1])
                )
                dt = 0.0 if time_stop is None else float(time_stop - time_value)
                step_sample_time = self._projection_sample_time(float(time_value), time_stop)
                step_hamiltonian = self._step_hamiltonian_artifacts(float(step_sample_time))
                checkpoint_ctx = make_measurement_checkpoint_context(
                    checkpoint_index=int(checkpoint_index),
                    time_start=float(time_value),
                    time_stop=(None if time_stop is None else float(time_stop)),
                    scaffold_labels=self._current_scaffold_labels(),
                    theta=self.current_theta,
                    logical_count=int(self.current_layout.logical_parameter_count),
                    runtime_count=int(self.current_layout.runtime_parameter_count),
                    resolved_family=str(
                        self.replay_context.family_info.get("resolved", "unknown")
                    ),
                    grouping_mode=str(self.cfg.grouping_mode),
                    structure_locked=False,
                )
                cache = ExactCheckpointValueCache(
                    checkpoint_id=str(checkpoint_ctx.checkpoint_id),
                    grouping_mode=str(self.cfg.grouping_mode),
                )
                geometry_memo = DerivedGeometryMemo(
                    checkpoint_id=str(checkpoint_ctx.checkpoint_id),
                )
                oracle_cache = OracleCheckpointValueCache(
                    checkpoint_id=str(checkpoint_ctx.checkpoint_id),
                )
                raw_group_pool = (
                    BackendScheduledRawGroupPool(checkpoint_id=str(checkpoint_ctx.checkpoint_id))
                    if self._oracle_base_config is not None
                    and bool(controller_oracle_supports_raw_group_sampling(self._oracle_base_config))
                    else None
                )
                oracle_budget_scale = 1.0
                try:
                    baseline = self._oracle_measured_baseline_geometry(
                        checkpoint_ctx=checkpoint_ctx,
                        cache=cache,
                        geometry_memo=geometry_memo,
                        raw_group_pool=raw_group_pool,
                        h_poly_step=step_hamiltonian.h_poly,
                        tier_name="confirm",
                        budget_scale=float(oracle_budget_scale),
                    )
                except Exception as exc:
                    reason = f"measured_baseline_error: {type(exc).__name__}: {exc}"
                    early_stop_reason = f"strict_fail_closed:{reason}"
                    return _finish(
                        strict_fail_closed=True,
                        strict_fail_closed_reason=reason,
                        status="strict_fail_closed",
                    )

                predicted_displacement = self._predicted_displacement(
                    dt=float(dt),
                    baseline=baseline,
                )
                motion_telemetry = self._motion_telemetry(
                    theta_dot=np.asarray(baseline["theta_dot_step"], dtype=float).reshape(-1),
                    predicted_displacement=float(predicted_displacement),
                )
                self._record_high_miss_history(baseline=baseline)
                base_refresh_pressure = self._temporal_ledger.refresh_pressure(
                    predicted_displacement=float(predicted_displacement),
                    rho_miss=float(baseline["summary"].rho_miss),
                    condition_number=float(baseline["summary"].condition_number),
                )
                refresh_pressure = self._effective_refresh_pressure(
                    base_refresh_pressure=str(base_refresh_pressure),
                    motion=motion_telemetry,
                )
                oracle_budget_scale = self._oracle_budget_scale_for_motion(
                    refresh_pressure=str(refresh_pressure),
                    motion=motion_telemetry,
                )
                oracle_confirm_limit = 0
                shortlist: list[dict[str, Any]] = []
                scout_records: list[dict[str, Any]] = []
                confirmed: list[dict[str, Any]] = []
                selected: Mapping[str, Any] | None = None
                action_kind = "stay"
                proposed_action_kind = "stay"
                controller_lane = "stay"
                controller_lane_reason = "measured_rho_miss_below_threshold"
                selection_metric = "measured_baseline_stay"
                degraded_reason: str | None = None
                decision_override_reason: str | None = None
                if time_stop is None:
                    controller_lane_reason = "terminal_checkpoint"
                elif float(baseline["summary"].rho_miss) > float(self.cfg.miss_threshold):
                    controller_lane = "append"
                    controller_lane_reason = "measured_rho_miss_above_threshold"
                    shortlist, scout_records = self._strict_qpu_hh_shortlist(
                        checkpoint_ctx=checkpoint_ctx,
                        predicted_displacement=float(predicted_displacement),
                    )
                    oracle_confirm_limit = self._oracle_confirm_limit_with_selection_policy(
                        confirmed_count=len(shortlist),
                        refresh_pressure=str(refresh_pressure),
                        motion=motion_telemetry,
                    )
                    measured_baseline, measured_confirmed, geometry_error = (
                        self._confirm_candidates_oracle_geometry(
                            checkpoint_ctx=checkpoint_ctx,
                            cache=cache,
                            geometry_memo=geometry_memo,
                            confirmed=shortlist,
                            raw_group_pool=raw_group_pool,
                            h_poly_step=step_hamiltonian.h_poly,
                            confirm_limit=int(oracle_confirm_limit),
                            budget_scale=float(oracle_budget_scale),
                        )
                    )
                    if geometry_error is not None or measured_baseline is None:
                        reason = str(geometry_error or "measured_geometry_missing")
                        early_stop_reason = f"strict_fail_closed:{reason}"
                        return _finish(
                            strict_fail_closed=True,
                            strict_fail_closed_reason=reason,
                            status="strict_fail_closed",
                        )
                    baseline = measured_baseline
                    confirmed = [dict(rec) for rec in measured_confirmed]
                    viable = [
                        rec
                        for rec in confirmed
                        if rec.get("gain_exact") is not None
                        and rec.get("gain_ratio") is not None
                    ]
                    selection_metric = "measured_incremental_gain_ratio"
                    for rec in sorted(viable, key=self._confirm_rank_key):
                        if (
                            float(rec.get("gain_ratio", 0.0))
                            >= float(self.cfg.gain_ratio_threshold)
                            and float(rec.get("gain_exact", 0.0))
                            >= float(self.cfg.append_margin_abs)
                        ):
                            action_kind = "append_candidate"
                            selected = dict(rec)
                            break
                    proposed_action_kind = str(action_kind)
                else:
                    confirmed = []

                oracle_commit_payload, commit_degraded_reason = self._oracle_commit_payload(
                    checkpoint_ctx=checkpoint_ctx,
                    oracle_cache=oracle_cache,
                    raw_group_pool=raw_group_pool,
                    baseline=baseline,
                    selected=selected,
                    action_kind=str(action_kind),
                    dt=float(dt),
                    oracle_observable=step_hamiltonian.oracle_observable,
                    budget_scale=float(oracle_budget_scale),
                )
                if commit_degraded_reason is not None:
                    reason = f"measured_commit_error: {commit_degraded_reason}"
                    early_stop_reason = f"strict_fail_closed:{reason}"
                    return _finish(
                        strict_fail_closed=True,
                        strict_fail_closed_reason=reason,
                        status="strict_fail_closed",
                    )
                override_reason = self._oracle_commit_override_reason(
                    motion=motion_telemetry,
                    selected=selected,
                    action_kind=str(action_kind),
                    oracle_commit_payload=oracle_commit_payload,
                    predicted_displacement=float(predicted_displacement),
                    runtime_parameter_count_before=int(
                        self.current_layout.runtime_parameter_count
                    ),
                )
                if override_reason is not None:
                    decision_override_reason = str(override_reason)
                    action_kind = "stay"
                    selected = None

                logical_before = int(self.current_layout.logical_parameter_count)
                runtime_before = int(self.current_layout.runtime_parameter_count)
                selected_candidate_label = (
                    None if selected is None else str(selected["candidate_label"])
                )
                selected_position_id = None if selected is None else int(selected["position_id"])
                selected_groups_new = 0.0 if selected is None else float(selected.get("groups_new", 0.0))
                selected_gain_ratio = 0.0 if selected is None else float(selected.get("gain_ratio", 0.0))
                rate_change_l2 = _overlap_l2(
                    np.asarray(baseline["theta_dot_step"], dtype=float),
                    self._previous_theta_dot,
                )
                integrator_diagnostics: dict[str, Any] = self._no_advance_integrator_diagnostics()
                commit_theta_dot = np.asarray(
                    baseline["theta_dot_step"], dtype=float
                ).reshape(-1)
                commit_layout = self.current_layout
                commit_theta_start = np.asarray(self.current_theta, dtype=float).reshape(-1)
                commit_scaffold_labels = self._current_scaffold_labels()
                commit_planning_audit = self._planning_audit
                strict_integrator_forced_policy: str | None = None
                strict_integrator_forced_policy_reason: str | None = None
                if str(action_kind) == "append_candidate" and selected is not None:
                    if int(self._strict_measurement_active_window_size()) > 0:
                        strict_integrator_forced_policy = "euler"
                        strict_integrator_forced_policy_reason = (
                            "measurement_active_window_append_euler"
                        )
                    candidate_data = dict(selected["candidate_data"])
                    commit_layout = candidate_data["aug_layout"]
                    commit_theta_start = np.asarray(candidate_data["theta_aug"], dtype=float).reshape(-1)
                    commit_scaffold_labels = [
                        str(term.label) for term in candidate_data["aug_terms"]
                    ]
                    commit_planning_audit = self._build_planning_audit_for_terms(
                        candidate_data["aug_terms"]
                    )
                    commit_theta_dot = np.asarray(
                        selected["theta_dot_aug"], dtype=float
                    ).reshape(-1)
                try:
                    commit_theta_next, commit_theta_dot, integrator_diagnostics = (
                        self._strict_qpu_hh_integrate_theta_one_step(
                            checkpoint_index=int(checkpoint_index),
                            time_start=float(time_value),
                            time_stop=(None if time_stop is None else float(time_stop)),
                            layout=commit_layout,
                            theta_runtime=commit_theta_start,
                            baseline=baseline,
                            planning_audit=commit_planning_audit,
                            scaffold_labels=commit_scaffold_labels,
                            tier_name="confirm",
                            budget_scale=float(oracle_budget_scale),
                            euler_theta_dot=commit_theta_dot,
                            forced_policy=strict_integrator_forced_policy,
                            forced_policy_reason=strict_integrator_forced_policy_reason,
                        )
                    )
                except Exception as exc:
                    reason = f"measured_integrator_error: {type(exc).__name__}: {exc}"
                    early_stop_reason = f"strict_fail_closed:{reason}"
                    return _finish(
                        strict_fail_closed=True,
                        strict_fail_closed_reason=reason,
                        status="strict_fail_closed",
                    )

                baseline_summary = baseline["summary"]
                drive_diagnostics = self._drive_diagnostic_payload(
                    physical_time=float(step_hamiltonian.physical_time),
                    drive_term_count=int(step_hamiltonian.drive_term_count),
                )
                theta_dot_l2 = float(np.linalg.norm(np.asarray(commit_theta_dot, dtype=float).reshape(-1)))
                theta_update_l2 = float(
                    _overlap_l2(
                        np.asarray(commit_theta_next, dtype=float).reshape(-1),
                        np.asarray(commit_theta_start, dtype=float).reshape(-1),
                    )
                    or 0.0
                )
                runtime_after_planned = int(commit_layout.runtime_parameter_count)
                try:
                    observable_telemetry = self._strict_qpu_measured_observable_telemetry(
                        checkpoint_ctx=checkpoint_ctx,
                        raw_group_pool=raw_group_pool,
                        layout=self.current_layout,
                        theta_runtime=self.current_theta,
                        tier_name="commit",
                        budget_scale=float(oracle_budget_scale),
                    )
                except Exception as exc:
                    observable_telemetry = {
                        "observable_family": str(self._family_key),
                        "observable_telemetry_supported": False,
                        "observable_telemetry_reason": f"{type(exc).__name__}: {exc}",
                        "observable_telemetry_kind": "oracle_measured",
                        "observable_telemetry_noise_mode": (
                            None
                            if self._oracle_base_config is None
                            else str(self._oracle_base_config.noise_mode)
                        ),
                        "observable_telemetry_backend_info": {},
                        "observable_telemetry_backend_info_count": 0,
                        "observable_telemetry_estimates": {},
                        "site_occupations": [],
                        "site_occupations_up": [],
                        "site_occupations_dn": [],
                        "staggered": None,
                        "doublon": None,
                        "primary_density": None,
                    }
                candidate_pool_diagnostics = dict(self._last_candidate_pool_diagnostics)
                shortlist_payload = [
                    {
                        "candidate_label": str(item["candidate_label"]),
                        "candidate_pool_index": int(item["candidate_pool_index"]),
                        "position_id": int(item["position_id"]),
                        "runtime_insert_position": int(item["runtime_insert_position"]),
                        "runtime_block_indices": list(item["runtime_block_indices"]),
                        "residual_overlap_l2": float(item.get("residual_overlap_l2", 0.0)),
                        "compile_proxy_total": float(item["compile_proxy_total"]),
                        "groups_new": float(item["groups_new"]),
                        "novelty": None,
                        "position_jump_penalty": float(item["position_jump_penalty"]),
                        "temporal_prior_bonus": float(item.get("temporal_prior_bonus", 0.0)),
                        "scout_score": float(item.get("scout_score", 0.0)),
                        "scout_score_kind": item.get("scout_score_kind", None),
                        "simple_score": float(item.get("simple_score", 0.0)),
                    }
                    for item in shortlist
                ]
                confirmed_payload = [
                    {
                        "candidate_label": str(rec["candidate_label"]),
                        "candidate_pool_index": int(rec["candidate_pool_index"]),
                        "position_id": int(rec["position_id"]),
                        "gain_exact": (
                            None if rec.get("gain_exact") is None else float(rec["gain_exact"])
                        ),
                        "gain_ratio": (
                            None if rec.get("gain_ratio") is None else float(rec["gain_ratio"])
                        ),
                        "adjusted_gain": float(rec.get("adjusted_gain", float("-inf"))),
                        "confirm_score": (
                            None
                            if rec.get("confirm_score", rec.get("adjusted_gain", None))
                            is None
                            else float(rec.get("confirm_score", rec.get("adjusted_gain")))
                        ),
                        "confirm_score_kind": rec.get(
                            "confirm_score_kind",
                            "measured_geometry_gain_ratio_minus_penalties",
                        ),
                        "confirm_error": rec.get("confirm_error", None),
                        "candidate_summary": dataclass_to_payload(rec["candidate_summary"]),
                    }
                    for rec in confirmed
                ]
                common_row = {
                    "checkpoint_index": int(checkpoint_index),
                    "time": float(time_value),
                    "physical_time": float(step_hamiltonian.physical_time),
                    "action_kind": str(action_kind),
                    "trajectory_sample_kind": "state_sample",
                    "advances_time": True,
                    "candidate_label": selected_candidate_label,
                    "proposed_action_kind": str(proposed_action_kind),
                    "proposed_candidate_label": selected_candidate_label,
                    "controller_lane": str(controller_lane),
                    "controller_lane_reason": str(controller_lane_reason),
                    "position_id": selected_position_id,
                    "requested_mode": str(self.cfg.mode),
                    "decision_backend": (
                        "ideal_observable"
                        if str(self.cfg.mode) == "observable_v1"
                        else "oracle"
                    ),
                    "decision_noise_mode": (
                        "ideal"
                        if str(self.cfg.mode) == "observable_v1"
                        else None
                        if self._oracle_base_config is None
                        else str(self._oracle_base_config.noise_mode)
                    ),
                    **decision_data_flow_fields(
                        controller_mode=str(self.cfg.mode),
                        controller_exact_input_mode=self._reference_mode(),
                        decision_backend=(
                            "ideal_observable"
                            if str(self.cfg.mode) == "observable_v1"
                            else "oracle"
                        ),
                        decision_noise_mode=(
                            "ideal"
                            if str(self.cfg.mode) == "observable_v1"
                            else None
                            if self._oracle_base_config is None
                            else str(self._oracle_base_config.noise_mode)
                        ),
                        strict_qpu_faithful=True,
                        uses_reference_for_decision=False,
                        uses_future_exact_forecast_for_decision=False,
                    ),
                    "oracle_attempted": True,
                    "oracle_decision_used": True,
                    "oracle_estimate_kind": self._oracle_estimate_kind(),
                    "selection_metric": str(selection_metric),
                    "decision_path_kind": STRICT_QPU_FAITHFUL_DECISION_PATH_KIND,
                    "strict_qpu_faithful": True,
                    "strict_qpu_hh": bool(self.strict_qpu_hh),
                    "strict_qpu_family": str(self._family_key),
                    "decision_override_reason": decision_override_reason,
                    "selection_reason": None,
                    "forecast_mode": None,
                    "forecast_error": None,
                    "exact_forecast_error": None,
                    "integrator_policy": str(integrator_diagnostics["integrator_policy"]),
                    "integrator_used": str(integrator_diagnostics["integrator_used"]),
                    "integrator_columnarity": (
                        None
                        if integrator_diagnostics.get("integrator_columnarity") is None
                        else float(integrator_diagnostics["integrator_columnarity"])
                    ),
                    "integrator_curvature": (
                        None
                        if integrator_diagnostics.get("integrator_curvature") is None
                        else float(integrator_diagnostics["integrator_curvature"])
                    ),
                    "integrator_euler_fs_error": (
                        None
                        if integrator_diagnostics.get("integrator_euler_fs_error") is None
                        else float(integrator_diagnostics["integrator_euler_fs_error"])
                    ),
                    "integrator_condition_number": (
                        None
                        if integrator_diagnostics.get("integrator_condition_number") is None
                        else float(integrator_diagnostics["integrator_condition_number"])
                    ),
                    "integrator_condition_pass": (
                        None
                        if integrator_diagnostics.get("integrator_condition_pass") is None
                        else bool(integrator_diagnostics["integrator_condition_pass"])
                    ),
                    "integrator_geometry_gate_pass": integrator_diagnostics.get(
                        "integrator_geometry_gate_pass"
                    ),
                    "integrator_euler_error_pass": integrator_diagnostics.get(
                        "integrator_euler_error_pass"
                    ),
                    "integrator_auto_policy_schema": integrator_diagnostics.get(
                        "integrator_auto_policy_schema"
                    ),
                    "integrator_auto_admit_euler": integrator_diagnostics.get(
                        "integrator_auto_admit_euler"
                    ),
                    "integrator_euler_blockers": list(
                        integrator_diagnostics.get("integrator_euler_blockers") or []
                    ),
                    "integrator_rho_miss_pass": (
                        None
                        if integrator_diagnostics.get("integrator_rho_miss_pass") is None
                        else bool(integrator_diagnostics["integrator_rho_miss_pass"])
                    ),
                    "integrator_time_fraction": integrator_diagnostics.get(
                        "integrator_time_fraction"
                    ),
                    "integrator_euler_min_time_fraction": integrator_diagnostics.get(
                        "integrator_euler_min_time_fraction"
                    ),
                    "integrator_euler_time_gate_pass": integrator_diagnostics.get(
                        "integrator_euler_time_gate_pass"
                    ),
                    "integrator_euler_observable_gate_pass": integrator_diagnostics.get(
                        "integrator_euler_observable_gate_pass"
                    ),
                    "integrator_euler_site_span": integrator_diagnostics.get(
                        "integrator_euler_site_span"
                    ),
                    "integrator_euler_primary_density_span": integrator_diagnostics.get(
                        "integrator_euler_primary_density_span"
                    ),
                    "integrator_euler_energy_span": integrator_diagnostics.get(
                        "integrator_euler_energy_span"
                    ),
                    "integrator_error": integrator_diagnostics.get("integrator_error"),
                    "integrator_forced_policy": integrator_diagnostics.get(
                        "integrator_forced_policy"
                    ),
                    "integrator_forced_policy_reason": integrator_diagnostics.get(
                        "integrator_forced_policy_reason"
                    ),
                    "temporal_refresh_pressure": str(refresh_pressure),
                    "oracle_confirm_limit": int(oracle_confirm_limit),
                    "oracle_budget_scale": float(oracle_budget_scale),
                    "rho_miss": float(baseline_summary.rho_miss),
                    "rho_real": float(baseline_summary.rho_real),
                    "rho_num": float(baseline_summary.rho_num),
                    "epsilon_proj_sq": float(baseline_summary.epsilon_proj_sq),
                    "epsilon_step_sq": float(baseline_summary.epsilon_step_sq),
                    "theta_dot_l2": float(theta_dot_l2),
                    "theta_update_l2": float(theta_update_l2),
                    "energy_total": float(baseline_summary.energy),
                    "energy_total_controller": float(baseline_summary.energy),
                    "energy_total_exact": None,
                    "abs_energy_total_error": None,
                    "fidelity_exact": None,
                    "fidelity_initial_controller": None,
                    "fidelity_initial_exact": None,
                    "primary_density_mode": str(
                        self._exact_forecast_primary_density_target_mode()
                    ),
                    "primary_density": observable_telemetry.get("primary_density", None),
                    "primary_density_exact": None,
                    "abs_primary_density_error": None,
                    "observable_family": str(observable_telemetry.get("observable_family", self._family_key)),
                    "observable_telemetry_kind": observable_telemetry.get(
                        "observable_telemetry_kind", None
                    ),
                    "observable_telemetry_supported": bool(
                        observable_telemetry.get("observable_telemetry_supported", False)
                    ),
                    "observable_telemetry_reason": observable_telemetry.get(
                        "observable_telemetry_reason", None
                    ),
                    "observable_telemetry_noise_mode": observable_telemetry.get(
                        "observable_telemetry_noise_mode", None
                    ),
                    "observable_telemetry_primary_density_mode": observable_telemetry.get(
                        "observable_telemetry_primary_density_mode", None
                    ),
                    "observable_telemetry_spec_count": observable_telemetry.get(
                        "observable_telemetry_spec_count", None
                    ),
                    "observable_telemetry_max_terms": observable_telemetry.get(
                        "observable_telemetry_max_terms", None
                    ),
                    "observable_telemetry_backend_info": dict(
                        observable_telemetry.get("observable_telemetry_backend_info", {})
                    ),
                    "observable_telemetry_backend_info_count": int(
                        observable_telemetry.get("observable_telemetry_backend_info_count", 0)
                    ),
                    "observable_telemetry_estimates": dict(
                        observable_telemetry.get("observable_telemetry_estimates", {})
                    ),
                    "staggered": observable_telemetry.get("staggered", None),
                    "staggered_exact": None,
                    "abs_staggered_error": None,
                    "doublon": observable_telemetry.get("doublon", None),
                    "doublon_exact": None,
                    "abs_doublon_error": None,
                    "site_occupations": list(
                        observable_telemetry.get("site_occupations", [])
                    ),
                    "site_occupations_exact": None,
                    "site_occupations_up": list(
                        observable_telemetry.get("site_occupations_up", [])
                    ),
                    "site_occupations_up_exact": None,
                    "site_occupations_dn": list(
                        observable_telemetry.get("site_occupations_dn", [])
                    ),
                    "site_occupations_dn_exact": None,
                    "site_occupations_label": observable_telemetry.get(
                        "site_occupations_label", None
                    ),
                    "site_occupations_component_labels": list(
                        observable_telemetry.get("site_occupations_component_labels", [])
                    ),
                    "emitter_mode_labels": list(
                        observable_telemetry.get("emitter_mode_labels", [])
                    ),
                    "emitter_ground_occupation": observable_telemetry.get(
                        "emitter_ground_occupation", None
                    ),
                    "emitter_excited_occupation": observable_telemetry.get(
                        "emitter_excited_occupation", None
                    ),
                    "boson_number": observable_telemetry.get("boson_number", None),
                    "emitter_imbalance": observable_telemetry.get("emitter_imbalance", None),
                    "spin_x": observable_telemetry.get("spin_x", None),
                    "spinless_particle_number": observable_telemetry.get(
                        "spinless_particle_number", None
                    ),
                    "spinless_staggered_density": observable_telemetry.get(
                        "spinless_staggered_density", None
                    ),
                    "boson_number_total": observable_telemetry.get(
                        "boson_number_total", None
                    ),
                    "site0_occupation": observable_telemetry.get("site0_occupation", None),
                    "site_occupations_abs_error": None,
                    "site_occupations_abs_error_max": None,
                    "logical_block_count": int(logical_before),
                    "runtime_parameter_count": int(runtime_before),
                    "runtime_parameter_count_before": int(runtime_before),
                    "runtime_parameter_count_after": int(runtime_after_planned),
                    "runtime_parameter_count_delta": int(runtime_after_planned) - int(runtime_before),
                    "selected_noisy_energy_mean": oracle_commit_payload.get(
                        "selected_noisy_energy_mean", None
                    ),
                    "selected_noisy_energy_stderr": oracle_commit_payload.get(
                        "selected_noisy_energy_stderr", None
                    ),
                    "selected_noisy_backend_info": oracle_commit_payload.get(
                        "selected_noisy_backend_info", None
                    ),
                    "stay_noisy_energy_mean": oracle_commit_payload.get(
                        "stay_noisy_energy_mean", None
                    ),
                    "stay_noisy_energy_stderr": oracle_commit_payload.get(
                        "stay_noisy_energy_stderr", None
                    ),
                    "stay_noisy_backend_info": oracle_commit_payload.get(
                        "stay_noisy_backend_info", None
                    ),
                    "baseline_backend_info": dict(baseline.get("backend_info", {})),
                    "selected_noisy_improvement_abs": oracle_commit_payload.get(
                        "selected_noisy_improvement_abs", None
                    ),
                    "selected_noisy_improvement_ratio": oracle_commit_payload.get(
                        "selected_noisy_improvement_ratio", None
                    ),
                    **drive_diagnostics,
                    "degraded_reason": degraded_reason,
                    "baseline_geometry": dataclass_to_payload(baseline_summary),
                    "candidate_pool_diagnostics": candidate_pool_diagnostics,
                    "raw_scout_record_count": int(len(scout_records)),
                    "shortlisted_candidate_count": int(len(shortlist)),
                    "confirmed_candidate_count": int(len(confirmed)),
                    "shortlist": shortlist_payload,
                    "confirmed": confirmed_payload,
                    "prune_candidates": [],
                    "predicted_displacement": float(predicted_displacement),
                    "motion_regime": str(motion_telemetry.regime),
                    "motion_direction_cosine": (
                        None
                        if motion_telemetry.direction_cosine is None
                        else float(motion_telemetry.direction_cosine)
                    ),
                    "motion_rate_change_ratio": (
                        None
                        if motion_telemetry.rate_change_ratio is None
                        else float(motion_telemetry.rate_change_ratio)
                    ),
                    "motion_acceleration_l2": (
                        None
                        if motion_telemetry.acceleration_l2 is None
                        else float(motion_telemetry.acceleration_l2)
                    ),
                    "motion_curvature_cosine": (
                        None
                        if motion_telemetry.curvature_cosine is None
                        else float(motion_telemetry.curvature_cosine)
                    ),
                    "motion_direction_reversal": bool(motion_telemetry.direction_reversal),
                    "motion_curvature_sign_flip": bool(motion_telemetry.curvature_sign_flip),
                    "motion_kink_score": float(motion_telemetry.kink_score),
                }
                self._trajectory.append(dict(common_row))

                if str(action_kind) == "append_candidate" and selected is not None:
                    candidate_data = dict(selected["candidate_data"])
                    self.current_terms = list(candidate_data["aug_terms"])
                    self.current_layout = candidate_data["aug_layout"]
                    self.current_executor = candidate_data["aug_executor"]
                    self.current_theta = np.asarray(commit_theta_next, dtype=float).reshape(-1)
                    self._append_counter += 1
                    self._previous_append_position = int(selected_position_id)
                    self._planning_audit.commit(
                        planning_group_keys_for_term(selected["candidate_term"])
                    )
                    appended_carrier = selected["candidate_data"].get("candidate_carrier")
                    appended_label = str(
                        selected_candidate_label
                        if appended_carrier is None
                        else getattr(appended_carrier, "label", selected_candidate_label)
                    )
                    self._block_birth_checkpoint[appended_label] = int(checkpoint_index)
                    self._block_cooldown[appended_label] = 0
                    self._block_burden[appended_label] = float(
                        selected["candidate_summary"].compile_proxy_total
                    )
                    self._block_origin[appended_label] = "append"
                    self._block_motion_history.setdefault(appended_label, [])
                    self._block_fit_history.setdefault(appended_label, [])
                    tier_reached = "commit"
                else:
                    self.current_theta = np.asarray(commit_theta_next, dtype=float).reshape(-1)
                    tier_reached = "confirm" if shortlist else "scout"
                self._record_theta_dot_history(np.asarray(commit_theta_dot, dtype=float))
                self._set_previous_block_theta_snapshot()

                ledger_row = {
                    **dict(common_row),
                    "shortlist_size": int(len(shortlist)),
                    "tier_reached": str(tier_reached),
                    "logical_block_count_before": int(logical_before),
                    "logical_block_count_after": int(self.current_layout.logical_parameter_count),
                    "runtime_parameter_count_before": int(runtime_before),
                    "runtime_parameter_count_after": int(self.current_layout.runtime_parameter_count),
                    "rate_change_l2": (
                        None if rate_change_l2 is None else float(rate_change_l2)
                    ),
                    "exact_cache_hits": int(cache.summary()["hits"]),
                    "exact_cache_misses": 0,
                    "geometry_memo_hits": int(geometry_memo.summary()["hits"]),
                    "geometry_memo_misses": int(geometry_memo.summary()["misses"]),
                    "planning_groups_new_selected": float(selected_groups_new),
                    "gain_ratio_selected": float(selected_gain_ratio),
                    "oracle_cache_hits": int(oracle_cache.summary()["hits"]),
                    "oracle_cache_misses": int(oracle_cache.summary()["misses"]),
                    "raw_group_cache_hits": (
                        0 if raw_group_pool is None else int(raw_group_pool.summary()["hits"])
                    ),
                    "raw_group_cache_misses": (
                        0 if raw_group_pool is None else int(raw_group_pool.summary()["misses"])
                    ),
                    "raw_group_cache_extensions": (
                        0
                        if raw_group_pool is None
                        else int(raw_group_pool.summary()["extensions"])
                    ),
                    "analytic_noise_std": float(self.cfg.analytic_noise_std),
                    "analytic_noise_seed": getattr(self.cfg, "analytic_noise_seed", None),
                }
                self._ledger.append(dict(ledger_row))
                self._temporal_ledger.record_checkpoint(
                    checkpoint_index=int(checkpoint_index),
                    selected_candidate_identity=(
                        None
                        if selected is None
                        else str(selected.get("candidate_identity", selected_candidate_label))
                    ),
                    selected_position_id=selected_position_id,
                    selected_groups_new=float(selected_groups_new),
                    selected_gain_ratio=float(selected_gain_ratio),
                    predicted_displacement=float(predicted_displacement),
                    refresh_pressure=str(refresh_pressure),
                )
                self._write_progress(
                    stage="checkpoint_done",
                    force=True,
                    checkpoint_index=int(checkpoint_index),
                    time=float(time_value),
                    physical_time=float(step_hamiltonian.physical_time),
                    action_kind=str(action_kind),
                    controller_lane=str(controller_lane),
                    decision_backend="oracle",
                    oracle_decision_used=True,
                    shortlist_size=int(len(shortlist)),
                    oracle_confirm_limit=int(oracle_confirm_limit),
                    oracle_budget_scale=float(oracle_budget_scale),
                    strict_qpu_faithful=True,
                    strict_qpu_hh=bool(self.strict_qpu_hh),
                    strict_qpu_family=str(self._family_key),
                )
                self._write_partial_payload(stage="checkpoint_done")
            return _finish(
                strict_fail_closed=False,
                strict_fail_closed_reason=None,
                status="completed",
            )
        finally:
            self._close_oracles()

    def run(self, *, checkpoint_observer: Any | None = None) -> ControllerRunArtifacts:
        if bool(getattr(self, "strict_qpu_faithful", False)) and str(self.cfg.mode) == "oracle_v1":
            return self._run_strict_qpu_hh(checkpoint_observer=checkpoint_observer)
        if checkpoint_observer is None and self._reference_mode() == "benchmark_exact":
            from pipelines.time_dynamics.legacy.checkpoint_exact_audit import (
                run_controller_with_exact_audit,
            )

            return run_controller_with_exact_audit(self)
        self._run_wallclock_start = time.perf_counter()
        early_stop_reason: str | None = None
        early_stop_checkpoint_index: int | None = None
        early_stop_time: float | None = None
        try:
            self._write_progress(stage="run_start", force=True)
            self._write_partial_payload(stage="run_start")
            checkpoint_index = 0
            repair_retry_attempts: dict[int, int] = {}
            while int(checkpoint_index) < int(len(self.times)):
                time_value = float(self.times[int(checkpoint_index)])
                repair_attempt_index = int(repair_retry_attempts.get(int(checkpoint_index), 0))
                repair_attempt = self._set_repair_attempt_state(repair_attempt_index)
                time_stop = None
                if int(checkpoint_index) + 1 < int(len(self.times)):
                    time_stop = float(self.times[int(checkpoint_index) + 1])
                step_sample_time = self._projection_sample_time(float(time_value), time_stop)
                step_hamiltonian = self._step_hamiltonian_artifacts(float(step_sample_time))
                self._write_progress(
                    stage="checkpoint_start",
                    force=True,
                    checkpoint_index=int(checkpoint_index),
                    time=float(time_value),
                    physical_time=float(step_hamiltonian.physical_time),
                    drive_term_count=int(step_hamiltonian.drive_term_count),
                    repair_attempt_index=int(repair_attempt.attempt_index),
                    repair_max_attempts=repair_attempt.max_attempts,
                    repair_escalation_kind=repair_attempt.escalation_kind,
                )
                psi_current = self.current_executor.prepare_state(self.current_theta, self.replay_context.psi_ref)
                checkpoint_ctx = make_checkpoint_context(
                    checkpoint_index=int(checkpoint_index),
                    time_start=float(time_value),
                    time_stop=(None if time_stop is None else float(time_stop)),
                    scaffold_labels=self._current_scaffold_labels(),
                    theta=self.current_theta,
                    psi=psi_current,
                    logical_count=int(self.current_layout.logical_parameter_count),
                    runtime_count=int(self.current_layout.runtime_parameter_count),
                    resolved_family=str(self.replay_context.family_info.get("resolved", "unknown")),
                    grouping_mode=str(self.cfg.grouping_mode),
                    structure_locked=False,
                )
                cache = ExactCheckpointValueCache(
                    checkpoint_id=str(checkpoint_ctx.checkpoint_id),
                    grouping_mode=str(self.cfg.grouping_mode),
                )
                geometry_memo = DerivedGeometryMemo(
                    checkpoint_id=str(checkpoint_ctx.checkpoint_id),
                )
                oracle_cache = OracleCheckpointValueCache(
                    checkpoint_id=str(checkpoint_ctx.checkpoint_id),
                ) if str(self.cfg.mode) == "oracle_v1" else None
                raw_group_pool = (
                    BackendScheduledRawGroupPool(checkpoint_id=str(checkpoint_ctx.checkpoint_id))
                    if self._oracle_base_config is not None
                    and bool(controller_oracle_supports_raw_group_sampling(self._oracle_base_config))
                    else None
                )
                layout_at_checkpoint = self.current_layout
                theta_runtime_at_checkpoint = np.asarray(
                    self.current_theta, dtype=float
                ).reshape(-1).copy()
                scaffold_labels_at_checkpoint = list(self._current_scaffold_labels())
                baseline_exact = self._baseline_geometry(
                    checkpoint_ctx,
                    cache,
                    geometry_memo,
                    step_hamiltonian=step_hamiltonian,
                )
                baseline_for_decision = baseline_exact
                degraded_reason = baseline_exact.get("analytic_noise_degraded_reason")
                decision_backend = (
                    "ideal_observable" if str(self.cfg.mode) == "observable_v1" else "exact"
                )
                decision_noise_mode: str | None = (
                    "ideal" if str(self.cfg.mode) == "observable_v1" else None
                )
                oracle_attempted = False
                oracle_decision_used = False
                oracle_estimate_kind = None
                baseline_step_scale: float | None = None
                baseline_blend_weight: float | None = None
                baseline_gain_scale: float | None = None
                baseline_proposal_kind: str | None = None
                baseline_step_forecast: dict[str, Any] | None = None
                selection_metric = (
                    "off_stay_baseline"
                    if str(self.cfg.mode) == "off"
                    else "incremental_gain_ratio"
                )
                self._last_exact_v1_postcross_compare_diag = None
                dt = 0.0 if time_stop is None else float(time_stop - float(time_value))
                if (
                    time_stop is not None
                    and str(self.cfg.mode) == "exact_v1"
                    and bool(self._drive_aligned_density_active)
                    and degraded_reason is None
                ):
                    try:
                        (
                            scaled_theta_dot,
                            baseline_step_scale,
                            baseline_blend_weight,
                            baseline_gain_scale,
                            baseline_step_forecast,
                        ) = (
                            self._select_exact_v1_baseline_step_scale(
                                checkpoint_index=int(checkpoint_index),
                                baseline_theta_dot=np.asarray(
                                    baseline_exact["theta_dot_step"], dtype=float
                                ).reshape(-1),
                                baseline=baseline_exact,
                                dt=float(dt),
                                time_stop=float(time_stop),
                            )
                        )
                        baseline_for_decision = dict(baseline_exact)
                        baseline_for_decision["theta_dot_step"] = np.asarray(
                            scaled_theta_dot, dtype=float
                        ).reshape(-1)
                        baseline_proposal_kind = (
                            None
                            if baseline_step_forecast is None
                            else baseline_step_forecast.get("baseline_proposal_kind")
                        )
                    except Exception as exc:
                        degraded_reason = f"exact_baseline_step_scale_error: {type(exc).__name__}: {exc}"
                predicted_displacement = self._predicted_displacement(dt=float(dt), baseline=baseline_for_decision)
                motion_telemetry = self._motion_telemetry(
                    theta_dot=np.asarray(baseline_for_decision["theta_dot_step"], dtype=float).reshape(-1),
                    predicted_displacement=float(predicted_displacement),
                )
                repair_noadvance_snapshot = self._repair_noadvance_state_snapshot()
                self._decrement_prune_cooldowns()
                self._record_prune_histories(baseline=baseline_for_decision)
                prune_candidates: list[dict[str, Any]] = []
                prune_reason = (
                    str(degraded_reason)
                    if degraded_reason is not None
                    else "exact_rho_miss_below_threshold"
                )
                prune_blocker_reason = str(prune_reason)
                if time_stop is None:
                    prune_blocker_reason = "terminal_checkpoint"
                elif str(self.cfg.mode) == "off":
                    prune_blocker_reason = "controller_disabled"
                elif str(getattr(self.cfg, "prune_mode", "off")) == "off":
                    prune_blocker_reason = "prune_disabled"
                elif (
                    self._high_miss_active(baseline=baseline_exact)
                    and not self._recoverability_prune_enabled()
                ):
                    prune_blocker_reason = "high_miss_append_lane"
                elif (
                    degraded_reason is None
                    and (
                        float(baseline_exact["summary"].rho_miss) <= float(self.cfg.miss_threshold)
                        or self._recoverability_prune_enabled()
                    )
                ):
                    prune_candidates, prune_reason = self._prune_candidates(
                        checkpoint_index=int(checkpoint_index),
                        baseline=baseline_exact,
                        motion=motion_telemetry,
                    )
                    if not prune_candidates:
                        prune_blocker_reason = str(prune_reason)
                if degraded_reason is not None:
                    controller_lane, controller_lane_reason = "stay", str(degraded_reason)
                else:
                    controller_lane, controller_lane_reason = self._controller_lane(
                        time_stop=time_stop,
                        baseline=baseline_exact,
                        prune_candidates_available=bool(prune_candidates),
                        prune_reason=str(prune_reason),
                    )
                if str(controller_lane) != "prune":
                    self._record_prune_blocker_reason(str(prune_blocker_reason or controller_lane_reason))
                self._record_high_miss_history(baseline=baseline_exact)
                base_refresh_pressure = self._temporal_ledger.refresh_pressure(
                    predicted_displacement=float(predicted_displacement),
                    rho_miss=float(baseline_exact["summary"].rho_miss),
                    condition_number=float(baseline_exact["summary"].condition_number),
                )
                refresh_pressure = self._effective_refresh_pressure(
                    base_refresh_pressure=str(base_refresh_pressure),
                    motion=motion_telemetry,
                )
                shortlist_cfg = self._shortlist_cfg_for_motion(motion_telemetry)
                oracle_budget_scale = self._oracle_budget_scale_for_motion(
                    refresh_pressure=str(refresh_pressure),
                    motion=motion_telemetry,
                )
                oracle_commit_payload = {
                    "stay_noisy_energy_mean": None,
                    "stay_noisy_energy_stderr": None,
                    "selected_noisy_energy_mean": None,
                    "selected_noisy_energy_stderr": None,
                    "selected_noisy_improvement_abs": None,
                    "selected_noisy_improvement_ratio": None,
                }
                noisy_override_reason: str | None = None
                controller_override_reason: str | None = None
                self._last_append_no_harm_diagnostics = None
                proposed_selected_override: Mapping[str, Any] | None = None
                proposed_action_kind_override: str | None = None
                scout_records: list[dict[str, Any]] = []
                if time_stop is None:
                    shortlist = []
                    scout_records = []
                    confirmed = []
                    oracle_confirm_limit = 0
                    oracle_budget_scale = 0.0
                    if str(self.cfg.mode) == "off":
                        decision_backend = "off"
                    action_kind, selected = "stay", None
                elif str(self.cfg.mode) == "off":
                    shortlist = []
                    scout_records = []
                    confirmed = []
                    oracle_confirm_limit = 0
                    action_kind, selected = "stay", None
                    decision_backend = "off"
                    if self._oracle_base_config is not None:
                        oracle_attempted = True
                        decision_noise_mode = str(self._oracle_base_config.noise_mode)
                        oracle_estimate_kind = self._oracle_estimate_kind()
                        selection_metric = "measured_baseline_energy"
                        try:
                            baseline_for_decision = self._oracle_measured_baseline_geometry(
                                checkpoint_ctx=checkpoint_ctx,
                                cache=cache,
                                geometry_memo=geometry_memo,
                                raw_group_pool=raw_group_pool,
                                h_poly_step=step_hamiltonian.h_poly,
                                tier_name="confirm",
                                budget_scale=float(oracle_budget_scale),
                            )
                        except Exception as exc:
                            degraded_reason = f"measured_off_baseline_error: {exc}"
                elif str(controller_lane) == "prune":
                    shortlist = []
                    scout_records = []
                    confirmed = []
                    oracle_confirm_limit = 0
                    oracle_budget_scale = 0.0
                    selection_metric = "cached_prune_loss"
                    action_kind, selected, prune_proposed, prune_candidates, prune_error = self._select_prune_action(
                        checkpoint_index=int(checkpoint_index),
                        time_value=float(time_value),
                        time_stop=time_stop,
                        baseline=baseline_exact,
                        step_hamiltonian=step_hamiltonian,
                        prune_candidates=prune_candidates,
                    )
                    if prune_proposed is not None and str(action_kind) != "prune_coordinate":
                        proposed_selected_override = dict(prune_proposed)
                        proposed_action_kind_override = "prune_coordinate"
                        controller_override_reason = str(prune_error or "prune_rejected")
                        self._record_prune_blocker_reason(str(controller_override_reason))
                    elif prune_error is not None:
                        controller_override_reason = str(prune_error)
                        self._record_prune_blocker_reason(str(controller_override_reason))
                else:
                    if str(controller_lane) == "append":
                        self._last_scout_records = []
                        shortlist = self._scout_candidates(
                            checkpoint_ctx=checkpoint_ctx,
                            cache=cache,
                            geometry_memo=geometry_memo,
                            baseline=baseline_exact,
                            predicted_displacement=float(predicted_displacement),
                            shortlist_cfg=shortlist_cfg,
                        )
                        scout_records = [dict(item) for item in self._last_scout_records]
                    else:
                        shortlist = []
                        scout_records = []
                    confirmed = self._confirm_candidates(
                        checkpoint_ctx=checkpoint_ctx,
                        cache=cache,
                        geometry_memo=geometry_memo,
                        baseline=baseline_exact,
                        shortlist=shortlist,
                    ) if shortlist else []
                    oracle_confirm_limit = 0
                    if str(self.cfg.mode) == "oracle_v1" and shortlist and oracle_cache is not None:
                        oracle_attempted = True
                        oracle_confirm_limit = self._oracle_confirm_limit_with_selection_policy(
                            confirmed_count=len(confirmed),
                            refresh_pressure=str(refresh_pressure),
                            motion=motion_telemetry,
                        )
                        geometry_error = None
                        measured_baseline, measured_confirmed, geometry_error = self._confirm_candidates_oracle_geometry(
                            checkpoint_ctx=checkpoint_ctx,
                            cache=cache,
                            geometry_memo=geometry_memo,
                            confirmed=confirmed,
                            raw_group_pool=raw_group_pool,
                            h_poly_step=step_hamiltonian.h_poly,
                            confirm_limit=int(oracle_confirm_limit),
                            budget_scale=float(oracle_budget_scale),
                        )
                        if geometry_error is None and measured_baseline is not None:
                            baseline_for_decision = measured_baseline
                            confirmed = list(measured_confirmed)
                            decision_backend = "oracle"
                            decision_noise_mode = (
                                None if self._oracle_base_config is None else str(self._oracle_base_config.noise_mode)
                            )
                            oracle_decision_used = True
                            oracle_estimate_kind = self._oracle_estimate_kind()
                            viable_measured = [
                                rec
                                for rec in confirmed
                                if rec.get("gain_exact") is not None and rec.get("gain_ratio") is not None
                            ]
                            if float(baseline_for_decision["summary"].rho_miss) <= float(self.cfg.miss_threshold):
                                action_kind, selected = "stay", None
                                selection_metric = "measured_incremental_gain_ratio"
                            elif not viable_measured:
                                oracle_decision_used = False
                                decision_backend = "exact"
                                decision_noise_mode = None
                                oracle_estimate_kind = None
                                degraded_reason = "measured_geometry_no_viable_candidates"
                            else:
                                selection_metric = "measured_incremental_gain_ratio"
                                if str(self.cfg.oracle_selection_policy) == "measured_topk_oracle_energy":
                                    confirmed_ranked = self._sorted_confirmed_by_gain(confirmed)
                                    confirmed_for_oracle = list(confirmed_ranked[:oracle_confirm_limit])
                                    confirmed_remainder = list(confirmed_ranked[oracle_confirm_limit:])
                                    reranked_confirmed, _, rerank_error = self._confirm_candidates_oracle(
                                        checkpoint_ctx=checkpoint_ctx,
                                        baseline=baseline_for_decision,
                                        confirmed=confirmed_for_oracle,
                                        dt=float(dt),
                                        oracle_cache=oracle_cache,
                                        raw_group_pool=raw_group_pool,
                                        oracle_observable=step_hamiltonian.oracle_observable,
                                        budget_scale=float(oracle_budget_scale),
                                    )
                                    if rerank_error is None:
                                        confirmed = list(reranked_confirmed)
                                        for record in confirmed_remainder:
                                            rec = dict(record)
                                            rec["predicted_noisy_energy_mean"] = None
                                            rec["predicted_noisy_energy_stderr"] = None
                                            rec["predicted_noisy_improvement_abs"] = None
                                            rec["predicted_noisy_improvement_ratio"] = None
                                            rec["predicted_noisy_improvement_stderr"] = None
                                            rec["adjusted_noisy_improvement"] = float("-inf")
                                            rec["confirm_backend_info"] = None
                                            rec["confirm_error"] = "deferred_by_oracle_rerank_limit"
                                            confirmed.append(rec)
                                        action_kind, selected = self._select_action_oracle(
                                            baseline=baseline_for_decision,
                                            confirmed=confirmed,
                                        )
                                        selection_metric = "oracle_energy_improvement"
                                    else:
                                        if degraded_reason is None:
                                            degraded_reason = f"oracle_rerank_error: {rerank_error}"
                                        action_kind, selected = self._select_action(
                                            baseline=baseline_for_decision,
                                            confirmed=confirmed,
                                        )
                                else:
                                    action_kind, selected = self._select_action(
                                        baseline=baseline_for_decision,
                                        confirmed=confirmed,
                                    )
                            if oracle_decision_used:
                                oracle_commit_payload, commit_degraded_reason = self._oracle_commit_payload(
                                    checkpoint_ctx=checkpoint_ctx,
                                    oracle_cache=oracle_cache,
                                    raw_group_pool=raw_group_pool,
                                    baseline=baseline_for_decision,
                                    selected=selected,
                                    action_kind=str(action_kind),
                                    dt=float(dt),
                                    oracle_observable=step_hamiltonian.oracle_observable,
                                    budget_scale=float(oracle_budget_scale),
                                )
                                if commit_degraded_reason is not None:
                                    degraded_reason = str(commit_degraded_reason)
                                override_reason = self._oracle_commit_override_reason(
                                    motion=motion_telemetry,
                                    selected=selected,
                                    action_kind=str(action_kind),
                                    oracle_commit_payload=oracle_commit_payload,
                                    predicted_displacement=float(predicted_displacement),
                                    runtime_parameter_count_before=int(self.current_layout.runtime_parameter_count),
                                )
                                if override_reason is not None:
                                    noisy_override_reason = str(override_reason)
                        if geometry_error is not None and degraded_reason is None:
                            degraded_reason = str(geometry_error)
                        if not oracle_decision_used:
                            confirmed_ranked = sorted(
                                confirmed,
                                key=self._confirm_rank_key,
                            )
                            confirmed_for_oracle = list(confirmed_ranked[:oracle_confirm_limit])
                            confirmed_remainder = list(confirmed_ranked[oracle_confirm_limit:])
                            confirmed_oracle, _, scalar_degraded_reason = self._confirm_candidates_oracle(
                                checkpoint_ctx=checkpoint_ctx,
                                baseline=baseline_exact,
                                confirmed=confirmed_for_oracle,
                                dt=float(dt),
                                oracle_cache=oracle_cache,
                                raw_group_pool=raw_group_pool,
                                oracle_observable=step_hamiltonian.oracle_observable,
                                budget_scale=float(oracle_budget_scale),
                            )
                            confirmed = list(confirmed_oracle)
                            for record in confirmed_remainder:
                                rec = dict(record)
                                rec["predicted_noisy_energy_mean"] = None
                                rec["predicted_noisy_energy_stderr"] = None
                                rec["predicted_noisy_improvement_abs"] = None
                                rec["predicted_noisy_improvement_ratio"] = None
                                rec["predicted_noisy_improvement_stderr"] = None
                                rec["adjusted_noisy_improvement"] = float("-inf")
                                rec["confirm_backend_info"] = None
                                rec["confirm_error"] = "deferred_by_refresh_pressure"
                                confirmed.append(rec)
                            if scalar_degraded_reason is None:
                                decision_backend = "oracle"
                                decision_noise_mode = (
                                    None if self._oracle_base_config is None else str(self._oracle_base_config.noise_mode)
                                )
                                oracle_decision_used = True
                                oracle_estimate_kind = self._oracle_estimate_kind()
                                selection_metric = "oracle_energy_improvement"
                                action_kind, selected = self._select_action_oracle(
                                    baseline=baseline_exact,
                                    confirmed=confirmed,
                                )
                                oracle_commit_payload, commit_degraded_reason = self._oracle_commit_payload(
                                    checkpoint_ctx=checkpoint_ctx,
                                    oracle_cache=oracle_cache,
                                    raw_group_pool=raw_group_pool,
                                    baseline=baseline_exact,
                                    selected=selected,
                                    action_kind=str(action_kind),
                                    dt=float(dt),
                                    oracle_observable=step_hamiltonian.oracle_observable,
                                    budget_scale=float(oracle_budget_scale),
                                )
                                if commit_degraded_reason is not None:
                                    degraded_reason = str(commit_degraded_reason)
                                override_reason = self._oracle_commit_override_reason(
                                    motion=motion_telemetry,
                                    selected=selected,
                                    action_kind=str(action_kind),
                                    oracle_commit_payload=oracle_commit_payload,
                                    predicted_displacement=float(predicted_displacement),
                                    runtime_parameter_count_before=int(self.current_layout.runtime_parameter_count),
                                )
                                if override_reason is not None:
                                    noisy_override_reason = str(override_reason)
                            else:
                                if degraded_reason is None:
                                    degraded_reason = str(scalar_degraded_reason)
                                action_kind, selected = "stay", None
                    else:
                        oracle_confirm_limit = 0
                        oracle_budget_scale = 1.0
                        if str(self.cfg.mode) in {"exact_v1", "observable_v1"}:
                            selection_metric = (
                                "local_projective_forecast"
                                if str(self.cfg.mode) == "exact_v1"
                                else "observable_local_projective_forecast"
                            )
                            action_kind, selected = self._select_action_exact_v1(
                                checkpoint_index=int(checkpoint_index),
                                baseline=baseline_for_decision,
                                confirmed=confirmed,
                                dt=float(dt),
                                time_stop=float(time_stop),
                                stay_forecast=baseline_step_forecast,
                                motion=motion_telemetry,
                            )
                        else:
                            selection_metric = "incremental_gain_ratio"
                            action_kind, selected = self._select_action(
                                baseline=baseline_for_decision,
                                confirmed=confirmed,
                            )

                proposed_action_kind = (
                    str(proposed_action_kind_override)
                    if proposed_action_kind_override is not None
                    else str(action_kind)
                )
                proposed_selected = (
                    proposed_selected_override
                    if proposed_selected_override is not None
                    else selected
                )
                proposed_candidate_label = (
                    None
                    if proposed_selected is None
                    else str(proposed_selected["candidate_label"])
                )
                decision_override_reason: str | None = (
                    str(controller_override_reason)
                    if controller_override_reason is not None
                    else (None if noisy_override_reason is None else str(noisy_override_reason))
                )
                exact_forecast_error: str | None = None
                forecast_stay: dict[str, Any] | None = None
                forecast_selected: dict[str, Any] | None = None
                if (
                    str(self.cfg.mode) in {"oracle_v1", "exact_v1", "observable_v1"}
                    and time_stop is not None
                    and str(proposed_action_kind) == "append_candidate"
                    and proposed_selected is not None
                ):
                    try:
                        anchor_predicted_displacement = self._predicted_displacement(
                            dt=float(dt),
                            baseline=baseline_for_decision,
                        )
                        if baseline_step_forecast is None:
                            stay_theta_forecast = np.asarray(
                                self.current_theta
                                + float(dt) * np.asarray(baseline_for_decision["theta_dot_step"], dtype=float),
                                dtype=float,
                            ).reshape(-1)
                            forecast_stay = self._local_projective_forecast_rollout(
                                checkpoint_index=int(checkpoint_index),
                                time_stop=float(time_stop),
                                executor=self.current_executor,
                                layout=self.current_layout,
                                theta_runtime_start=stay_theta_forecast,
                                theta_dot_step=np.asarray(baseline_for_decision["theta_dot_step"], dtype=float).reshape(-1),
                                planning_audit=self._planning_audit,
                                scaffold_labels=self._current_scaffold_labels(),
                                immediate_gain_ratio=float(getattr(baseline_for_decision["summary"], "step_gain_ratio", 0.0)),
                                anchor_summary=baseline_for_decision["summary"],
                                anchor_predicted_displacement=float(anchor_predicted_displacement),
                            )[0]
                        else:
                            forecast_stay = dict(baseline_step_forecast)
                        if str(proposed_action_kind) == "append_candidate" and proposed_selected is not None:
                            if str(self.cfg.mode) == "exact_v1":
                                proposed_selected, forecast_selected = self._select_exact_v1_candidate_step_scale(
                                    checkpoint_index=int(checkpoint_index),
                                    baseline_theta_dot=np.asarray(
                                        baseline_for_decision["theta_dot_step"], dtype=float
                                    ).reshape(-1),
                                    selected=proposed_selected,
                                    dt=float(dt),
                                    time_stop=float(time_stop),
                                    anchor_summary=baseline_for_decision["summary"],
                                    anchor_predicted_displacement=float(anchor_predicted_displacement),
                                )
                            else:
                                selected_theta_forecast = np.asarray(
                                    proposed_selected["candidate_data"]["theta_aug"]
                                    + float(dt) * np.asarray(proposed_selected["theta_dot_aug"], dtype=float),
                                    dtype=float,
                                ).reshape(-1)
                                forecast_selected = self._local_projective_forecast_rollout(
                                    checkpoint_index=int(checkpoint_index),
                                    time_stop=float(time_stop),
                                    executor=proposed_selected["candidate_data"]["aug_executor"],
                                    layout=proposed_selected["candidate_data"]["aug_layout"],
                                    theta_runtime_start=selected_theta_forecast,
                                    theta_dot_step=np.asarray(proposed_selected["theta_dot_aug"], dtype=float).reshape(-1),
                                    planning_audit=self._build_planning_audit_for_terms(proposed_selected["candidate_data"]["aug_terms"]),
                                    scaffold_labels=[str(carrier.label) for carrier in proposed_selected["candidate_data"]["aug_terms"]],
                                    immediate_gain_ratio=float(proposed_selected.get("gain_ratio", 0.0)),
                                    anchor_summary=baseline_for_decision["summary"],
                                    anchor_predicted_displacement=float(anchor_predicted_displacement),
                                )[0]
                    except Exception as exc:
                        exact_forecast_error = f"{type(exc).__name__}: {exc}"
                        forecast_stay = None
                        forecast_selected = None
                        decision_override_reason = "local_forecast_error"
                        msg = f"local_forecast_error: {exact_forecast_error}"
                        degraded_reason = msg if degraded_reason is None else f"{degraded_reason}; {msg}"
                if (
                    decision_override_reason is None
                    and forecast_stay is not None
                    and forecast_selected is not None
                ):
                    forecast_override_reason = self._local_forecast_override_reason(
                        stay_forecast=forecast_stay,
                        selected_forecast=forecast_selected,
                        selected=proposed_selected,
                        motion=motion_telemetry,
                    )
                    if forecast_override_reason is not None:
                        decision_override_reason = str(forecast_override_reason)
                if decision_override_reason is not None:
                    action_kind, selected = "stay", None
                else:
                    action_kind = str(proposed_action_kind)
                    selected = proposed_selected

                no_admit = str(action_kind) == "stay" or selected is None
                selection_reason = getattr(self, "_last_exact_v1_selection_reason", None)
                repair_retry_next = False
                repair_terminal = False
                repair_failure_reason: str | None = None
                repair_no_admit_diagnostics: dict[str, Any] | None = None
                repair_rescue_candidate_label: str | None = None
                repair_rescue_reason: str | None = None
                repair_rescue_admitted = False
                strict_no_admit_reason = (
                    str(decision_override_reason)
                    if decision_override_reason not in {None, ""}
                    else (None if selection_reason in {None, ""} else str(selection_reason))
                )
                forecast_veto_reason = None
                if strict_no_admit_reason not in {None, ""} and str(strict_no_admit_reason).startswith(
                    ("local_forecast_", "exact_forecast_", "no_harm_")
                ):
                    forecast_veto_reason = str(strict_no_admit_reason)
                if decision_override_reason not in {None, ""} and str(decision_override_reason).startswith(
                    ("local_forecast_", "exact_forecast_", "no_harm_")
                ):
                    forecast_veto_reason = str(decision_override_reason)
                high_miss_policy = normalize_high_miss_no_admit_policy(
                    getattr(self.cfg, "high_miss_no_admit_policy", None)
                )
                high_miss_no_admit_soft_fallback = False
                high_miss_no_admit_soft_fallback_policy: str | None = None
                high_miss_no_admit_soft_fallback_reason: str | None = None
                high_miss_no_admit_soft_fallback_warning: str | None = None
                no_admit_resolution: str | None = None
                no_admit_resolution_advances_time = False
                if (
                    time_stop is not None
                    and str(controller_lane) == "append"
                    and bool(no_admit)
                    and high_miss_policy in HIGH_MISS_NO_ADMIT_POLICY_CANONICAL
                ):
                    if high_miss_policy == HIGH_MISS_NO_ADMIT_POLICY_DEFAULT:
                        high_miss_no_admit_soft_fallback = True
                        high_miss_no_admit_soft_fallback_policy = HIGH_MISS_NO_ADMIT_POLICY_DEFAULT
                        high_miss_no_admit_soft_fallback_reason = HIGH_MISS_NO_ADMIT_SOFT_FALLBACK_REASON
                        high_miss_no_admit_soft_fallback_warning = HIGH_MISS_NO_ADMIT_SOFT_FALLBACK_WARNING
                        no_admit_resolution = HIGH_MISS_NO_ADMIT_POLICY_DEFAULT
                        no_admit_resolution_advances_time = True
                    elif high_miss_policy == "repair_retry":
                        max_attempts = int(getattr(self.cfg, "repair_retry_max_attempts", 2))
                        if int(repair_attempt.attempt_index) < int(max_attempts):
                            action_kind, selected = "repair_miss", None
                            decision_override_reason = "repair_required_high_miss_no_admit"
                            repair_retry_next = True
                            no_admit_resolution = "repair_retry_next_attempt"
                            no_admit_resolution_advances_time = False
                        else:
                            (
                                rescue_action_kind,
                                rescue_selected,
                                rescue_reason,
                                rescue_forecast,
                            ) = self._select_repair_rescue_candidate(
                                confirmed_candidates=confirmed,
                                scout_records=scout_records,
                                baseline=baseline_for_decision,
                                repair_attempt=repair_attempt,
                                controller_lane=str(controller_lane),
                                proposed_action_kind=str(proposed_action_kind),
                                proposed_selected=proposed_selected,
                                checkpoint_index=int(checkpoint_index),
                                dt=float(dt),
                                time_stop=time_stop,
                                stay_forecast=forecast_stay,
                            )
                            repair_rescue_reason = rescue_reason
                            if rescue_reason not in {None, ""} and str(rescue_reason).startswith(
                                ("local_forecast_", "exact_forecast_", "no_harm_")
                            ):
                                forecast_veto_reason = str(rescue_reason)
                            if isinstance(rescue_forecast, Mapping):
                                forecast_selected = dict(rescue_forecast)
                            if str(rescue_action_kind) == "append_candidate" and rescue_selected is not None:
                                action_kind, selected = "append_candidate", dict(rescue_selected)
                                repair_rescue_admitted = True
                                repair_rescue_candidate_label = str(selected.get("candidate_label"))
                                decision_override_reason = None
                                no_admit_resolution = "repair_retry_rescue_append"
                                no_admit_resolution_advances_time = True
                            else:
                                action_kind, selected = "repair_miss", None
                                decision_override_reason = "repair_required_high_miss_no_admit"
                                repair_terminal = True
                                repair_failure_reason = "repair_retry_exhausted_high_miss_no_admit"
                                early_stop_reason = str(repair_failure_reason)
                                early_stop_checkpoint_index = int(checkpoint_index)
                                early_stop_time = float(time_value)
                                no_admit_resolution = "repair_retry_exhausted"
                                no_admit_resolution_advances_time = False
                    else:
                        action_kind, selected = "repair_miss", None
                        decision_override_reason = "repair_required_high_miss_no_admit"
                        repair_terminal = True
                        repair_failure_reason = "repair_required_high_miss_no_admit"
                        early_stop_reason = "repair_required_high_miss_no_admit"
                        early_stop_checkpoint_index = int(checkpoint_index)
                        early_stop_time = float(time_value)
                        no_admit_resolution = "repair_stop_terminal"
                        no_admit_resolution_advances_time = False
                    repair_no_admit_diagnostics = self._repair_no_admit_diagnostics(
                        controller_lane=str(controller_lane),
                        repair_attempt=repair_attempt,
                        scout_records=scout_records,
                        confirmed_candidates=confirmed,
                        proposed_action_kind=str(proposed_action_kind),
                        proposed_selected=proposed_selected,
                        strict_no_admit_reason=strict_no_admit_reason,
                        forecast_veto_reason=forecast_veto_reason,
                        no_admit_resolution=str(no_admit_resolution or high_miss_policy),
                        no_admit_resolution_advances_time=bool(no_admit_resolution_advances_time),
                        high_miss_no_admit_soft_fallback=bool(high_miss_no_admit_soft_fallback),
                        soft_fallback_reason=high_miss_no_admit_soft_fallback_reason,
                        soft_fallback_warning=high_miss_no_admit_soft_fallback_warning,
                    )

                forecast_mode = None
                if isinstance(forecast_selected, Mapping):
                    forecast_mode = forecast_selected.get("forecast_mode", None)
                if forecast_mode is None and isinstance(forecast_stay, Mapping):
                    forecast_mode = forecast_stay.get("forecast_mode", None)
                forecast_error = (
                    None if exact_forecast_error is None else str(exact_forecast_error)
                )
                selection_reason = getattr(self, "_last_exact_v1_selection_reason", None)
                forecast_stay_score_total = (
                    None
                    if not isinstance(forecast_stay, Mapping)
                    or forecast_stay.get("local_projective_score_total") is None
                    else float(forecast_stay.get("local_projective_score_total"))
                )
                forecast_selected_score_total = (
                    None
                    if not isinstance(forecast_selected, Mapping)
                    or forecast_selected.get("local_projective_score_total") is None
                    else float(forecast_selected.get("local_projective_score_total"))
                )
                forecast_score_delta_vs_stay = (
                    None
                    if forecast_stay_score_total is None or forecast_selected_score_total is None
                    else float(forecast_selected_score_total - forecast_stay_score_total)
                )
                forecast_selected_lower_than_stay = (
                    None
                    if forecast_score_delta_vs_stay is None
                    else bool(float(forecast_score_delta_vs_stay) < 0.0)
                )
                forecast_stay_predicted_displacement_next = self._forecast_first_scalar(
                    forecast_stay,
                    "predicted_displacement_next",
                    "predicted_displacement",
                )
                forecast_selected_predicted_displacement_next = self._forecast_first_scalar(
                    forecast_selected,
                    "predicted_displacement_next",
                    "predicted_displacement",
                )
                forecast_stay_epsilon_step_ratio_next = self._forecast_first_scalar(
                    forecast_stay,
                    "epsilon_step_ratio_next",
                    "step_residual_ratio_next",
                    "epsilon_step_ratio",
                )
                forecast_selected_epsilon_step_ratio_next = self._forecast_first_scalar(
                    forecast_selected,
                    "epsilon_step_ratio_next",
                    "step_residual_ratio_next",
                    "epsilon_step_ratio",
                )
                append_no_harm_diagnostics = (
                    None
                    if not isinstance(self._last_append_no_harm_diagnostics, Mapping)
                    else dict(self._last_append_no_harm_diagnostics)
                )
                append_no_harm_veto_reason = (
                    None
                    if append_no_harm_diagnostics is None
                    or append_no_harm_diagnostics.get("veto_reason") in {None, ""}
                    else str(append_no_harm_diagnostics.get("veto_reason"))
                )
                append_no_harm_exact_logging = (
                    None
                    if append_no_harm_diagnostics is None
                    or not isinstance(
                        append_no_harm_diagnostics.get("exact_reference_logging", None),
                        Mapping,
                    )
                    else dict(append_no_harm_diagnostics["exact_reference_logging"])
                )
                candidate_pool_diagnostics = dict(self._last_candidate_pool_diagnostics)
                if candidate_pool_diagnostics:
                    candidate_pool_diagnostics["raw_scout_record_count"] = int(len(scout_records))
                    candidate_pool_diagnostics["shortlisted_candidate_count"] = int(len(shortlist))
                    candidate_pool_diagnostics["confirmed_candidate_count"] = int(len(confirmed))

                if str(self.cfg.mode) == "exact_v1":
                    if (
                        str(action_kind) == "stay"
                        and str(controller_lane) == "append"
                        and str(proposed_action_kind) == "stay"
                    ):
                        self._exact_v1_append_lane_stall_streak = int(self._exact_v1_append_lane_stall_streak) + 1
                    elif str(action_kind) != "repair_miss":
                        self._exact_v1_append_lane_stall_streak = 0

                if degraded_reason is not None:
                    self._degraded_checkpoint_count += 1
                logical_before = int(self.current_layout.logical_parameter_count)
                runtime_before = int(self.current_layout.runtime_parameter_count)
                selected_groups_new = 0.0
                selected_gain_ratio = 0.0
                selected_prune_cached_loss: float | None = None
                selected_prune_stagnation_score: float | None = None
                selected_post_prune_state_jump_l2: float | None = None
                selected_prune_origin_kind: str | None = None
                selected_prune_age_checkpoints: int | None = None
                selected_prune_appended_origin_bias_factor: float | None = None
                selected_prune_appended_origin_bias_applied: bool | None = None
                selected_prune_schur_raw_loss: float | None = None
                selected_prune_schur_normalized_loss: float | None = None
                selected_prune_schur_selected_rung: int | None = None
                selected_prune_schur_monotonicity_status: str | None = None
                selected_prune_differential_miss: float | None = None
                selected_prune_permit_path: str | None = None
                selected_prune_projection_objective: float | None = None
                selected_prune_projected_state_jump_l2: float | None = None
                selected_prune_ray_distance: float | None = None
                selected_prune_shadow_score: float | None = None
                selected_prune_persistence_count: int | None = None
                selected_prune_persistence_required: int | None = None
                selected_prune_persistence_passed: bool | None = None
                selected_prune_block_theta_dot_norm: float | None = None
                selected_prune_block_theta_dot_rel: float | None = None
                selected_prune_loss_fields: dict[str, Any] = {}
                selected_candidate_label: str | None = None
                selected_position_id: int | None = None
                if selected is not None:
                    selected_candidate_label = str(selected["candidate_label"])
                    selected_position_id = int(selected["position_id"])
                    selected_prune_loss_fields = selected_prune_loss_payload(selected)
                    selected_groups_new = float(selected.get("groups_new", 0.0))
                    selected_gain_ratio = float(selected.get("gain_ratio", 0.0))
                    if selected.get("cached_prune_loss", None) is not None:
                        selected_prune_cached_loss = float(selected["cached_prune_loss"])
                    if selected.get("stagnation_score", None) is not None:
                        selected_prune_stagnation_score = float(selected["stagnation_score"])
                    if selected.get("post_prune_state_jump_l2", None) is not None:
                        selected_post_prune_state_jump_l2 = float(selected["post_prune_state_jump_l2"])
                    if selected.get("origin_kind", None) is not None:
                        selected_prune_origin_kind = str(selected["origin_kind"])
                    if selected.get("age_checkpoints", None) is not None:
                        selected_prune_age_checkpoints = int(selected["age_checkpoints"])
                    if selected.get("theta_dot_block_norm", None) is not None:
                        selected_prune_block_theta_dot_norm = float(selected["theta_dot_block_norm"])
                    if selected.get("theta_dot_block_rel", None) is not None:
                        selected_prune_block_theta_dot_rel = float(selected["theta_dot_block_rel"])
                    if selected.get("appended_origin_bias_factor", None) is not None:
                        selected_prune_appended_origin_bias_factor = float(selected["appended_origin_bias_factor"])
                    if selected.get("appended_origin_bias_applied", None) is not None:
                        selected_prune_appended_origin_bias_applied = bool(selected["appended_origin_bias_applied"])
                    if selected.get("prune_schur_raw_loss", None) is not None:
                        selected_prune_schur_raw_loss = float(selected["prune_schur_raw_loss"])
                    if selected.get("prune_schur_normalized_loss", None) is not None:
                        selected_prune_schur_normalized_loss = float(selected["prune_schur_normalized_loss"])
                    if selected.get("prune_schur_selected_rung", None) is not None:
                        selected_prune_schur_selected_rung = int(selected["prune_schur_selected_rung"])
                    if selected.get("prune_schur_monotonicity_status", None) is not None:
                        selected_prune_schur_monotonicity_status = str(selected["prune_schur_monotonicity_status"])
                    if selected.get("prune_differential_miss", selected.get("prune_delta_rho_miss", None)) is not None:
                        selected_prune_differential_miss = float(selected.get("prune_differential_miss", selected.get("prune_delta_rho_miss")))
                    if selected.get("prune_permit_path", None) is not None:
                        selected_prune_permit_path = str(selected["prune_permit_path"])
                    if selected.get("prune_projection_objective", None) is not None:
                        selected_prune_projection_objective = float(selected["prune_projection_objective"])
                    if selected.get("prune_projected_state_jump_l2", None) is not None:
                        selected_prune_projected_state_jump_l2 = float(selected["prune_projected_state_jump_l2"])
                    if selected.get("prune_ray_distance", None) is not None:
                        selected_prune_ray_distance = float(selected["prune_ray_distance"])
                    if selected.get("prune_shadow_score", None) is not None:
                        selected_prune_shadow_score = float(selected["prune_shadow_score"])
                    if selected.get("prune_persistence_count", None) is not None:
                        selected_prune_persistence_count = int(selected["prune_persistence_count"])
                    if selected.get("prune_persistence_required", None) is not None:
                        selected_prune_persistence_required = int(selected["prune_persistence_required"])
                    if selected.get("prune_persistence_passed", None) is not None:
                        selected_prune_persistence_passed = bool(selected["prune_persistence_passed"])
                selected_step_scale = (
                    None
                    if selected is None or selected.get("candidate_step_scale", None) is None
                    else float(selected["candidate_step_scale"])
                )
                tier_reached = "scout"
                rate_change_l2 = _overlap_l2(np.asarray(baseline_for_decision["theta_dot_step"], dtype=float), self._previous_theta_dot)
                commit_theta_next: np.ndarray | None = None
                commit_theta_dot: np.ndarray | None = None
                theta_update_start = np.asarray(self.current_theta, dtype=float).reshape(-1)
                integrator_diagnostics = self._no_advance_integrator_diagnostics()
                if str(action_kind) == "append_candidate" and selected is not None:
                    candidate_data_for_integrator = dict(selected["candidate_data"])
                    theta_update_start = np.asarray(
                        candidate_data_for_integrator["theta_aug"], dtype=float
                    ).reshape(-1)
                    append_baseline_for_integrator = dict(baseline_for_decision)
                    append_baseline_for_integrator["theta_dot_step"] = np.asarray(
                        selected["theta_dot_aug"], dtype=float
                    ).reshape(-1)
                    # A selected candidate step scale defines the Euler trial direction,
                    # but it must not force Euler in auto mode.  Early/tumultuous
                    # append commits still need the same RK4/Euler gates as stay and
                    # prune commits; otherwise the first append can silently bypass
                    # the Chapter 17A early-RK4 prior.
                    append_forced_policy: str | None = None
                    (
                        commit_theta_next,
                        commit_theta_dot,
                        integrator_diagnostics,
                    ) = self._integrate_theta_one_step(
                        checkpoint_index=int(checkpoint_index),
                        time_start=float(time_value),
                        time_stop=time_stop,
                        executor=candidate_data_for_integrator["aug_executor"],
                        layout=candidate_data_for_integrator["aug_layout"],
                        theta_runtime=np.asarray(
                            candidate_data_for_integrator["theta_aug"], dtype=float
                        ).reshape(-1),
                        baseline=append_baseline_for_integrator,
                        planning_audit=self._build_planning_audit_for_terms(
                            candidate_data_for_integrator["aug_terms"]
                        ),
                        scaffold_labels=[
                            str(carrier.label)
                            for carrier in candidate_data_for_integrator["aug_terms"]
                        ],
                        forced_policy=append_forced_policy,
                        euler_theta_dot=np.asarray(
                            selected["theta_dot_aug"], dtype=float
                        ).reshape(-1),
                    )
                elif str(action_kind) == "prune_coordinate" and selected is not None:
                    reduced_state_for_integrator = dict(selected["reduced_state"])
                    reduced_baseline_for_integrator = dict(selected["pruned_baseline"])
                    theta_update_start = np.asarray(
                        reduced_state_for_integrator["reduced_theta"], dtype=float
                    ).reshape(-1)
                    (
                        commit_theta_next,
                        commit_theta_dot,
                        integrator_diagnostics,
                    ) = self._integrate_theta_one_step(
                        checkpoint_index=int(checkpoint_index),
                        time_start=float(time_value),
                        time_stop=time_stop,
                        executor=reduced_state_for_integrator["reduced_executor"],
                        layout=reduced_state_for_integrator["reduced_layout"],
                        theta_runtime=np.asarray(
                            reduced_state_for_integrator["reduced_theta"], dtype=float
                        ).reshape(-1),
                        baseline=reduced_baseline_for_integrator,
                        planning_audit=reduced_state_for_integrator["reduced_planning_audit"],
                        scaffold_labels=[
                            str(carrier.label)
                            for carrier in reduced_state_for_integrator["reduced_terms"]
                        ],
                    )
                elif str(action_kind) != "repair_miss":
                    (
                        commit_theta_next,
                        commit_theta_dot,
                        integrator_diagnostics,
                    ) = self._integrate_theta_one_step(
                        checkpoint_index=int(checkpoint_index),
                        time_start=float(time_value),
                        time_stop=time_stop,
                        executor=self.current_executor,
                        layout=self.current_layout,
                        theta_runtime=self.current_theta,
                        baseline=baseline_for_decision,
                        planning_audit=self._planning_audit,
                        scaffold_labels=self._current_scaffold_labels(),
                    )

                drive_diagnostics = self._drive_diagnostic_payload(
                    physical_time=float(step_hamiltonian.physical_time),
                    drive_term_count=int(step_hamiltonian.drive_term_count),
                )
                theta_dot_source = (
                    np.asarray(commit_theta_dot, dtype=float).reshape(-1)
                    if commit_theta_dot is not None
                    else np.asarray(baseline_for_decision["theta_dot_step"], dtype=float).reshape(-1)
                )
                theta_dot_l2 = float(np.linalg.norm(theta_dot_source))
                theta_update_l2 = (
                    None
                    if commit_theta_next is None
                    else float(
                        _overlap_l2(
                            np.asarray(commit_theta_next, dtype=float).reshape(-1),
                            theta_update_start,
                        )
                        or 0.0
                    )
                )
                runtime_after_planned = (
                    int(runtime_before)
                    if commit_theta_next is None
                    else int(np.asarray(commit_theta_next, dtype=float).reshape(-1).size)
                )
                energy_controller = float(baseline_for_decision["summary"].energy)
                controller_obs = self._observable_snapshot(
                    np.asarray(baseline_exact["psi"], dtype=complex).reshape(-1)
                )
                site_occ_controller = np.asarray(controller_obs["site_occupations"], dtype=float)
                primary_density_mode = self._exact_forecast_primary_density_target_mode()
                primary_density_controller = float(
                    self._primary_density_value_from_snapshot(controller_obs)
                )
                psi_exact: np.ndarray | None = None
                exact_obs: dict[str, Any] | None = None
                energy_exact: float | None = None
                fidelity_exact: float | None = None
                abs_energy_total_error: float | None = None
                primary_density_exact: float | None = None
                abs_primary_density_error: float | None = None
                abs_staggered_error: float | None = None
                abs_doublon_error: float | None = None
                site_occ_exact: np.ndarray | None = None
                site_occ_abs_error: np.ndarray | None = None
                post_prune_payload: dict[str, Any] | None = None
                post_prune_psi: np.ndarray | None = None
                fidelity_initial_controller = float(
                    abs(
                        np.vdot(
                            np.asarray(self.psi_initial, dtype=complex).reshape(-1),
                            np.asarray(baseline_exact["psi"], dtype=complex).reshape(-1),
                        )
                    )
                    ** 2
                )
                fidelity_initial_exact = (
                    None
                    if psi_exact is None
                    else float(
                        abs(
                            np.vdot(
                                np.asarray(self.psi_initial, dtype=complex).reshape(-1),
                                np.asarray(psi_exact, dtype=complex).reshape(-1),
                            )
                        )
                        ** 2
                    )
                )

                shortlist_payload = [
                    {
                        "candidate_label": str(item["candidate_label"]),
                        "candidate_pool_index": int(item["candidate_pool_index"]),
                        "position_id": int(item["position_id"]),
                        "runtime_insert_position": int(item["runtime_insert_position"]),
                        "runtime_block_indices": list(item["runtime_block_indices"]),
                        "residual_overlap_l2": float(item["residual_overlap_l2"]),
                        "compile_proxy_total": float(item["compile_proxy_total"]),
                        "groups_new": float(item["groups_new"]),
                        "novelty": (None if item.get("novelty") is None else float(item["novelty"])),
                        "position_jump_penalty": float(item["position_jump_penalty"]),
                        "temporal_prior_bonus": float(item.get("temporal_prior_bonus", 0.0)),
                        "scout_lower_gain": (
                            None if item.get("scout_lower_gain") is None else float(item["scout_lower_gain"])
                        ),
                        "scout_gain_ratio": (
                            None if item.get("scout_gain_ratio") is None else float(item["scout_gain_ratio"])
                        ),
                        "scout_score": float(item.get("scout_score", item.get("simple_score", float("nan")))),
                        "scout_score_kind": item.get("scout_score_kind", None),
                        "simple_score": float(item["simple_score"]),
                    }
                    for item in shortlist
                ]
                confirmed_payload = [
                    {
                        "candidate_label": str(rec["candidate_label"]),
                        "candidate_pool_index": int(rec["candidate_pool_index"]),
                        "position_id": int(rec["position_id"]),
                        "gain_exact": (
                            None if rec.get("gain_exact") is None else float(rec["gain_exact"])
                        ),
                        "gain_ratio": (
                            None if rec.get("gain_ratio") is None else float(rec["gain_ratio"])
                        ),
                        "adjusted_gain": float(rec["adjusted_gain"]),
                        "confirm_score": (
                            None if rec.get("confirm_score", rec.get("adjusted_gain", None)) is None else float(rec.get("confirm_score", rec.get("adjusted_gain")))
                        ),
                        "confirm_score_kind": rec.get("confirm_score_kind", "geometry_gain_ratio_minus_penalties"),
                        "confirm_compress_modes_used": int(rec.get("confirm_compress_modes_used", 0) or 0),
                        "confirm_support_rank": int(rec.get("confirm_support_rank", 0) or 0),
                        "confirm_compressed_gain_ratio": (
                            None if rec.get("confirm_compressed_gain_ratio") is None else float(rec.get("confirm_compressed_gain_ratio"))
                        ),
                        "confirm_compressed_gain_exact": (
                            None if rec.get("confirm_compressed_gain_exact") is None else float(rec.get("confirm_compressed_gain_exact"))
                        ),
                        "confirm_score_breakdown": {
                            "lower_is_better": False,
                            "raw_gain_ratio": (
                                None if rec.get("confirm_gain_ratio_raw") is None else float(rec.get("confirm_gain_ratio_raw"))
                            ),
                            "raw_gain_exact": (
                                None if rec.get("confirm_gain_exact_raw") is None else float(rec.get("confirm_gain_exact_raw"))
                            ),
                            "compressed_gain_ratio": (
                                None if rec.get("confirm_compressed_gain_ratio") is None else float(rec.get("confirm_compressed_gain_ratio"))
                            ),
                            "compressed_gain_exact": (
                                None if rec.get("confirm_compressed_gain_exact") is None else float(rec.get("confirm_compressed_gain_exact"))
                            ),
                            "directional_change_l2": (
                                None if rec.get("confirm_directional_change_l2") is None else float(rec.get("confirm_directional_change_l2"))
                            ),
                            "directional_penalty_value": (
                                None if rec.get("confirm_directional_penalty_value") is None else float(rec.get("confirm_directional_penalty_value"))
                            ),
                            "groups_new": (
                                None if rec.get("confirm_groups_new") is None else float(rec.get("confirm_groups_new"))
                            ),
                            "measurement_penalty_value": (
                                None if rec.get("confirm_measurement_penalty_value") is None else float(rec.get("confirm_measurement_penalty_value"))
                            ),
                            "final_confirm_score": (
                                None if rec.get("confirm_score", rec.get("adjusted_gain", None)) is None else float(rec.get("confirm_score", rec.get("adjusted_gain")))
                            ),
                            "threshold": (
                                None if rec.get("confirm_score_threshold") is None else float(rec.get("confirm_score_threshold"))
                            ),
                            "gain_ratio_threshold": (
                                None if rec.get("confirm_gain_ratio_threshold") is None else float(rec.get("confirm_gain_ratio_threshold"))
                            ),
                            "gain_exact_threshold": (
                                None if rec.get("confirm_gain_exact_threshold") is None else float(rec.get("confirm_gain_exact_threshold"))
                            ),
                            "gain_ratio_gate": bool(rec.get("confirm_gain_ratio_gate", False)),
                            "gain_exact_gate": bool(rec.get("confirm_gain_exact_gate", False)),
                            "score_gate": bool(rec.get("confirm_score_gate", False)),
                            "gate_passed": bool(rec.get("confirm_gate_passed", False)),
                            "gate_reason": rec.get("confirm_gate_reason", None),
                        },
                        "adjusted_noisy_improvement": (
                            None if rec.get("adjusted_noisy_improvement") is None or not np.isfinite(rec.get("adjusted_noisy_improvement", float("nan"))) else float(rec.get("adjusted_noisy_improvement"))
                        ),
                        "candidate_step_scale": (
                            None if rec.get("candidate_step_scale", None) is None else float(rec["candidate_step_scale"])
                        ),
                        "confirm_error": rec.get("confirm_error", None),
                        "candidate_summary": dataclass_to_payload(rec["candidate_summary"]),
                    }
                    for rec in confirmed
                ]
                prune_candidates_payload = [
                    {
                        "candidate_label": str(item["candidate_label"]),
                        "position_id": int(item["position_id"]),
                        "runtime_block_indices": [int(x) for x in item.get("runtime_block_indices", [])],
                        "cached_prune_loss": float(item.get("cached_prune_loss", 0.0)),
                        "cached_prune_loss_semantics": str(item.get("cached_prune_loss_semantics", "legacy_proxy_v1")),
                        "prune_selection_score": float(item.get("prune_selection_score", item.get("cached_prune_loss", 0.0))),
                        "prune_loss_delta_g_theorem": item.get("prune_loss_delta_g_theorem", None),
                        "prune_loss_delta_g_theorem_signed": item.get("prune_loss_delta_g_theorem_signed", None),
                        "prune_loss_delta_k_damped": item.get("prune_loss_delta_k_damped", None),
                        "prune_loss_delta_k_damped_signed": item.get("prune_loss_delta_k_damped_signed", None),
                        "prune_loss_legacy_proxy": item.get("prune_loss_legacy_proxy", None),
                        "prune_loss_selected": item.get("prune_loss_selected", None),
                        "prune_loss_selected_kind": item.get("prune_loss_selected_kind", None),
                        "prune_loss_denominator": item.get("prune_loss_denominator", None),
                        "prune_loss_denominator_kind": item.get("prune_loss_denominator_kind", None),
                        "prune_loss_theorem_denominator": item.get("prune_loss_theorem_denominator", None),
                        "prune_loss_theorem_denominator_kind": item.get("prune_loss_theorem_denominator_kind", None),
                        "prune_loss_support_kind": item.get("prune_loss_support_kind", None),
                        "prune_loss_removed_runtime_indices": item.get("prune_loss_removed_runtime_indices", None),
                        "prune_loss_support_runtime_indices": item.get("prune_loss_support_runtime_indices", None),
                        "prune_loss_support_size": item.get("prune_loss_support_size", None),
                        "prune_loss_matrix_for_selection": item.get("prune_loss_matrix_for_selection", None),
                        "prune_loss_pinv_policy_id": item.get("prune_loss_pinv_policy_id", None),
                        "prune_loss_pinv_rcond": item.get("prune_loss_pinv_rcond", None),
                        "prune_loss_regularization_lambda": item.get("prune_loss_regularization_lambda", None),
                        "prune_loss_regularization_source": item.get("prune_loss_regularization_source", None),
                        "prune_loss_negative_clip_applied": item.get("prune_loss_negative_clip_applied", None),
                        "prune_loss_monotonicity_status": item.get("prune_loss_monotonicity_status", None),
                        "prune_rank_score": item.get("prune_rank_score", None),
                        "prune_rank_score_kind": item.get("prune_rank_score_kind", None),
                        "prune_rank_score_terms": item.get("prune_rank_score_terms", None),
                        "stagnation_score": float(item.get("stagnation_score", 0.0)),
                        "stagnation_score_for_gate": float(item.get("stagnation_score_for_gate", item.get("stagnation_score", 0.0))),
                        "motion_mean": float(item.get("motion_mean", 0.0)),
                        "fit_mean": float(item.get("fit_mean", 0.0)),
                        "theta_block_norm": float(item.get("theta_block_norm", 0.0)),
                        "burden": float(item.get("burden", 0.0)),
                        "origin_kind": str(item.get("origin_kind", "initial_scaffold")),
                        "append_origin": bool(item.get("append_origin", False)),
                        "birth_checkpoint": int(item.get("birth_checkpoint", 0)),
                        "age_checkpoints": int(item.get("age_checkpoints", 0)),
                        "appended_origin_grace_steps": int(item.get("appended_origin_grace_steps", 0)),
                        "appended_origin_bias_enabled": bool(item.get("appended_origin_bias_enabled", False)),
                        "appended_origin_bias_factor": float(item.get("appended_origin_bias_factor", 0.0)),
                        "appended_origin_bias_applied": bool(item.get("appended_origin_bias_applied", False)),
                        "post_prune_state_jump_l2": (
                            None if item.get("post_prune_state_jump_l2") is None else float(item.get("post_prune_state_jump_l2"))
                        ),
                        "prune_delta_rho_miss": (
                            None if item.get("prune_delta_rho_miss") is None else float(item.get("prune_delta_rho_miss"))
                        ),
                        "prune_no_harm_diagnostics": dict(item.get("prune_no_harm_diagnostics", {})),
                        "prune_no_harm_score_delta": (
                            None if item.get("prune_no_harm_score_delta") is None else float(item.get("prune_no_harm_score_delta"))
                        ),
                        "prune_no_harm_step_residual_ratio_delta": (
                            None
                            if item.get("prune_no_harm_step_residual_ratio_delta") is None
                            else float(item.get("prune_no_harm_step_residual_ratio_delta"))
                        ),
                        "prune_no_harm_rho_miss_next_delta": (
                            None
                            if item.get("prune_no_harm_rho_miss_next_delta") is None
                            else float(item.get("prune_no_harm_rho_miss_next_delta"))
                        ),
                        "prune_schur_raw_loss": (
                            None if item.get("prune_schur_raw_loss") is None else float(item.get("prune_schur_raw_loss"))
                        ),
                        "prune_schur_normalized_loss": (
                            None
                            if item.get("prune_schur_normalized_loss") is None
                            else float(item.get("prune_schur_normalized_loss"))
                        ),
                        "prune_schur_selected_rung": (
                            None if item.get("prune_schur_selected_rung") is None else int(item.get("prune_schur_selected_rung"))
                        ),
                        "prune_schur_monotonicity_status": item.get("prune_schur_monotonicity_status", None),
                        "prune_permit_path": item.get("prune_permit_path", None),
                        "prune_differential_miss": (
                            None if item.get("prune_differential_miss") is None else float(item.get("prune_differential_miss"))
                        ),
                        "prune_projection_objective": (
                            None if item.get("prune_projection_objective") is None else float(item.get("prune_projection_objective"))
                        ),
                        "prune_projected_state_jump_l2": (
                            None if item.get("prune_projected_state_jump_l2") is None else float(item.get("prune_projected_state_jump_l2"))
                        ),
                        "prune_ray_distance": (
                            None if item.get("prune_ray_distance") is None else float(item.get("prune_ray_distance"))
                        ),
                        "prune_shadow_score": (
                            None if item.get("prune_shadow_score") is None else float(item.get("prune_shadow_score"))
                        ),
                        "prune_persistence_count": (
                            None if item.get("prune_persistence_count") is None else int(item.get("prune_persistence_count"))
                        ),
                        "prune_persistence_required": (
                            None if item.get("prune_persistence_required") is None else int(item.get("prune_persistence_required"))
                        ),
                        "prune_persistence_passed": (
                            None if item.get("prune_persistence_passed") is None else bool(item.get("prune_persistence_passed"))
                        ),
                        "prune_accept": (None if item.get("prune_accept") is None else bool(item.get("prune_accept"))),
                        "prune_rejection_reason": item.get("prune_rejection_reason", None),
                    }
                    for item in prune_candidates
                ]

                exact_future_decision_keys = (
                    "baseline_tangent_secant_next_exact_energy_delta",
                    "baseline_tangent_secant_signed_energy_lead",
                    "fidelity_exact_next",
                    "energy_total_exact_next",
                    "abs_energy_total_error_next",
                    "primary_density_exact_next",
                    "site_occupations_exact_next",
                    "doublon_exact_next",
                    "staggered_exact_next",
                )

                def _payload_has_future_exact_signal(payload: Any) -> bool:
                    if not isinstance(payload, Mapping):
                        return False
                    return any(
                        payload.get(key, None) is not None
                        for key in exact_future_decision_keys
                    )

                uses_future_exact_forecast_for_decision = bool(
                    str(decision_override_reason or "").startswith("exact_forecast_")
                    or (
                        str(self.cfg.mode) == "exact_v1"
                        and (
                            _payload_has_future_exact_signal(baseline_step_forecast)
                            or _payload_has_future_exact_signal(forecast_stay)
                            or _payload_has_future_exact_signal(forecast_selected)
                        )
                        and (
                            baseline_step_scale is not None
                            or str(proposed_action_kind) == "append_candidate"
                            or decision_override_reason is not None
                        )
                    )
                )
                decision_flow_fields = decision_data_flow_fields(
                    controller_mode=str(self.cfg.mode),
                    controller_exact_input_mode=self._reference_mode(),
                    decision_backend=str(decision_backend),
                    decision_noise_mode=decision_noise_mode,
                    strict_qpu_faithful=bool(getattr(self, "strict_qpu_faithful", False)),
                    uses_reference_for_decision=bool(
                        uses_future_exact_forecast_for_decision
                    ),
                    uses_future_exact_forecast_for_decision=bool(
                        uses_future_exact_forecast_for_decision
                    ),
                )

                self._trajectory.append(
                    {
                        "checkpoint_index": int(checkpoint_index),
                        "time": float(time_value),
                        "physical_time": float(step_hamiltonian.physical_time),
                        "action_kind": str(action_kind),
                        "trajectory_sample_kind": (
                            "repair_event" if str(action_kind) == "repair_miss" else "state_sample"
                        ),
                        "advances_time": bool(str(action_kind) != "repair_miss"),
                        "repair_attempt_index": int(repair_attempt.attempt_index),
                        "repair_max_attempts": repair_attempt.max_attempts,
                        "repair_escalation_kind": repair_attempt.escalation_kind,
                        "repair_retry_next": bool(repair_retry_next),
                        "repair_terminal": bool(repair_terminal),
                        "repair_failure_reason": repair_failure_reason,
                        "accepted_after_repair": bool(
                            (str(action_kind) != "repair_miss" and int(repair_attempt.attempt_index) > 0)
                            or bool(repair_rescue_admitted)
                        ),
                        "repair_no_admit_diagnostics": repair_no_admit_diagnostics,
                        "repair_rescue_candidate_label": repair_rescue_candidate_label,
                        "repair_rescue_reason": repair_rescue_reason,
                        "repair_rescue_admitted": bool(repair_rescue_admitted),
                        "high_miss_no_admit_soft_fallback": bool(high_miss_no_admit_soft_fallback),
                        "high_miss_no_admit_soft_fallback_policy": high_miss_no_admit_soft_fallback_policy,
                        "high_miss_no_admit_soft_fallback_reason": high_miss_no_admit_soft_fallback_reason,
                        "high_miss_no_admit_soft_fallback_warning": high_miss_no_admit_soft_fallback_warning,
                        "candidate_label": selected_candidate_label,
                        "proposed_action_kind": str(proposed_action_kind),
                        "proposed_candidate_label": proposed_candidate_label,
                        "controller_lane": str(controller_lane),
                        "controller_lane_reason": str(controller_lane_reason),
                        "requested_mode": str(self.cfg.mode),
                        "decision_backend": str(decision_backend),
                        "decision_noise_mode": decision_noise_mode,
                        **decision_flow_fields,
                        "oracle_attempted": bool(oracle_attempted),
                        "oracle_decision_used": bool(oracle_decision_used),
                        "oracle_estimate_kind": oracle_estimate_kind,
                        "selection_metric": str(selection_metric),
                        "exact_v1_selection_reason": getattr(self, "_last_exact_v1_selection_reason", None),
                        **(
                            {
                                "exact_v1_postcross_compare_diag": getattr(
                                    self,
                                    "_last_exact_v1_postcross_compare_diag",
                                    None,
                                )
                            }
                            if self._exact_v1_postcross_compare_diag_enabled()
                            else {}
                        ),
                        "exact_v1_append_lane_stall_streak": (
                            int(self._exact_v1_append_lane_stall_streak)
                            if str(self.cfg.mode) == "exact_v1"
                            else None
                        ),
                        "decision_override_reason": decision_override_reason,
                        "selection_reason": selection_reason,
                        "forecast_mode": forecast_mode,
                        "forecast_error": forecast_error,
                        "exact_forecast_error": exact_forecast_error,
                        "forecast_stay_score_total": forecast_stay_score_total,
                        "forecast_selected_score_total": forecast_selected_score_total,
                        "forecast_score_delta_vs_stay": forecast_score_delta_vs_stay,
                        "forecast_score_interpretation": "lower_is_better",
                        "forecast_selected_lower_than_stay": forecast_selected_lower_than_stay,
                        "baseline_step_scale": baseline_step_scale,
                        "baseline_blend_weight": baseline_blend_weight,
                        "baseline_gain_scale": baseline_gain_scale,
                        "baseline_proposal_kind": baseline_proposal_kind,
                        "baseline_current_theta_dot_norm": (
                            None
                            if baseline_step_forecast is None
                            else baseline_step_forecast.get("baseline_current_theta_dot_norm")
                        ),
                        "baseline_current_drive_only_norm": (
                            None
                            if baseline_step_forecast is None
                            else baseline_step_forecast.get("baseline_current_drive_only_norm")
                        ),
                        "baseline_lookahead_drive_only_norm": (
                            None
                            if baseline_step_forecast is None
                            else baseline_step_forecast.get("baseline_lookahead_drive_only_norm")
                        ),
                        "baseline_tangent_secant_current_energy_bias": (
                            None
                            if baseline_step_forecast is None
                            else baseline_step_forecast.get("baseline_tangent_secant_current_energy_bias")
                        ),
                        "baseline_tangent_secant_next_exact_energy_delta": (
                            None
                            if baseline_step_forecast is None
                            else baseline_step_forecast.get("baseline_tangent_secant_next_exact_energy_delta")
                        ),
                        "baseline_tangent_secant_signed_energy_lead": (
                            None
                            if baseline_step_forecast is None
                            else baseline_step_forecast.get("baseline_tangent_secant_signed_energy_lead")
                        ),
                        "baseline_tangent_secant_signed_energy_lead_taper": (
                            None
                            if baseline_step_forecast is None
                            else baseline_step_forecast.get("baseline_tangent_secant_signed_energy_lead_taper")
                        ),
                        "selected_step_scale": selected_step_scale,
                        "forecast_stay_rho_miss_next": (
                            None
                            if not isinstance(forecast_stay, Mapping)
                            or forecast_stay.get("rho_miss_next") is None
                            else float(forecast_stay.get("rho_miss_next"))
                        ),
                        "forecast_selected_rho_miss_next": (
                            None
                            if not isinstance(forecast_selected, Mapping)
                            or forecast_selected.get("rho_miss_next") is None
                            else float(forecast_selected.get("rho_miss_next"))
                        ),
                        "forecast_stay_step_gain_ratio_next": (
                            None
                            if not isinstance(forecast_stay, Mapping)
                            or forecast_stay.get("step_gain_ratio_next") is None
                            else float(forecast_stay.get("step_gain_ratio_next"))
                        ),
                        "forecast_selected_step_gain_ratio_next": (
                            None
                            if not isinstance(forecast_selected, Mapping)
                            or forecast_selected.get("step_gain_ratio_next") is None
                            else float(forecast_selected.get("step_gain_ratio_next"))
                        ),
                        "forecast_stay_condition_number_next": (
                            None
                            if not isinstance(forecast_stay, Mapping)
                            or forecast_stay.get("condition_number_next") is None
                            else float(forecast_stay.get("condition_number_next"))
                        ),
                        "forecast_selected_condition_number_next": (
                            None
                            if not isinstance(forecast_selected, Mapping)
                            or forecast_selected.get("condition_number_next") is None
                            else float(forecast_selected.get("condition_number_next"))
                        ),
                        "forecast_stay_predicted_displacement_next": (
                            None
                            if forecast_stay_predicted_displacement_next is None
                            else float(forecast_stay_predicted_displacement_next)
                        ),
                        "forecast_selected_predicted_displacement_next": (
                            None
                            if forecast_selected_predicted_displacement_next is None
                            else float(forecast_selected_predicted_displacement_next)
                        ),
                        "forecast_stay_epsilon_step_ratio_next": (
                            None
                            if forecast_stay_epsilon_step_ratio_next is None
                            else float(forecast_stay_epsilon_step_ratio_next)
                        ),
                        "forecast_selected_epsilon_step_ratio_next": (
                            None
                            if forecast_selected_epsilon_step_ratio_next is None
                            else float(forecast_selected_epsilon_step_ratio_next)
                        ),
                        "append_no_harm_veto_reason": append_no_harm_veto_reason,
                        "append_no_harm_condition_ratio": (
                            None
                            if append_no_harm_diagnostics is None
                            or append_no_harm_diagnostics.get("condition_ratio_selected_vs_stay") is None
                            else float(append_no_harm_diagnostics["condition_ratio_selected_vs_stay"])
                        ),
                        "append_no_harm_rho_miss_delta": (
                            None
                            if append_no_harm_diagnostics is None
                            or append_no_harm_diagnostics.get("rho_miss_delta_stay_minus_selected") is None
                            else float(append_no_harm_diagnostics["rho_miss_delta_stay_minus_selected"])
                        ),
                        "append_no_harm_step_gain_delta": (
                            None
                            if append_no_harm_diagnostics is None
                            or append_no_harm_diagnostics.get("step_gain_delta_selected_minus_stay") is None
                            else float(append_no_harm_diagnostics["step_gain_delta_selected_minus_stay"])
                        ),
                        "append_no_harm_step_residual_ratio": (
                            None
                            if append_no_harm_diagnostics is None
                            or append_no_harm_diagnostics.get("step_residual_ratio_selected_vs_stay") is None
                            else float(append_no_harm_diagnostics["step_residual_ratio_selected_vs_stay"])
                        ),
                        "append_no_harm_displacement_ratio": (
                            None
                            if append_no_harm_diagnostics is None
                            or append_no_harm_diagnostics.get("displacement_ratio_selected_vs_stay") is None
                            else float(append_no_harm_diagnostics["displacement_ratio_selected_vs_stay"])
                        ),
                        "append_no_harm_diagnostics": append_no_harm_diagnostics,
                        "append_no_harm_exact_logging": append_no_harm_exact_logging,
                        "forecast_stay_fidelity_exact_next": (
                            None
                            if not isinstance(forecast_stay, Mapping)
                            or forecast_stay.get("fidelity_exact_next") is None
                            else float(forecast_stay.get("fidelity_exact_next"))
                        ),
                        "forecast_selected_fidelity_exact_next": (
                            None
                            if not isinstance(forecast_selected, Mapping)
                            or forecast_selected.get("fidelity_exact_next") is None
                            else float(forecast_selected.get("fidelity_exact_next"))
                        ),
                        "forecast_stay_abs_energy_total_error_next": (
                            None
                            if not isinstance(forecast_stay, Mapping)
                            or forecast_stay.get("abs_energy_total_error_next") is None
                            else float(forecast_stay.get("abs_energy_total_error_next"))
                        ),
                        "forecast_selected_abs_energy_total_error_next": (
                            None
                            if not isinstance(forecast_selected, Mapping)
                            or forecast_selected.get("abs_energy_total_error_next") is None
                            else float(forecast_selected.get("abs_energy_total_error_next"))
                        ),
                        "forecast_stay_abs_primary_density_error_next": (
                            None
                            if not isinstance(forecast_stay, Mapping)
                            or forecast_stay.get("abs_primary_density_error_next") is None
                            else float(forecast_stay.get("abs_primary_density_error_next"))
                        ),
                        "forecast_selected_abs_primary_density_error_next": (
                            None
                            if not isinstance(forecast_selected, Mapping)
                            or forecast_selected.get("abs_primary_density_error_next") is None
                            else float(forecast_selected.get("abs_primary_density_error_next"))
                        ),
                        "forecast_stay_abs_primary_density_slope_error_next": (
                            None
                            if not isinstance(forecast_stay, Mapping)
                            or forecast_stay.get("abs_primary_density_slope_error_next") is None
                            else float(forecast_stay.get("abs_primary_density_slope_error_next"))
                        ),
                        "forecast_selected_abs_primary_density_slope_error_next": (
                            None
                            if not isinstance(forecast_selected, Mapping)
                            or forecast_selected.get("abs_primary_density_slope_error_next") is None
                            else float(forecast_selected.get("abs_primary_density_slope_error_next"))
                        ),
                        "forecast_stay_abs_staggered_error_next": (
                            None
                            if not isinstance(forecast_stay, Mapping)
                            or forecast_stay.get("abs_staggered_error_next") is None
                            else float(forecast_stay.get("abs_staggered_error_next"))
                        ),
                        "forecast_selected_abs_staggered_error_next": (
                            None
                            if not isinstance(forecast_selected, Mapping)
                            or forecast_selected.get("abs_staggered_error_next") is None
                            else float(forecast_selected.get("abs_staggered_error_next"))
                        ),
                        "forecast_stay_abs_doublon_error_next": (
                            None
                            if not isinstance(forecast_stay, Mapping)
                            or forecast_stay.get("abs_doublon_error_next") is None
                            else float(forecast_stay.get("abs_doublon_error_next"))
                        ),
                        "forecast_selected_abs_doublon_error_next": (
                            None
                            if not isinstance(forecast_selected, Mapping)
                            or forecast_selected.get("abs_doublon_error_next") is None
                            else float(forecast_selected.get("abs_doublon_error_next"))
                        ),
                        "forecast_stay_site_occupations_abs_error_max_next": (
                            None
                            if not isinstance(forecast_stay, Mapping)
                            or forecast_stay.get("site_occupations_abs_error_max_next") is None
                            else float(forecast_stay.get("site_occupations_abs_error_max_next"))
                        ),
                        "forecast_selected_site_occupations_abs_error_max_next": (
                            None
                            if not isinstance(forecast_selected, Mapping)
                            or forecast_selected.get("site_occupations_abs_error_max_next") is None
                            else float(forecast_selected.get("site_occupations_abs_error_max_next"))
                        ),
                        "predicted_displacement": float(predicted_displacement),
                        "motion_regime": str(motion_telemetry.regime),
                        "motion_direction_cosine": (
                            None if motion_telemetry.direction_cosine is None else float(motion_telemetry.direction_cosine)
                        ),
                        "motion_rate_change_ratio": (
                            None if motion_telemetry.rate_change_ratio is None else float(motion_telemetry.rate_change_ratio)
                        ),
                        "motion_acceleration_l2": (
                            None if motion_telemetry.acceleration_l2 is None else float(motion_telemetry.acceleration_l2)
                        ),
                        "motion_curvature_cosine": (
                            None if motion_telemetry.curvature_cosine is None else float(motion_telemetry.curvature_cosine)
                        ),
                        "motion_direction_reversal": bool(motion_telemetry.direction_reversal),
                        "motion_curvature_sign_flip": bool(motion_telemetry.curvature_sign_flip),
                        "motion_kink_score": float(motion_telemetry.kink_score),
                        "integrator_policy": str(integrator_diagnostics["integrator_policy"]),
                        "integrator_used": str(integrator_diagnostics["integrator_used"]),
                        "integrator_columnarity": integrator_diagnostics.get("integrator_columnarity"),
                        "integrator_curvature": integrator_diagnostics.get("integrator_curvature"),
                        "integrator_euler_fs_error": integrator_diagnostics.get("integrator_euler_fs_error"),
                        "integrator_condition_number": integrator_diagnostics.get("integrator_condition_number"),
                        "integrator_condition_pass": integrator_diagnostics.get("integrator_condition_pass"),
                        "integrator_geometry_gate_pass": integrator_diagnostics.get("integrator_geometry_gate_pass"),
                        "integrator_euler_error_pass": integrator_diagnostics.get("integrator_euler_error_pass"),
                        "integrator_auto_policy_schema": integrator_diagnostics.get("integrator_auto_policy_schema"),
                        "integrator_auto_admit_euler": integrator_diagnostics.get("integrator_auto_admit_euler"),
                        "integrator_euler_blockers": list(integrator_diagnostics.get("integrator_euler_blockers") or []),
                        "integrator_rho_miss_pass": integrator_diagnostics.get("integrator_rho_miss_pass"),
                        "integrator_time_fraction": integrator_diagnostics.get("integrator_time_fraction"),
                        "integrator_euler_min_time_fraction": integrator_diagnostics.get("integrator_euler_min_time_fraction"),
                        "integrator_euler_time_gate_pass": integrator_diagnostics.get("integrator_euler_time_gate_pass"),
                        "integrator_euler_observable_gate_pass": integrator_diagnostics.get("integrator_euler_observable_gate_pass"),
                        "integrator_euler_site_span": integrator_diagnostics.get("integrator_euler_site_span"),
                        "integrator_euler_primary_density_span": integrator_diagnostics.get("integrator_euler_primary_density_span"),
                        "integrator_euler_energy_span": integrator_diagnostics.get("integrator_euler_energy_span"),
                        "integrator_error": integrator_diagnostics.get("integrator_error"),
                        "temporal_refresh_pressure": str(refresh_pressure),
                        "oracle_confirm_limit": int(oracle_confirm_limit),
                        "oracle_budget_scale": float(oracle_budget_scale),
                        "rho_miss": float(baseline_for_decision["summary"].rho_miss),
                        "rho_real": float(baseline_for_decision["summary"].rho_real),
                        "rho_num": float(baseline_for_decision["summary"].rho_num),
                        "epsilon_proj_sq": float(baseline_for_decision["summary"].epsilon_proj_sq),
                        "epsilon_step_sq": float(baseline_for_decision["summary"].epsilon_step_sq),
                        "theta_dot_l2": float(theta_dot_l2),
                        "theta_update_l2": theta_update_l2,
                        "energy_total": float(energy_controller),
                        "energy_total_controller": float(energy_controller),
                        "energy_total_exact": (
                            None if energy_exact is None else float(energy_exact)
                        ),
                        "abs_energy_total_error": (
                            None
                            if abs_energy_total_error is None
                            else float(abs_energy_total_error)
                        ),
                        "fidelity_exact": (
                            None if fidelity_exact is None else float(fidelity_exact)
                        ),
                        "fidelity_initial_controller": float(fidelity_initial_controller),
                        "fidelity_initial_exact": (
                            None
                            if fidelity_initial_exact is None
                            else float(fidelity_initial_exact)
                        ),
                        "primary_density_mode": str(primary_density_mode),
                        "primary_density": float(primary_density_controller),
                        "primary_density_exact": (
                            None
                            if primary_density_exact is None
                            else float(primary_density_exact)
                        ),
                        "abs_primary_density_error": (
                            None
                            if abs_primary_density_error is None
                            else float(abs_primary_density_error)
                        ),
                        "staggered": float(controller_obs["staggered"]),
                        "staggered_exact": (
                            None if exact_obs is None else float(exact_obs["staggered"])
                        ),
                        "abs_staggered_error": (
                            None
                            if abs_staggered_error is None
                            else float(abs_staggered_error)
                        ),
                        "doublon": float(controller_obs["doublon"]),
                        "doublon_exact": (
                            None if exact_obs is None else float(exact_obs["doublon"])
                        ),
                        "abs_doublon_error": (
                            None if abs_doublon_error is None else float(abs_doublon_error)
                        ),
                        "site_occupations": list(controller_obs["site_occupations"]),
                        "site_occupations_exact": (
                            None if exact_obs is None else list(exact_obs["site_occupations"])
                        ),
                        "site_occupations_up": list(controller_obs["n_up_site"]),
                        "site_occupations_up_exact": (
                            None if exact_obs is None else list(exact_obs["n_up_site"])
                        ),
                        "site_occupations_dn": list(controller_obs["n_dn_site"]),
                        "site_occupations_dn_exact": (
                            None if exact_obs is None else list(exact_obs["n_dn_site"])
                        ),
                        "site_occupations_abs_error": (
                            None
                            if site_occ_abs_error is None
                            else [float(x) for x in site_occ_abs_error.tolist()]
                        ),
                        "site_occupations_abs_error_max": (
                            None
                            if site_occ_abs_error is None or site_occ_abs_error.size <= 0
                            else float(np.max(site_occ_abs_error))
                        ),
                        **{
                            str(key): value
                            for key, value in controller_obs.items()
                            if str(key)
                            not in {
                                "n_up_site",
                                "n_dn_site",
                                "site_occupations",
                                "doublon",
                                "staggered",
                            }
                        },
                        "logical_block_count": int(logical_before),
                        "runtime_parameter_count": int(runtime_before),
                        "runtime_parameter_count_before": int(runtime_before),
                        "runtime_parameter_count_after": int(runtime_after_planned),
                        "runtime_parameter_count_delta": int(runtime_after_planned) - int(runtime_before),
                        "selected_noisy_energy_mean": oracle_commit_payload.get("selected_noisy_energy_mean", None),
                        "selected_noisy_energy_stderr": oracle_commit_payload.get("selected_noisy_energy_stderr", None),
                        "selected_noisy_backend_info": oracle_commit_payload.get("selected_noisy_backend_info", None),
                        "stay_noisy_energy_mean": oracle_commit_payload.get("stay_noisy_energy_mean", None),
                        "stay_noisy_energy_stderr": oracle_commit_payload.get("stay_noisy_energy_stderr", None),
                        "stay_noisy_backend_info": oracle_commit_payload.get("stay_noisy_backend_info", None),
                        "baseline_backend_info": dict(baseline_for_decision.get("backend_info", {})),
                        "selected_noisy_improvement_abs": oracle_commit_payload.get("selected_noisy_improvement_abs", None),
                        "selected_noisy_improvement_ratio": oracle_commit_payload.get("selected_noisy_improvement_ratio", None),
                        "selected_prune_cached_loss": selected_prune_cached_loss,
                        **dict(selected_prune_loss_fields),
                        "selected_prune_stagnation_score": selected_prune_stagnation_score,
                        "selected_post_prune_state_jump_l2": selected_post_prune_state_jump_l2,
                        "selected_prune_origin_kind": selected_prune_origin_kind,
                        "selected_prune_age_checkpoints": selected_prune_age_checkpoints,
                        "selected_prune_block_theta_dot_norm": selected_prune_block_theta_dot_norm,
                        "selected_prune_block_theta_dot_rel": selected_prune_block_theta_dot_rel,
                        "selected_prune_appended_origin_bias_factor": selected_prune_appended_origin_bias_factor,
                        "selected_prune_appended_origin_bias_applied": selected_prune_appended_origin_bias_applied,
                        **drive_diagnostics,
                        "degraded_reason": degraded_reason,
                        "baseline_geometry": dataclass_to_payload(baseline_for_decision["summary"]),
                        "candidate_pool_diagnostics": candidate_pool_diagnostics,
                        "raw_scout_record_count": int(len(scout_records)),
                        "shortlisted_candidate_count": int(len(shortlist)),
                        "confirmed_candidate_count": int(len(confirmed)),
                        "shortlist": shortlist_payload,
                        "confirmed": confirmed_payload,
                        "prune_candidates": prune_candidates_payload,
                    }
                )

                if str(action_kind) == "append_candidate" and selected is not None:
                    tier_reached = "commit"
                    candidate_data = dict(selected["candidate_data"])
                    self.current_terms = list(candidate_data["aug_terms"])
                    self.current_layout = candidate_data["aug_layout"]
                    self.current_executor = candidate_data["aug_executor"]
                    if commit_theta_next is None or commit_theta_dot is None:
                        raise RuntimeError("append integrator plan was not prepared")
                    self.current_theta = np.asarray(commit_theta_next, dtype=float).reshape(-1)
                    self._append_counter += 1
                    self._previous_append_position = int(selected_position_id)
                    self._planning_audit.commit(planning_group_keys_for_term(selected["candidate_term"]))
                    appended_carrier = selected["candidate_data"].get("candidate_carrier")
                    appended_label = str(
                        selected_candidate_label
                        if appended_carrier is None
                        else getattr(appended_carrier, "label", selected_candidate_label)
                    )
                    self._block_birth_checkpoint[appended_label] = int(checkpoint_index)
                    self._block_cooldown[appended_label] = 0
                    self._block_burden[appended_label] = float(selected["candidate_summary"].compile_proxy_total)
                    self._block_origin[appended_label] = "append"
                    self._block_motion_history.setdefault(appended_label, [])
                    self._block_fit_history.setdefault(appended_label, [])
                    self._record_theta_dot_history(
                        np.asarray(commit_theta_dot, dtype=float).reshape(-1)
                    )
                elif str(action_kind) == "repair_miss":
                    tier_reached = "repair"
                elif str(action_kind) == "prune_coordinate" and selected is not None:
                    tier_reached = "commit"
                    reduced_state = dict(selected["reduced_state"])
                    reduced_baseline = dict(selected["pruned_baseline"])
                    removed_label = str(reduced_state["removed_label"])
                    reduced_psi = np.asarray(reduced_state["reduced_psi"], dtype=complex).reshape(-1)
                    reduced_obs = self._observable_snapshot(reduced_psi)
                    post_prune_psi = np.asarray(reduced_psi, dtype=complex).reshape(-1)
                    post_prune_energy_total, _, _ = self._energy_hpsi_variance(
                        reduced_psi,
                        compiled_h=step_hamiltonian.compiled_h,
                    )
                    post_prune_payload = {
                        "post_prune_energy_total": float(post_prune_energy_total),
                        "post_prune_fidelity_exact": None,
                        "post_prune_abs_energy_total_error": None,
                        "post_prune_staggered": float(reduced_obs["staggered"]),
                        "post_prune_abs_staggered_error": None,
                        "post_prune_doublon": float(reduced_obs["doublon"]),
                        "post_prune_abs_doublon_error": None,
                        "post_prune_site_occupations": [float(x) for x in np.asarray(reduced_obs["site_occupations"], dtype=float).tolist()],
                        "post_prune_site_occupations_abs_error": None,
                        "post_prune_site_occupations_abs_error_max": None,
                        "post_prune_baseline_geometry": dataclass_to_payload(reduced_baseline["summary"]),
                    }
                    self._record_compile_audit_prune_event(
                        checkpoint_index=int(checkpoint_index),
                        time_value=float(time_value),
                        selected_candidate_label=selected_candidate_label,
                        removed_label=str(removed_label),
                        logical_before=int(logical_before),
                        runtime_before=int(runtime_before),
                        reduced_state=reduced_state,
                    )
                    self.current_terms = list(reduced_state["reduced_terms"])
                    self.current_layout = reduced_state["reduced_layout"]
                    self.current_executor = reduced_state["reduced_executor"]
                    if commit_theta_next is None or commit_theta_dot is None:
                        raise RuntimeError("prune integrator plan was not prepared")
                    self.current_theta = np.asarray(commit_theta_next, dtype=float).reshape(-1)
                    self._planning_audit = reduced_state["reduced_planning_audit"]
                    self._block_birth_checkpoint.pop(removed_label, None)
                    self._block_cooldown.pop(removed_label, None)
                    self._block_burden.pop(removed_label, None)
                    self._block_origin.pop(removed_label, None)
                    self._block_motion_history.pop(removed_label, None)
                    self._block_fit_history.pop(removed_label, None)
                    self._previous_block_theta_snapshot.pop(removed_label, None)
                    self._record_theta_dot_history(
                        np.asarray(commit_theta_dot, dtype=float).reshape(-1)
                    )
                else:
                    if commit_theta_next is None or commit_theta_dot is None:
                        raise RuntimeError("stay integrator plan was not prepared")
                    self.current_theta = np.asarray(commit_theta_next, dtype=float).reshape(-1)
                    self._record_theta_dot_history(
                        np.asarray(commit_theta_dot, dtype=float).reshape(-1)
                    )
                    if shortlist or prune_candidates:
                        tier_reached = "confirm"
                self._set_previous_block_theta_snapshot()
                if post_prune_payload is not None and self._trajectory:
                    self._trajectory[-1].update(post_prune_payload)

                ledger_entry = CheckpointLedgerEntry(
                    checkpoint_index=int(checkpoint_index),
                    time=float(time_value),
                    physical_time=float(step_hamiltonian.physical_time),
                    action_kind=str(action_kind),
                    trajectory_sample_kind=(
                        "repair_event" if str(action_kind) == "repair_miss" else "state_sample"
                    ),
                    advances_time=bool(str(action_kind) != "repair_miss"),
                    repair_attempt_index=int(repair_attempt.attempt_index),
                    repair_max_attempts=repair_attempt.max_attempts,
                    repair_escalation_kind=repair_attempt.escalation_kind,
                    repair_retry_next=bool(repair_retry_next),
                    repair_terminal=bool(repair_terminal),
                    repair_failure_reason=repair_failure_reason,
                    accepted_after_repair=bool(
                        (str(action_kind) != "repair_miss" and int(repair_attempt.attempt_index) > 0)
                        or bool(repair_rescue_admitted)
                    ),
                    repair_no_admit_diagnostics=repair_no_admit_diagnostics,
                    repair_rescue_candidate_label=repair_rescue_candidate_label,
                    repair_rescue_reason=repair_rescue_reason,
                    repair_rescue_admitted=bool(repair_rescue_admitted),
                    high_miss_no_admit_soft_fallback=bool(high_miss_no_admit_soft_fallback),
                    high_miss_no_admit_soft_fallback_policy=high_miss_no_admit_soft_fallback_policy,
                    high_miss_no_admit_soft_fallback_reason=high_miss_no_admit_soft_fallback_reason,
                    high_miss_no_admit_soft_fallback_warning=high_miss_no_admit_soft_fallback_warning,
                    candidate_label=selected_candidate_label,
                    proposed_action_kind=str(proposed_action_kind),
                    proposed_candidate_label=proposed_candidate_label,
                    controller_lane=str(controller_lane),
                    controller_lane_reason=str(controller_lane_reason),
                    position_id=selected_position_id,
                    rho_miss=float(baseline_for_decision["summary"].rho_miss),
                    rho_real=float(baseline_for_decision["summary"].rho_real),
                    rho_num=float(baseline_for_decision["summary"].rho_num),
                    gain_ratio_selected=float(selected_gain_ratio),
                    prune_cached_loss_selected=(None if selected_prune_cached_loss is None else float(selected_prune_cached_loss)),
                    **dict(selected_prune_loss_fields),
                    prune_stagnation_score_selected=(None if selected_prune_stagnation_score is None else float(selected_prune_stagnation_score)),
                    post_prune_state_jump_l2=(None if selected_post_prune_state_jump_l2 is None else float(selected_post_prune_state_jump_l2)),
                    prune_origin_kind_selected=selected_prune_origin_kind,
                    prune_age_checkpoints_selected=selected_prune_age_checkpoints,
                    prune_block_theta_dot_norm_selected=(
                        None
                        if selected_prune_block_theta_dot_norm is None
                        else float(selected_prune_block_theta_dot_norm)
                    ),
                    prune_block_theta_dot_rel_selected=(
                        None
                        if selected_prune_block_theta_dot_rel is None
                        else float(selected_prune_block_theta_dot_rel)
                    ),
                    prune_appended_origin_bias_factor_selected=(None if selected_prune_appended_origin_bias_factor is None else float(selected_prune_appended_origin_bias_factor)),
                    prune_appended_origin_bias_applied_selected=selected_prune_appended_origin_bias_applied,
                    prune_schur_raw_loss_selected=(None if selected_prune_schur_raw_loss is None else float(selected_prune_schur_raw_loss)),
                    prune_schur_normalized_loss_selected=(None if selected_prune_schur_normalized_loss is None else float(selected_prune_schur_normalized_loss)),
                    prune_schur_selected_rung=(None if selected_prune_schur_selected_rung is None else int(selected_prune_schur_selected_rung)),
                    prune_schur_monotonicity_status_selected=selected_prune_schur_monotonicity_status,
                    prune_differential_miss_selected=(None if selected_prune_differential_miss is None else float(selected_prune_differential_miss)),
                    prune_permit_path_selected=selected_prune_permit_path,
                    prune_projection_objective_selected=(None if selected_prune_projection_objective is None else float(selected_prune_projection_objective)),
                    prune_projected_state_jump_l2_selected=(None if selected_prune_projected_state_jump_l2 is None else float(selected_prune_projected_state_jump_l2)),
                    prune_ray_distance_selected=(None if selected_prune_ray_distance is None else float(selected_prune_ray_distance)),
                    prune_shadow_score_selected=(None if selected_prune_shadow_score is None else float(selected_prune_shadow_score)),
                    prune_persistence_count_selected=(None if selected_prune_persistence_count is None else int(selected_prune_persistence_count)),
                    prune_persistence_required_selected=(None if selected_prune_persistence_required is None else int(selected_prune_persistence_required)),
                    prune_persistence_passed_selected=selected_prune_persistence_passed,
                    integrator_policy=str(integrator_diagnostics["integrator_policy"]),
                    integrator_used=str(integrator_diagnostics["integrator_used"]),
                    integrator_columnarity=integrator_diagnostics.get("integrator_columnarity"),
                    integrator_curvature=integrator_diagnostics.get("integrator_curvature"),
                    integrator_euler_fs_error=integrator_diagnostics.get("integrator_euler_fs_error"),
                    integrator_geometry_gate_pass=integrator_diagnostics.get("integrator_geometry_gate_pass"),
                    integrator_euler_error_pass=integrator_diagnostics.get("integrator_euler_error_pass"),
                    integrator_auto_policy_schema=integrator_diagnostics.get("integrator_auto_policy_schema"),
                    integrator_auto_admit_euler=integrator_diagnostics.get("integrator_auto_admit_euler"),
                    integrator_euler_blockers=list(integrator_diagnostics.get("integrator_euler_blockers") or []),
                    integrator_condition_number=integrator_diagnostics.get("integrator_condition_number"),
                    integrator_condition_pass=integrator_diagnostics.get("integrator_condition_pass"),
                    integrator_rho_miss_pass=integrator_diagnostics.get("integrator_rho_miss_pass"),
                    integrator_time_fraction=integrator_diagnostics.get("integrator_time_fraction"),
                    integrator_euler_min_time_fraction=integrator_diagnostics.get("integrator_euler_min_time_fraction"),
                    integrator_euler_time_gate_pass=integrator_diagnostics.get("integrator_euler_time_gate_pass"),
                    integrator_euler_observable_gate_pass=integrator_diagnostics.get("integrator_euler_observable_gate_pass"),
                    integrator_euler_site_span=integrator_diagnostics.get("integrator_euler_site_span"),
                    integrator_euler_primary_density_span=integrator_diagnostics.get("integrator_euler_primary_density_span"),
                    integrator_euler_energy_span=integrator_diagnostics.get("integrator_euler_energy_span"),
                    integrator_error=integrator_diagnostics.get("integrator_error"),
                    shortlist_size=int(len(shortlist)),
                    tier_reached=str(tier_reached),
                    logical_block_count_before=int(logical_before),
                    logical_block_count_after=int(self.current_layout.logical_parameter_count),
                    runtime_parameter_count_before=int(runtime_before),
                    runtime_parameter_count_after=int(self.current_layout.runtime_parameter_count),
                    rate_change_l2=(None if rate_change_l2 is None else float(rate_change_l2)),
                    theta_dot_l2=float(theta_dot_l2),
                    theta_update_l2=theta_update_l2,
                    observable_family=str(controller_obs.get("observable_family", self._family_key)),
                    primary_density_mode=str(primary_density_mode),
                    drive_enabled=bool(drive_diagnostics.get("drive_enabled", False)),
                    drive_operator_label=drive_diagnostics.get("drive_operator_label"),
                    drive_family_key=drive_diagnostics.get("drive_family_key"),
                    drive_coefficient=drive_diagnostics.get("drive_coefficient"),
                    drive_coefficient_linf=drive_diagnostics.get("drive_coefficient_linf"),
                    motion_regime=str(motion_telemetry.regime),
                    motion_direction_cosine=(None if motion_telemetry.direction_cosine is None else float(motion_telemetry.direction_cosine)),
                    motion_rate_change_ratio=(None if motion_telemetry.rate_change_ratio is None else float(motion_telemetry.rate_change_ratio)),
                    motion_acceleration_l2=(None if motion_telemetry.acceleration_l2 is None else float(motion_telemetry.acceleration_l2)),
                    motion_curvature_cosine=(None if motion_telemetry.curvature_cosine is None else float(motion_telemetry.curvature_cosine)),
                    motion_direction_reversal=bool(motion_telemetry.direction_reversal),
                    motion_curvature_sign_flip=bool(motion_telemetry.curvature_sign_flip),
                    motion_kink_score=float(motion_telemetry.kink_score),
                    exact_cache_hits=int(cache.summary()["hits"]),
                    exact_cache_misses=int(cache.summary()["misses"]),
                    geometry_memo_hits=int(geometry_memo.summary()["hits"]),
                    geometry_memo_misses=int(geometry_memo.summary()["misses"]),
                    planning_groups_new_selected=float(selected_groups_new),
                    energy_total_controller=float(energy_controller),
                    energy_total_exact=(None if energy_exact is None else float(energy_exact)),
                    abs_energy_total_error=(
                        None if abs_energy_total_error is None else float(abs_energy_total_error)
                    ),
                    fidelity_exact=(None if fidelity_exact is None else float(fidelity_exact)),
                    requested_mode=str(self.cfg.mode),
                    decision_backend=str(decision_backend),
                    decision_noise_mode=decision_noise_mode,
                    **decision_flow_fields,
                    oracle_decision_used=bool(oracle_decision_used),
                    oracle_attempted=bool(oracle_attempted),
                    oracle_estimate_kind=oracle_estimate_kind,
                    selection_metric=str(selection_metric),
                    decision_override_reason=decision_override_reason,
                    selection_reason=selection_reason,
                    forecast_mode=forecast_mode,
                    forecast_error=forecast_error,
                    exact_forecast_error=exact_forecast_error,
                    forecast_stay_score_total=forecast_stay_score_total,
                    forecast_selected_score_total=forecast_selected_score_total,
                    forecast_score_delta_vs_stay=forecast_score_delta_vs_stay,
                    forecast_score_interpretation="lower_is_better",
                    forecast_selected_lower_than_stay=forecast_selected_lower_than_stay,
                    baseline_step_scale=baseline_step_scale,
                    baseline_blend_weight=baseline_blend_weight,
                    baseline_gain_scale=baseline_gain_scale,
                    baseline_proposal_kind=(
                        None if baseline_proposal_kind is None else str(baseline_proposal_kind)
                    ),
                    selected_step_scale=selected_step_scale,
                    forecast_stay_rho_miss_next=(
                        None
                        if not isinstance(forecast_stay, Mapping)
                        or forecast_stay.get("rho_miss_next") is None
                        else float(forecast_stay.get("rho_miss_next"))
                    ),
                    forecast_selected_rho_miss_next=(
                        None
                        if not isinstance(forecast_selected, Mapping)
                        or forecast_selected.get("rho_miss_next") is None
                        else float(forecast_selected.get("rho_miss_next"))
                    ),
                    forecast_stay_step_gain_ratio_next=(
                        None
                        if not isinstance(forecast_stay, Mapping)
                        or forecast_stay.get("step_gain_ratio_next") is None
                        else float(forecast_stay.get("step_gain_ratio_next"))
                    ),
                    forecast_selected_step_gain_ratio_next=(
                        None
                        if not isinstance(forecast_selected, Mapping)
                        or forecast_selected.get("step_gain_ratio_next") is None
                        else float(forecast_selected.get("step_gain_ratio_next"))
                    ),
                    forecast_stay_condition_number_next=(
                        None
                        if not isinstance(forecast_stay, Mapping)
                        or forecast_stay.get("condition_number_next") is None
                        else float(forecast_stay.get("condition_number_next"))
                    ),
                    forecast_selected_condition_number_next=(
                        None
                        if not isinstance(forecast_selected, Mapping)
                        or forecast_selected.get("condition_number_next") is None
                        else float(forecast_selected.get("condition_number_next"))
                    ),
                    forecast_stay_predicted_displacement_next=(
                        None
                        if forecast_stay_predicted_displacement_next is None
                        else float(forecast_stay_predicted_displacement_next)
                    ),
                    forecast_selected_predicted_displacement_next=(
                        None
                        if forecast_selected_predicted_displacement_next is None
                        else float(forecast_selected_predicted_displacement_next)
                    ),
                    forecast_stay_epsilon_step_ratio_next=(
                        None
                        if forecast_stay_epsilon_step_ratio_next is None
                        else float(forecast_stay_epsilon_step_ratio_next)
                    ),
                    forecast_selected_epsilon_step_ratio_next=(
                        None
                        if forecast_selected_epsilon_step_ratio_next is None
                        else float(forecast_selected_epsilon_step_ratio_next)
                    ),
                    append_no_harm_veto_reason=append_no_harm_veto_reason,
                    append_no_harm_condition_ratio=(
                        None
                        if append_no_harm_diagnostics is None
                        or append_no_harm_diagnostics.get("condition_ratio_selected_vs_stay") is None
                        else float(append_no_harm_diagnostics["condition_ratio_selected_vs_stay"])
                    ),
                    append_no_harm_rho_miss_delta=(
                        None
                        if append_no_harm_diagnostics is None
                        or append_no_harm_diagnostics.get("rho_miss_delta_stay_minus_selected") is None
                        else float(append_no_harm_diagnostics["rho_miss_delta_stay_minus_selected"])
                    ),
                    append_no_harm_step_gain_delta=(
                        None
                        if append_no_harm_diagnostics is None
                        or append_no_harm_diagnostics.get("step_gain_delta_selected_minus_stay") is None
                        else float(append_no_harm_diagnostics["step_gain_delta_selected_minus_stay"])
                    ),
                    append_no_harm_step_residual_ratio=(
                        None
                        if append_no_harm_diagnostics is None
                        or append_no_harm_diagnostics.get("step_residual_ratio_selected_vs_stay") is None
                        else float(append_no_harm_diagnostics["step_residual_ratio_selected_vs_stay"])
                    ),
                    append_no_harm_displacement_ratio=(
                        None
                        if append_no_harm_diagnostics is None
                        or append_no_harm_diagnostics.get("displacement_ratio_selected_vs_stay") is None
                        else float(append_no_harm_diagnostics["displacement_ratio_selected_vs_stay"])
                    ),
                    append_no_harm_diagnostics=append_no_harm_diagnostics,
                    append_no_harm_exact_logging=append_no_harm_exact_logging,
                    forecast_stay_fidelity_exact_next=(
                        None
                        if not isinstance(forecast_stay, Mapping)
                        or forecast_stay.get("fidelity_exact_next") is None
                        else float(forecast_stay.get("fidelity_exact_next"))
                    ),
                    forecast_selected_fidelity_exact_next=(
                        None
                        if not isinstance(forecast_selected, Mapping)
                        or forecast_selected.get("fidelity_exact_next") is None
                        else float(forecast_selected.get("fidelity_exact_next"))
                    ),
                    forecast_stay_abs_energy_total_error_next=(
                        None
                        if not isinstance(forecast_stay, Mapping)
                        or forecast_stay.get("abs_energy_total_error_next") is None
                        else float(forecast_stay.get("abs_energy_total_error_next"))
                    ),
                    forecast_selected_abs_energy_total_error_next=(
                        None
                        if not isinstance(forecast_selected, Mapping)
                        or forecast_selected.get("abs_energy_total_error_next") is None
                        else float(forecast_selected.get("abs_energy_total_error_next"))
                    ),
                    forecast_stay_abs_primary_density_error_next=(
                        None
                        if not isinstance(forecast_stay, Mapping)
                        or forecast_stay.get("abs_primary_density_error_next") is None
                        else float(forecast_stay.get("abs_primary_density_error_next"))
                    ),
                    forecast_selected_abs_primary_density_error_next=(
                        None
                        if not isinstance(forecast_selected, Mapping)
                        or forecast_selected.get("abs_primary_density_error_next") is None
                        else float(forecast_selected.get("abs_primary_density_error_next"))
                    ),
                    forecast_stay_abs_primary_density_slope_error_next=(
                        None
                        if not isinstance(forecast_stay, Mapping)
                        or forecast_stay.get("abs_primary_density_slope_error_next") is None
                        else float(forecast_stay.get("abs_primary_density_slope_error_next"))
                    ),
                    forecast_selected_abs_primary_density_slope_error_next=(
                        None
                        if not isinstance(forecast_selected, Mapping)
                        or forecast_selected.get("abs_primary_density_slope_error_next") is None
                        else float(forecast_selected.get("abs_primary_density_slope_error_next"))
                    ),
                    forecast_stay_abs_staggered_error_next=(
                        None
                        if not isinstance(forecast_stay, Mapping)
                        or forecast_stay.get("abs_staggered_error_next") is None
                        else float(forecast_stay.get("abs_staggered_error_next"))
                    ),
                    forecast_selected_abs_staggered_error_next=(
                        None
                        if not isinstance(forecast_selected, Mapping)
                        or forecast_selected.get("abs_staggered_error_next") is None
                        else float(forecast_selected.get("abs_staggered_error_next"))
                    ),
                    forecast_stay_abs_doublon_error_next=(
                        None
                        if not isinstance(forecast_stay, Mapping)
                        or forecast_stay.get("abs_doublon_error_next") is None
                        else float(forecast_stay.get("abs_doublon_error_next"))
                    ),
                    forecast_selected_abs_doublon_error_next=(
                        None
                        if not isinstance(forecast_selected, Mapping)
                        or forecast_selected.get("abs_doublon_error_next") is None
                        else float(forecast_selected.get("abs_doublon_error_next"))
                    ),
                    forecast_stay_site_occupations_abs_error_max_next=(
                        None
                        if not isinstance(forecast_stay, Mapping)
                        or forecast_stay.get("site_occupations_abs_error_max_next") is None
                        else float(forecast_stay.get("site_occupations_abs_error_max_next"))
                    ),
                    forecast_selected_site_occupations_abs_error_max_next=(
                        None
                        if not isinstance(forecast_selected, Mapping)
                        or forecast_selected.get("site_occupations_abs_error_max_next") is None
                        else float(forecast_selected.get("site_occupations_abs_error_max_next"))
                    ),
                    predicted_displacement=float(predicted_displacement),
                    temporal_refresh_pressure=str(refresh_pressure),
                    selected_noisy_energy_mean=(None if oracle_commit_payload.get("selected_noisy_energy_mean", None) is None else float(oracle_commit_payload["selected_noisy_energy_mean"])),
                    selected_noisy_energy_stderr=(None if oracle_commit_payload.get("selected_noisy_energy_stderr", None) is None else float(oracle_commit_payload["selected_noisy_energy_stderr"])),
                    stay_noisy_energy_mean=(None if oracle_commit_payload.get("stay_noisy_energy_mean", None) is None else float(oracle_commit_payload["stay_noisy_energy_mean"])),
                    stay_noisy_energy_stderr=(None if oracle_commit_payload.get("stay_noisy_energy_stderr", None) is None else float(oracle_commit_payload["stay_noisy_energy_stderr"])),
                    selected_noisy_improvement_abs=(None if oracle_commit_payload.get("selected_noisy_improvement_abs", None) is None else float(oracle_commit_payload["selected_noisy_improvement_abs"])),
                    selected_noisy_improvement_ratio=(None if oracle_commit_payload.get("selected_noisy_improvement_ratio", None) is None else float(oracle_commit_payload["selected_noisy_improvement_ratio"])),
                    oracle_confirm_limit=int(oracle_confirm_limit),
                    oracle_budget_scale=float(oracle_budget_scale),
                    oracle_cache_hits=(0 if oracle_cache is None else int(oracle_cache.summary()["hits"])),
                    oracle_cache_misses=(0 if oracle_cache is None else int(oracle_cache.summary()["misses"])),
                    raw_group_cache_hits=(0 if raw_group_pool is None else int(raw_group_pool.summary()["hits"])),
                    raw_group_cache_misses=(0 if raw_group_pool is None else int(raw_group_pool.summary()["misses"])),
                    raw_group_cache_extensions=(0 if raw_group_pool is None else int(raw_group_pool.summary()["extensions"])),
                    drive_term_count=int(step_hamiltonian.drive_term_count),
                    analytic_noise_std=float(self.cfg.analytic_noise_std),
                    analytic_noise_seed=getattr(self.cfg, "analytic_noise_seed", None),
                    degraded_reason=degraded_reason,
                )
                self._ledger.append(dataclass_to_payload(ledger_entry))
                if checkpoint_observer is not None:
                    observer_result = checkpoint_observer.on_checkpoint(
                        {
                            "checkpoint_index": int(checkpoint_index),
                            "time": float(time_value),
                            "time_stop": (None if time_stop is None else float(time_stop)),
                            "physical_time": float(step_hamiltonian.physical_time),
                            "step_hamiltonian": step_hamiltonian,
                            "layout_at_checkpoint": layout_at_checkpoint,
                            "theta_runtime_at_checkpoint": theta_runtime_at_checkpoint,
                            "scaffold_labels_at_checkpoint": scaffold_labels_at_checkpoint,
                            "psi_current": np.asarray(baseline_exact["psi"], dtype=complex).reshape(-1),
                            "controller_obs": dict(controller_obs),
                            "energy_total_controller": float(energy_controller),
                            "trajectory_row": dict(self._trajectory[-1]),
                            "ledger_row": dict(self._ledger[-1]),
                            "post_prune_psi": (
                                None
                                if post_prune_psi is None
                                else np.asarray(post_prune_psi, dtype=complex).reshape(-1)
                            ),
                        }
                    )
                    if isinstance(observer_result, Mapping):
                        trajectory_update = observer_result.get("trajectory_update")
                        if isinstance(trajectory_update, Mapping) and self._trajectory:
                            self._trajectory[-1].update(dict(trajectory_update))
                        ledger_update = observer_result.get("ledger_update")
                        if isinstance(ledger_update, Mapping) and self._ledger:
                            self._ledger[-1].update(dict(ledger_update))
                if str(action_kind) != "repair_miss":
                    self._temporal_ledger.record_checkpoint(
                        checkpoint_index=int(checkpoint_index),
                        selected_candidate_identity=(
                            None if selected is None else str(selected.get("candidate_identity", selected_candidate_label))
                        ),
                        selected_position_id=selected_position_id,
                        selected_groups_new=float(selected_groups_new),
                        selected_gain_ratio=float(selected_gain_ratio),
                        predicted_displacement=float(predicted_displacement),
                        refresh_pressure=str(refresh_pressure),
                    )
                if bool(repair_retry_next):
                    self._restore_repair_noadvance_state(repair_noadvance_snapshot)
                self._write_progress(
                    stage="checkpoint_done",
                    force=True,
                    checkpoint_index=int(checkpoint_index),
                    time=float(time_value),
                    physical_time=float(step_hamiltonian.physical_time),
                    action_kind=str(action_kind),
                    controller_lane=str(controller_lane),
                    decision_backend=str(decision_backend),
                    oracle_decision_used=bool(oracle_decision_used),
                    shortlist_size=int(len(shortlist)),
                    oracle_confirm_limit=int(oracle_confirm_limit),
                    oracle_budget_scale=float(oracle_budget_scale),
                    degraded_reason=(None if degraded_reason is None else str(degraded_reason)),
                    trajectory_sample_kind=(
                        "repair_event" if str(action_kind) == "repair_miss" else "state_sample"
                    ),
                    advances_time=bool(str(action_kind) != "repair_miss"),
                    repair_attempt_index=int(repair_attempt.attempt_index),
                    repair_max_attempts=repair_attempt.max_attempts,
                    repair_escalation_kind=repair_attempt.escalation_kind,
                    repair_retry_next=bool(repair_retry_next),
                    repair_terminal=bool(repair_terminal),
                    repair_failure_reason=repair_failure_reason,
                    high_miss_no_admit_soft_fallback=bool(high_miss_no_admit_soft_fallback),
                    high_miss_no_admit_soft_fallback_policy=high_miss_no_admit_soft_fallback_policy,
                    high_miss_no_admit_soft_fallback_reason=high_miss_no_admit_soft_fallback_reason,
                    high_miss_no_admit_soft_fallback_warning=high_miss_no_admit_soft_fallback_warning,
                )
                self._write_partial_payload(stage="checkpoint_done")
                if bool(repair_retry_next):
                    repair_retry_attempts[int(checkpoint_index)] = int(repair_attempt.attempt_index) + 1
                    self._write_progress(
                        stage="repair_retry",
                        force=True,
                        checkpoint_index=int(checkpoint_index),
                        time=float(time_value),
                        repair_attempt_index=int(repair_attempt.attempt_index),
                        repair_next_attempt_index=int(repair_attempt.attempt_index) + 1,
                        repair_max_attempts=repair_attempt.max_attempts,
                        repair_escalation_kind=repair_attempt.escalation_kind,
                    )
                    self._write_partial_payload(stage="repair_retry")
                    continue
                if early_stop_reason is None:
                    early_stop_reason = self._progress_early_stop_reason(
                        checkpoint_index=int(checkpoint_index)
                    )
                if early_stop_reason is not None:
                    early_stop_checkpoint_index = int(checkpoint_index)
                    early_stop_time = float(time_value)
                    self._write_progress(
                        stage="early_stop",
                        force=True,
                        status="stopped_early",
                        checkpoint_index=int(checkpoint_index),
                        time=float(time_value),
                        early_stop_reason=str(early_stop_reason),
                    )
                    self._write_partial_payload(
                        status="stopped_early",
                        stage="early_stop",
                    )
                    break
                repair_retry_attempts.pop(int(checkpoint_index), None)
                checkpoint_index = int(checkpoint_index) + 1

            self._set_repair_attempt_state(0)
            append_count = int(sum(1 for row in self._ledger if str(row.get("action_kind")) == "append_candidate"))
            prune_count = int(sum(1 for row in self._ledger if str(row.get("action_kind")) == "prune_coordinate"))
            repair_count = int(sum(1 for row in self._ledger if str(row.get("action_kind", "")).startswith("repair_")))
            repair_retry_attempt_count = int(
                sum(
                    1
                    for row in self._ledger
                    if str(row.get("action_kind")) == "repair_miss"
                    and row.get("repair_max_attempts") is not None
                )
            )
            repair_retry_terminal_count = int(
                sum(1 for row in self._ledger if bool(row.get("repair_terminal", False)))
            )
            repair_retry_exhausted_count = int(
                sum(
                    1
                    for row in self._ledger
                    if str(row.get("repair_failure_reason", ""))
                    == "repair_retry_exhausted_high_miss_no_admit"
                )
            )
            repair_rescue_admitted_count = int(
                sum(1 for row in self._ledger if bool(row.get("repair_rescue_admitted", False)))
            )
            repair_escalation_kinds_used = sorted(
                {
                    str(row.get("repair_escalation_kind"))
                    for row in self._ledger
                    if row.get("repair_escalation_kind") not in {None, ""}
                }
            )
            stay_count = int(sum(1 for row in self._ledger if str(row.get("action_kind")) == "stay"))
            soft_fallback_counts = high_miss_no_admit_soft_fallback_counts(self._ledger)
            high_miss_no_admit_counts = high_miss_no_admit_diagnostic_counts(self._ledger)
            exact_decision_checkpoints = int(sum(1 for row in self._ledger if str(row.get("decision_backend")) == "exact"))
            oracle_decision_checkpoints = int(sum(1 for row in self._ledger if str(row.get("decision_backend")) == "oracle"))
            ideal_observable_decision_checkpoints = int(
                sum(1 for row in self._ledger if str(row.get("decision_backend")) == "ideal_observable")
            )
            oracle_attempted_checkpoints = int(sum(1 for row in self._ledger if bool(row.get("oracle_attempted", False))))
            decision_override_count = int(
                sum(1 for row in self._ledger if row.get("decision_override_reason") not in {None, ""})
            )
            forecast_override_count = int(
                sum(
                    1
                    for row in self._ledger
                    if str(row.get("decision_override_reason", "")).startswith(("local_forecast_", "exact_forecast_"))
                )
            )
            append_no_harm_veto_count = int(
                sum(
                    1
                    for row in self._ledger
                    if row.get("append_no_harm_veto_reason") not in {None, ""}
                )
            )
            exact_forecast_veto_count = int(forecast_override_count)
            integrator_used_values = [
                str(row.get("integrator_used"))
                for row in self._ledger
                if row.get("integrator_used", None) not in {None, ""}
            ]
            integrator_auto_euler_blocker_counts: dict[str, int] = {}
            for row in self._ledger:
                blockers = row.get("integrator_euler_blockers", [])
                if isinstance(blockers, str):
                    blockers_iterable: Sequence[Any] = [blockers]
                elif isinstance(blockers, Sequence):
                    blockers_iterable = blockers
                else:
                    blockers_iterable = []
                for blocker in blockers_iterable:
                    if blocker in {None, ""}:
                        continue
                    key = str(blocker)
                    integrator_auto_euler_blocker_counts[key] = int(
                        integrator_auto_euler_blocker_counts.get(key, 0)
                    ) + 1
            executed_backends = sorted({str(row.get("decision_backend", "exact")) for row in self._ledger}) or ["exact"]
            executed_data_flows = sorted(
                {
                    str(row.get("decision_data_flow"))
                    for row in self._ledger
                    if row.get("decision_data_flow") not in {None, ""}
                }
            )
            decision_data_flow = (
                "unknown"
                if not executed_data_flows
                else (executed_data_flows[0] if len(executed_data_flows) == 1 else "mixed")
            )
            uses_reference_for_decision = bool(
                any(bool(row.get("uses_reference_for_decision", False)) for row in self._ledger)
            )
            uses_future_exact_forecast_for_decision = bool(
                any(
                    bool(row.get("uses_future_exact_forecast_for_decision", False))
                    for row in self._ledger
                )
            )
            uses_statevector_as_ideal_observable_estimator = bool(
                any(
                    bool(row.get("uses_statevector_as_ideal_observable_estimator", False))
                    for row in self._ledger
                )
            )
            strict_measurement_oracle_certified = bool(
                self._ledger
                and all(
                    bool(row.get("strict_measurement_oracle_certified", False))
                    for row in self._ledger
                )
            )
            physical_rows = physical_trajectory_rows(self._trajectory)
            repair_row_counts = trajectory_repair_counts(self._trajectory)
            final_row = physical_rows[-1] if physical_rows else {}
            full_horizon_fields = full_horizon_completion_fields(
                self._trajectory,
                expected_t_final=float(self.times[-1]) if len(self.times) else 0.0,
                expected_row_count=int(len(self.times)),
                early_stop_reason=early_stop_reason,
                stable_early_stop_accepted=is_successful_stable_early_stop_reason(early_stop_reason),
            )
            staggered_error_vals = [
                float(row["abs_staggered_error"])
                for row in physical_rows
                if row.get("abs_staggered_error") is not None
            ]
            doublon_error_vals = [
                float(row["abs_doublon_error"])
                for row in physical_rows
                if row.get("abs_doublon_error") is not None
            ]
            site_occupation_error_vals = [
                float(row["site_occupations_abs_error_max"])
                for row in physical_rows
                if row.get("site_occupations_abs_error_max") is not None
            ]
            baseline_step_scale_vals = [
                float(row.get("baseline_step_scale"))
                for row in physical_rows
                if row.get("baseline_step_scale", None) is not None
            ]
            baseline_blend_weight_vals = [
                float(row.get("baseline_blend_weight"))
                for row in physical_rows
                if row.get("baseline_blend_weight", None) is not None
            ]
            baseline_gain_scale_vals = [
                float(row.get("baseline_gain_scale"))
                for row in physical_rows
                if row.get("baseline_gain_scale", None) is not None
            ]
            baseline_proposal_kind_vals = [
                str(row.get("baseline_proposal_kind"))
                for row in physical_rows
                if row.get("baseline_proposal_kind", None) not in {None, ""}
            ]
            final_ledger_row = self._ledger[-1] if self._ledger else {}
            oracle_backend_infos: list[dict[str, Any]] = []
            for row in physical_rows:
                for key in (
                    "selected_noisy_backend_info",
                    "stay_noisy_backend_info",
                    "baseline_backend_info",
                ):
                    info = row.get(key, None)
                    if isinstance(info, Mapping) and info:
                        oracle_backend_infos.append(dict(info))
            final_oracle_backend_info = (
                {} if not oracle_backend_infos else dict(oracle_backend_infos[-1])
            )
            final_oracle_backend_details = (
                {}
                if not isinstance(final_oracle_backend_info.get("details", {}), Mapping)
                else dict(final_oracle_backend_info.get("details", {}))
            )
            runtime_job_ids = sorted(
                {
                    str(job_id)
                    for info in oracle_backend_infos
                    if isinstance(info.get("details", {}), Mapping)
                    for job_id in info.get("details", {}).get("runtime_job_ids", [])
                    if job_id not in {None, ""}
                }
            )
            summary = {
                "mode": str(self.cfg.mode),
                "decision_forecast_mode": "local_projective_v1",
                "forecast_score_interpretation": "lower_is_better",
                "reference_mode": self._reference_mode(),
                "reference_enabled": bool(self._reference_enabled()),
                "controller_exact_input_mode": self._reference_mode(),
                "requested_decision_backend": (
                    "oracle"
                    if str(self.cfg.mode) == "oracle_v1"
                    else (
                        "ideal_observable"
                        if str(self.cfg.mode) == "observable_v1"
                        else ("off" if str(self.cfg.mode) == "off" else "exact")
                    )
                ),
                "status": (
                    "stopped_early"
                    if early_stop_reason is not None
                    else ("completed_with_fallback" if int(self._degraded_checkpoint_count) > 0 else "completed")
                ),
                "decision_backend": (
                    executed_backends[0]
                    if len(executed_backends) == 1
                    else "mixed"
                ),
                "executed_decision_backends": list(executed_backends),
                "decision_data_flow": str(decision_data_flow),
                "uses_reference_for_decision": bool(uses_reference_for_decision),
                "uses_future_exact_forecast_for_decision": bool(
                    uses_future_exact_forecast_for_decision
                ),
                "uses_statevector_as_ideal_observable_estimator": bool(
                    uses_statevector_as_ideal_observable_estimator
                ),
                "strict_measurement_oracle_certified": bool(
                    strict_measurement_oracle_certified
                ),
                "decision_noise_mode": (
                    "ideal"
                    if str(self.cfg.mode) == "observable_v1"
                    else (
                        None
                        if oracle_attempted_checkpoints <= 0 or self._oracle_base_config is None
                        else str(self._oracle_base_config.noise_mode)
                    )
                ),
                "oracle_estimate_kind": (
                    None if oracle_attempted_checkpoints <= 0 else self._oracle_estimate_kind()
                ),
                "ideal_observable_decision_checkpoints": int(ideal_observable_decision_checkpoints),
                "oracle_selection_policy": str(self.cfg.oracle_selection_policy),
                "confirm_score_mode": str(getattr(self.cfg, "confirm_score_mode", "exact_gain_ratio")),
                "prune_mode": str(getattr(self.cfg, "prune_mode", "off")),
                "prune_appended_origin_bias_enabled": bool(
                    getattr(self.cfg, "prune_appended_origin_bias_enabled", True)
                ),
                "prune_appended_origin_target_policy": str(
                    getattr(self.cfg, "prune_appended_origin_target_policy", "append_only")
                ),
                "prune_appended_origin_grace_steps": int(
                    getattr(self.cfg, "prune_appended_origin_grace_steps", 1)
                ),
                "prune_initial_scaffold_grace_steps": int(
                    getattr(self.cfg, "prune_initial_scaffold_grace_steps", 64)
                ),
                "prune_state_jump_l2_hard_cap": float(
                    getattr(self.cfg, "prune_state_jump_l2_hard_cap", 1.0e-2)
                ),
                "prune_active_block_theta_dot_rel_tol": float(
                    getattr(self.cfg, "prune_active_block_theta_dot_rel_tol", 0.03)
                ),
                "prune_active_block_theta_dot_abs_tol": float(
                    getattr(self.cfg, "prune_active_block_theta_dot_abs_tol", 1.0e-8)
                ),
                "prune_active_block_theta_dot_abs_hard_tol": float(
                    getattr(self.cfg, "prune_active_block_theta_dot_abs_hard_tol", 5.0e-2)
                ),
                "prune_appended_origin_bias_scale": float(
                    getattr(self.cfg, "prune_appended_origin_bias_scale", 0.10)
                ),
                "prune_appended_origin_bias_max_factor": float(
                    getattr(self.cfg, "prune_appended_origin_bias_max_factor", 0.50)
                ),
                "prune_appended_origin_bias_influenced_count": int(
                    sum(
                        1
                        for row in self._ledger
                        if bool(row.get("prune_appended_origin_bias_applied_selected", False))
                    )
                ),
                "candidate_pool_diagnostics_last": dict(self._last_candidate_pool_diagnostics),
                "prune_blocker_reason_counts": {
                    str(k): int(v)
                    for k, v in self._prune_blocker_reason_counts.items()
                    if not str(k).startswith("category:")
                },
                "prune_blocker_category_counts": {
                    str(k).split("category:", 1)[1]: int(v)
                    for k, v in self._prune_blocker_reason_counts.items()
                    if str(k).startswith("category:")
                },
                "high_miss_no_admit_policy": str(
                    getattr(self.cfg, "high_miss_no_admit_policy", HIGH_MISS_NO_ADMIT_POLICY_DEFAULT)
                ),
                "repair_retry_max_attempts": int(getattr(self.cfg, "repair_retry_max_attempts", 2)),
                "repair_retry_escalation_mode": str(
                    getattr(self.cfg, "repair_retry_escalation_mode", "append_budget_then_stabilize_v1")
                ),
                "repair_retry_admission_policy": str(
                    getattr(self.cfg, "repair_retry_admission_policy", "strict")
                ),
                "repair_retry_rescue_min_gain_ratio": float(
                    getattr(self.cfg, "repair_retry_rescue_min_gain_ratio", 0.0)
                ),
                "repair_retry_rescue_attempt": str(
                    getattr(self.cfg, "repair_retry_rescue_attempt", "terminal_attempt_only")
                ),
                "miss_abs_threshold": float(getattr(self.cfg, "miss_abs_threshold", 0.0)),
                "miss_persistence_window": int(getattr(self.cfg, "miss_persistence_window", 1)),
                "miss_persistence_count": int(getattr(self.cfg, "miss_persistence_count", 1)),
                "integrator_policy": str(self._integrator_policy()),
                "integrator_auto_policy_schema": (
                    AUTO_EULER_RK4_POLICY_SCHEMA
                    if str(self._integrator_policy()) == "auto_euler_rk4"
                    else None
                ),
                "integrator_columnarity_threshold": float(
                    getattr(self.cfg, "integrator_columnarity_threshold", 0.80)
                ),
                "integrator_curvature_threshold": float(
                    getattr(self.cfg, "integrator_curvature_threshold", 0.10)
                ),
                "integrator_euler_fs_error_threshold": float(
                    getattr(self.cfg, "integrator_euler_fs_error_threshold", 1.0e-3)
                ),
                "integrator_condition_max": float(
                    getattr(self.cfg, "integrator_condition_max", 1.0e10)
                ),
                "integrator_euler_min_time_fraction": float(
                    getattr(self.cfg, "integrator_euler_min_time_fraction", 0.0)
                ),
                "integrator_euler_observable_window": int(
                    getattr(self.cfg, "integrator_euler_observable_window", 16)
                ),
                "integrator_euler_site_span_max": getattr(
                    self.cfg, "integrator_euler_site_span_max", None
                ),
                "integrator_euler_primary_density_span_max": getattr(
                    self.cfg, "integrator_euler_primary_density_span_max", None
                ),
                "integrator_euler_energy_span_max": getattr(
                    self.cfg, "integrator_euler_energy_span_max", None
                ),
                "integrator_used_values": sorted(set(integrator_used_values)),
                "integrator_auto_euler_admitted_count": int(
                    sum(
                        1
                        for row in self._ledger
                        if bool(row.get("integrator_auto_admit_euler", False))
                    )
                ),
                "integrator_auto_euler_blocker_counts": {
                    str(key): int(value)
                    for key, value in sorted(integrator_auto_euler_blocker_counts.items())
                },
                "integrator_euler_count": int(
                    sum(1 for value in integrator_used_values if str(value) == "euler")
                ),
                "integrator_rk4_count": int(
                    sum(1 for value in integrator_used_values if str(value) == "rk4")
                ),
                "candidate_step_scales": [float(x) for x in self._candidate_step_scales()],
                "exact_forecast_baseline_step_refine_rounds": int(
                    self._exact_forecast_baseline_step_refine_rounds()
                ),
                "exact_forecast_baseline_proposal_mode": str(
                    self._exact_forecast_baseline_proposal_mode()
                ),
                "exact_forecast_baseline_blend_weights": [
                    float(x) for x in self._exact_forecast_baseline_blend_weights()
                ],
                "exact_forecast_baseline_gain_scales": [
                    float(x) for x in self._exact_forecast_baseline_gain_scales()
                ],
                "exact_forecast_include_tangent_secant_proposal": bool(
                    self._exact_forecast_include_tangent_secant_proposal()
                ),
                "exact_forecast_tangent_secant_trust_radius": float(
                    self._exact_forecast_tangent_secant_trust_radius()
                ),
                "exact_forecast_tangent_secant_signed_energy_lead_limit": float(
                    self._exact_forecast_tangent_secant_signed_energy_lead_limit()
                ),
                "exact_forecast_tracking_horizon_steps": int(
                    self._exact_forecast_tracking_horizon_steps()
                ),
                "exact_forecast_tracking_horizon_weights": [
                    float(x)
                    for x in self._exact_forecast_tracking_horizon_weights(
                        steps=self._exact_forecast_tracking_horizon_steps()
                    )
                ],
                "exact_forecast_primary_density_target_mode": str(
                    self._exact_forecast_primary_density_target_mode()
                ),
                "exact_forecast_tracking_fidelity_defect_weight": float(
                    getattr(self.cfg, "exact_forecast_tracking_fidelity_defect_weight", 1.0)
                ),
                "exact_forecast_tracking_primary_density_error_weight": float(
                    self._exact_forecast_tracking_primary_density_error_weight()
                ),
                "exact_forecast_tracking_staggered_error_weight": float(
                    getattr(self.cfg, "exact_forecast_tracking_staggered_error_weight", 1.0)
                ),
                "exact_forecast_tracking_doublon_error_weight": float(
                    getattr(self.cfg, "exact_forecast_tracking_doublon_error_weight", 1.0)
                ),
                "exact_forecast_tracking_site_occupations_error_weight": float(
                    getattr(self.cfg, "exact_forecast_tracking_site_occupations_error_weight", 1.0)
                ),
                "exact_forecast_tracking_energy_total_error_weight": float(
                    getattr(self.cfg, "exact_forecast_tracking_energy_total_error_weight", 1.0)
                ),
                "exact_forecast_density_slope_weight": float(
                    self._exact_forecast_density_slope_weight()
                ),
                "exact_forecast_energy_slope_weight": float(
                    getattr(self.cfg, "exact_forecast_energy_slope_weight", 0.0)
                ),
                "exact_forecast_energy_curvature_weight": float(
                    getattr(self.cfg, "exact_forecast_energy_curvature_weight", 0.0)
                ),
                "drive_aligned_density_active": bool(self._drive_aligned_density_active),
                "drive_aligned_density_label": self._drive_aligned_density_label,
                "baseline_step_scaling_active": bool(baseline_step_scale_vals),
                "baseline_step_scale_values_used": sorted(
                    {round(float(x), 12) for x in baseline_step_scale_vals}
                ),
                "baseline_blending_active": bool(baseline_blend_weight_vals),
                "baseline_blend_weight_values_used": sorted(
                    {round(float(x), 12) for x in baseline_blend_weight_vals}
                ),
                "baseline_gain_scaling_active": bool(baseline_gain_scale_vals),
                "baseline_gain_scale_values_used": sorted(
                    {round(float(x), 12) for x in baseline_gain_scale_vals}
                ),
                "baseline_proposal_kinds_used": sorted(set(baseline_proposal_kind_vals)),
                "exact_forecast_guardrail_mode": str(
                    getattr(self.cfg, "exact_forecast_guardrail_mode", "off")
                ),
                "append_enabled": bool(getattr(self.cfg, "append_enabled", True)),
                "append_no_harm_guard_enabled": bool(
                    getattr(self.cfg, "append_no_harm_guard_enabled", True)
                ),
                "append_no_harm_veto_count": int(append_no_harm_veto_count),
                "decision_override_count": int(decision_override_count),
                "forecast_override_count": int(forecast_override_count),
                "exact_forecast_veto_count": int(exact_forecast_veto_count),
                "append_count": int(append_count),
                "prune_count": int(prune_count),
                "repair_count": int(repair_count),
                "repair_retry_attempt_count": int(repair_retry_attempt_count),
                "repair_retry_terminal_count": int(repair_retry_terminal_count),
                "repair_retry_exhausted_count": int(repair_retry_exhausted_count),
                "repair_rescue_admitted_count": int(repair_rescue_admitted_count),
                "repair_escalation_kinds_used": list(repair_escalation_kinds_used),
                **repair_row_counts,
                "stay_count": int(stay_count),
                **soft_fallback_counts,
                **high_miss_no_admit_counts,
                **full_horizon_fields,
                "exact_decision_checkpoints": int(exact_decision_checkpoints),
                "oracle_decision_checkpoints": int(oracle_decision_checkpoints),
                "ideal_observable_decision_checkpoints": int(ideal_observable_decision_checkpoints),
                "oracle_attempted_checkpoints": int(oracle_attempted_checkpoints),
                "degraded_checkpoints": int(self._degraded_checkpoint_count),
                "raw_group_cache_hits": int(final_ledger_row.get("raw_group_cache_hits", 0)),
                "raw_group_cache_misses": int(final_ledger_row.get("raw_group_cache_misses", 0)),
                "raw_group_cache_extensions": int(final_ledger_row.get("raw_group_cache_extensions", 0)),
                "final_logical_block_count": int(self.current_layout.logical_parameter_count),
                "final_runtime_parameter_count": int(self.current_layout.runtime_parameter_count),
                "final_fidelity_exact": (
                    None if final_row.get("fidelity_exact") is None else float(final_row.get("fidelity_exact"))
                ),
                "final_abs_energy_total_error": (
                    None
                    if final_row.get("abs_energy_total_error") is None
                    else float(final_row.get("abs_energy_total_error"))
                ),
                "final_staggered": self._finite_float_or_none(final_row.get("staggered", None)),
                "final_staggered_exact": (
                    None if final_row.get("staggered_exact") is None else float(final_row.get("staggered_exact"))
                ),
                "final_abs_staggered_error": (
                    None
                    if final_row.get("abs_staggered_error") is None
                    else float(final_row.get("abs_staggered_error"))
                ),
                "max_abs_staggered_error": (
                    None if not staggered_error_vals else float(np.max(np.asarray(staggered_error_vals, dtype=float)))
                ),
                "final_doublon": self._finite_float_or_none(final_row.get("doublon", None)),
                "final_doublon_exact": (
                    None if final_row.get("doublon_exact") is None else float(final_row.get("doublon_exact"))
                ),
                "final_abs_doublon_error": (
                    None
                    if final_row.get("abs_doublon_error") is None
                    else float(final_row.get("abs_doublon_error"))
                ),
                "max_abs_doublon_error": (
                    None if not doublon_error_vals else float(np.max(np.asarray(doublon_error_vals, dtype=float)))
                ),
                "final_site_occupations": list(final_row.get("site_occupations", [])),
                "final_site_occupations_exact": final_row.get("site_occupations_exact", None),
                "final_site_occupations_abs_error_max": (
                    None
                    if final_row.get("site_occupations_abs_error_max") is None
                    else float(final_row.get("site_occupations_abs_error_max"))
                ),
                "max_abs_site_occupations_error": (
                    None if not site_occupation_error_vals else float(np.max(np.asarray(site_occupation_error_vals, dtype=float)))
                ),
                **summary_fields_from_row(final_row),
                "oracle_backend_info": dict(final_oracle_backend_info),
                "oracle_backend_snapshot": dict(
                    final_oracle_backend_details.get("backend_snapshot", {})
                ),
                "oracle_execution_surface": final_oracle_backend_details.get(
                    "execution_surface", None
                ),
                "oracle_runtime_profile": dict(
                    final_oracle_backend_details.get("runtime_profile", {})
                ),
                "oracle_runtime_raw_profile": dict(
                    final_oracle_backend_details.get("runtime_raw_profile", {})
                ),
                "oracle_runtime_session_policy": dict(
                    final_oracle_backend_details.get("runtime_session_policy", {})
                ),
                "oracle_raw_transport": final_oracle_backend_details.get(
                    "raw_transport", None
                ),
                "oracle_runtime_job_ids": list(runtime_job_ids),
                "oracle_runtime_job_count": int(len(runtime_job_ids)),
                "oracle_compile_request": {
                    "transpile_optimization_level": final_oracle_backend_details.get(
                        "transpile_optimization_level", None
                    ),
                    "transpile_seed": final_oracle_backend_details.get(
                        "transpile_seed", None
                    ),
                },
                "oracle_compile_observation": {
                    "layout_physical_qubits": list(
                        final_oracle_backend_details.get("layout_physical_qubits", [])
                    ),
                    "compiled_num_qubits": final_oracle_backend_details.get(
                        "compiled_num_qubits", None
                    ),
                    "compiled_depth": final_oracle_backend_details.get(
                        "compiled_depth", None
                    ),
                    "compiled_size": final_oracle_backend_details.get(
                        "compiled_size", None
                    ),
                    "compiled_count_2q": final_oracle_backend_details.get(
                        "compiled_count_2q", None
                    ),
                    "compiled_cx_count": final_oracle_backend_details.get(
                        "compiled_cx_count", None
                    ),
                    "compiled_ecr_count": final_oracle_backend_details.get(
                        "compiled_ecr_count", None
                    ),
                },
                "planning_audit": dict(self._planning_audit.summary()),
                "temporal_measurement_ledger": dict(self._temporal_ledger.summary()),
                "early_stop_reason": (None if early_stop_reason is None else str(early_stop_reason)),
                "early_stop_checkpoint_index": (
                    None if early_stop_checkpoint_index is None else int(early_stop_checkpoint_index)
                ),
                "early_stop_time": (None if early_stop_time is None else float(early_stop_time)),
            }
            reference = {
                "reference_mode": self._reference_mode(),
                "reference_enabled": False,
                "controller_exact_input_mode": self._reference_mode(),
                "uses_reference_for_decision": bool(uses_reference_for_decision),
                "uses_future_exact_forecast_for_decision": bool(
                    uses_future_exact_forecast_for_decision
                ),
                "kind": None,
                "initial_state": "stage_result.psi_final",
                "times": [float(x) for x in self.times.tolist()],
                "drive_profile": (None if self._drive_profile is None else dict(self._drive_profile)),
                "reference_method": None,
                "reference_steps_multiplier": None,
                "projection_time_sampling": None,
                "geometry_sample_time_policy": None,
            }
            if bool(getattr(self, "strict_qpu_faithful", False)):
                strict_contract = strict_qpu_faithful_decision_contract(
                    summary={
                        "reference_mode": "off",
                        "reference_enabled": False,
                        "exact_decision_checkpoints": int(exact_decision_checkpoints),
                        "oracle_decision_checkpoints": int(oracle_decision_checkpoints),
                        "ideal_observable_decision_checkpoints": int(ideal_observable_decision_checkpoints),
                    },
                    reference={"reference_mode": "off", "reference_enabled": False},
                    decision_rows=self._ledger,
                )
                strict_violations = [
                    str(item) for item in strict_contract.get("violations", [])
                ]
                strict_passed = bool(strict_contract.get("passed", False))
                summary.update(
                    {
                        "decision_path_kind": STRICT_QPU_FAITHFUL_DECISION_PATH_KIND,
                        "strict_qpu_faithful": True,
                        "strict_qpu_hh": bool(self.strict_qpu_hh),
                        "strict_qpu_family": str(self._family_key),
                        "qpu_faithful_decisions_expected": True,
                        "qpu_faithful_decisions_passed": bool(strict_passed),
                        "strict_decision_contract_passed": bool(strict_passed),
                        "strict_decision_contract_violations": list(strict_violations),
                        "strict_fail_closed": bool(not strict_passed),
                        "strict_fail_closed_reason": (
                            None
                            if strict_passed
                            else "; ".join(strict_violations) or "strict_decision_contract_failed"
                        ),
                        "controller_reference_mode": "off",
                        "controller_reference_enabled": False,
                        "controller_exact_input_mode": "off",
                        "uses_reference_for_decision": bool(
                            strict_contract.get("uses_reference_for_decision", False)
                        ),
                        "uses_future_exact_forecast_for_decision": bool(
                            strict_contract.get(
                                "uses_future_exact_forecast_for_decision", False
                            )
                        ),
                        "uses_statevector_as_ideal_observable_estimator": bool(
                            strict_contract.get(
                                "uses_statevector_as_ideal_observable_estimator", False
                            )
                        ),
                        "strict_measurement_oracle_certified": bool(
                            strict_contract.get(
                                "strict_measurement_oracle_certified", False
                            )
                        ),
                    }
                )
                if not strict_passed:
                    summary["status"] = "strict_fail_closed"
            if checkpoint_observer is not None and hasattr(checkpoint_observer, "finalize"):
                final_updates = checkpoint_observer.finalize(
                    summary=summary,
                    reference=reference,
                    trajectory=self._trajectory,
                    ledger=self._ledger,
                )
                if isinstance(final_updates, Mapping):
                    summary_update = final_updates.get("summary_update")
                    if isinstance(summary_update, Mapping):
                        summary.update(dict(summary_update))
                    reference_update = final_updates.get("reference")
                    if isinstance(reference_update, Mapping):
                        reference = dict(reference_update)
            final_status = str(summary.get("status", "completed"))
            self._write_progress(
                stage="run_complete",
                force=True,
                status=final_status,
                summary=summary,
            )
            self._write_partial_payload(
                status=final_status,
                stage="run_complete",
                summary=summary,
            )
            return ControllerRunArtifacts(
                trajectory=[dict(row) for row in self._trajectory],
                ledger=[dict(row) for row in self._ledger],
                summary=summary,
                reference=reference,
            )
        finally:
            self._close_oracles()


__all__ = [
    "ControllerDriveConfig",
    "RealtimeCheckpointController",
    "ControllerRunArtifacts",
    "RuntimeTermCarrier",
    "MotionSchedulerTelemetry",
]
