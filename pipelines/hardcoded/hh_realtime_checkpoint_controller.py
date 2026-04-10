#!/usr/bin/env python3
"""Exact/oracle HH adaptive realtime checkpoint controller."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from pipelines.hardcoded.hh_continuation_scoring import (
    FullScoreConfig,
    MeasurementCacheAudit,
    Phase1CompileCostOracle,
    Phase2NoveltyOracle,
    shortlist_records,
)
from pipelines.hardcoded.hh_continuation_stage_control import allowed_positions
from pipelines.hardcoded.hh_realtime_checkpoint_types import (
    BaselineGeometrySummary,
    CandidateProbeSummary,
    CheckpointLedgerEntry,
    DerivedGeometryKey,
    GeometryValueKey,
    OracleValueKey,
    RealtimeCheckpointConfig,
    dataclass_to_payload,
    hash_measurement_state,
    make_checkpoint_context,
)
from pipelines.hardcoded.hh_realtime_measurement import (
    BackendScheduledRawGroupPool,
    DerivedGeometryMemo,
    ExactCheckpointValueCache,
    OracleCheckpointValueCache,
    TemporalMeasurementLedger,
    build_controller_oracle_tier_configs,
    estimate_grouped_raw_mclachlan_incremental_block,
    controller_oracle_supports_raw_group_sampling,
    estimate_grouped_raw_mclachlan_geometry,
    planning_group_keys_for_term,
    planning_stats_for_term,
    validate_controller_oracle_base_config,
    validate_controller_tiers_mean_only,
)
from pipelines.hardcoded.hh_vqe_from_adapt_family import ReplayScaffoldContext
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
from src.quantum.hubbard_latex_python_pairs import build_hubbard_holstein_drive
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, PauliPolynomial

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


@dataclass(frozen=True)
class StepHamiltonianArtifacts:
    physical_time: float
    h_poly: Any
    hmat: np.ndarray
    compiled_h: Any
    oracle_observable: Any | None
    drive_term_count: int


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


@dataclass(frozen=True)
class MotionSchedulerTelemetry:
    regime: str
    direction_cosine: float | None
    rate_change_l2: float | None
    rate_change_ratio: float | None
    acceleration_l2: float | None
    curvature_cosine: float | None
    direction_reversal: bool
    curvature_sign_flip: bool
    kink_score: float


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


def _align_theta_vectors(lhs: np.ndarray | Sequence[float], rhs: np.ndarray | Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(lhs, dtype=float).reshape(-1)
    right = np.asarray(rhs, dtype=float).reshape(-1)
    width = max(int(left.size), int(right.size))
    out_left = np.zeros(int(width), dtype=float)
    out_right = np.zeros(int(width), dtype=float)
    out_left[: int(left.size)] = left
    out_right[: int(right.size)] = right
    return out_left, out_right


def _cosine_similarity(lhs: np.ndarray | Sequence[float], rhs: np.ndarray | Sequence[float]) -> float | None:
    left, right = _align_theta_vectors(lhs, rhs)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1.0e-12 or right_norm <= 1.0e-12:
        return None
    return float(np.dot(left, right) / max(left_norm * right_norm, 1.0e-12))


class RealtimeCheckpointController:
    """Exact/oracle horizon-1 stay/append/prune adaptive checkpoint controller."""

    def __init__(
        self,
        *,
        cfg: RealtimeCheckpointConfig,
        replay_context: ReplayScaffoldContext,
        h_poly: Any,
        hmat: np.ndarray,
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
    ) -> None:
        validate_controller_tiers_mean_only(cfg.tiers)
        self.cfg = cfg
        self.h_poly = h_poly
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
        self._drive_profile: dict[str, Any] | None = None
        self._reference_states: list[np.ndarray] | None = None
        replay_context_local = replay_context
        best_theta_arr = np.asarray(best_theta, dtype=float).reshape(-1)
        self._drive_aligned_density_active = False
        self._drive_aligned_density_label: str | None = None
        if self._drive_config is not None and str(cfg.mode) == "exact_v1":
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
        self._trajectory: list[dict[str, Any]] = []
        self._ledger: list[dict[str, Any]] = []
        self._previous_theta_dot: np.ndarray | None = None
        self._theta_dot_history: list[np.ndarray] = []
        self._previous_append_position: int | None = None
        self._block_birth_checkpoint: dict[str, int] = {}
        self._block_cooldown: dict[str, int] = {}
        self._block_burden: dict[str, float] = {}
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
        self._degraded_checkpoint_count = 0
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
        if mode not in {"off", "exact_v1", "oracle_v1"}:
            raise ValueError(f"Unsupported realtime checkpoint controller mode {mode!r}.")
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
        if forecast_guardrail_mode not in {"off", "dual_metric_v1"}:
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
        if prune_mode not in {"off", "exact_local_v1"}:
            raise ValueError(f"Unsupported prune mode {prune_mode!r}.")
        for field_name in (
            "exact_forecast_fidelity_loss_tol",
            "exact_forecast_abs_energy_error_increase_tol",
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
            "prune_state_jump_l2_tol",
            "prune_theta_block_tol",
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
        tracking_term_weights = (
            float(getattr(cfg, "exact_forecast_tracking_fidelity_defect_weight", 1.0)),
            float(getattr(cfg, "exact_forecast_tracking_staggered_error_weight", 1.0)),
            float(getattr(cfg, "exact_forecast_tracking_doublon_error_weight", 1.0)),
            float(getattr(cfg, "exact_forecast_tracking_site_occupations_error_weight", 1.0)),
            float(getattr(cfg, "exact_forecast_tracking_energy_total_error_weight", 1.0)),
        )
        for weight in tracking_term_weights:
            if (not np.isfinite(weight)) or weight < 0.0:
                raise ValueError(
                    "exact_forecast_tracking_*_weight values must be finite and nonnegative."
                )
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
        if float(cfg.confirm_compress_fraction) > 1.0:
            raise ValueError("confirm_compress_fraction must be <= 1.0.")
        if not (0.0 <= float(cfg.prune_stagnation_alpha) <= 1.0):
            raise ValueError("prune_stagnation_alpha must lie in [0, 1].")
        if float(cfg.prune_stale_score_threshold) > 1.0:
            raise ValueError("prune_stale_score_threshold must be <= 1.0.")
        for field_name in (
            "confirm_compress_min_modes",
            "confirm_compress_max_modes",
            "prune_protection_steps",
            "prune_stagnation_window",
            "prune_max_candidates",
            "prune_cooldown_steps",
        ):
            raw_value = int(getattr(cfg, field_name))
            if raw_value < 0:
                raise ValueError(f"{field_name} must be nonnegative; got {raw_value!r}.")
        if int(cfg.confirm_compress_max_modes) > 0 and int(cfg.confirm_compress_min_modes) > int(cfg.confirm_compress_max_modes):
            raise ValueError("confirm_compress_min_modes must be <= confirm_compress_max_modes.")
        if str(prune_mode) != "off" and float(cfg.prune_miss_threshold) > float(cfg.miss_threshold):
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

        if not bool(replay_context.pool_meta.get("candidate_pool_complete", True)):
            raise ValueError(
                "checkpoint controller exact_v1 requires a complete candidate family pool; sparse full_meta replay context is unsupported."
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
        psi_reconstructed = self.current_executor.prepare_state(
            self.current_theta,
            np.asarray(self.replay_context.psi_ref, dtype=complex).reshape(-1),
        )
        reconstruction_error = float(np.linalg.norm(psi_reconstructed - self.psi_initial))
        if reconstruction_error > float(cfg.reconstruction_tol):
            raise ValueError(
                f"Replay reconstruction mismatch: ||psi_reconstructed - psi_initial||={reconstruction_error:.3e} > {cfg.reconstruction_tol:.3e}."
            )

        for carrier in self.current_terms:
            self._planning_audit.commit(planning_group_keys_for_term(_carrier_to_term(carrier)))
        self._initialize_prune_state()

        if self._drive_config is not None:
            from pipelines.hardcoded.hh_fixed_manifold_measured import (
                FixedManifoldDriveConfig,
                FixedManifoldMeasuredConfig,
                _build_driven_reference_states,
            )

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
            self._reference_states = _build_driven_reference_states(
                psi_initial=np.asarray(self.psi_initial, dtype=complex),
                times=self.times,
                hmat_static=np.asarray(self.hmat, dtype=complex),
                h_poly_static=self.h_poly,
                drive_coeff_provider_exyz=self._drive_coeff_provider_exyz,
                drive_cfg=FixedManifoldDriveConfig(
                    enable_drive=True,
                    drive_A=float(self._drive_config.drive_A),
                    drive_omega=float(self._drive_config.drive_omega),
                    drive_tbar=float(self._drive_config.drive_tbar),
                    drive_phi=float(self._drive_config.drive_phi),
                    drive_pattern=str(self._drive_config.drive_pattern),
                    drive_custom_s=None,
                    drive_include_identity=bool(self._drive_config.drive_include_identity),
                    drive_time_sampling=str(self._drive_config.drive_time_sampling),
                    drive_t0=float(self._drive_config.drive_t0),
                    exact_steps_multiplier=int(self._drive_config.exact_steps_multiplier),
                ),
                geom_cfg=FixedManifoldMeasuredConfig(),
            )
            self._exact_evals = None
            self._exact_evecs = None
            self._exact_coeffs0 = None
        else:
            evals, evecs = np.linalg.eigh(self.hmat)
            self._exact_evals = np.asarray(evals, dtype=float)
            self._exact_evecs = np.asarray(evecs, dtype=complex)
            self._exact_coeffs0 = np.asarray(self._exact_evecs.conj().T @ self.psi_initial, dtype=complex)

    def _progress_payload(self, *, stage: str, **extra: Any) -> dict[str, Any]:
        elapsed = (
            None
            if self._run_wallclock_start is None
            else float(time.perf_counter() - float(self._run_wallclock_start))
        )
        payload: dict[str, Any] = {
            "status": "running",
            "stage": str(stage),
            "mode": str(self.cfg.mode),
            "append_count": int(self._append_counter),
            "prune_count": int(sum(1 for row in self._ledger if str(row.get("action_kind")) == "prune_coordinate")),
            "trajectory_points": int(len(self._trajectory)),
            "ledger_entries": int(len(self._ledger)),
            "logical_block_count": int(self.current_layout.logical_parameter_count),
            "runtime_parameter_count": int(self.current_layout.runtime_parameter_count),
            "total_checkpoints": int(len(self.times)),
            "wallclock_elapsed_s": elapsed,
        }
        payload.update(extra)
        return payload

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
        tmp_path = self._progress_path.with_suffix(f"{self._progress_path.suffix}.tmp")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self._progress_path)
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
        executed_backends = sorted({str(row.get("decision_backend", "exact")) for row in self._ledger})
        payload_summary = (
            dict(summary)
            if summary is not None
            else {
                "append_count": int(sum(1 for row in self._ledger if str(row.get("action_kind")) == "append_candidate")),
                "prune_count": int(sum(1 for row in self._ledger if str(row.get("action_kind")) == "prune_coordinate")),
                "stay_count": int(sum(1 for row in self._ledger if str(row.get("action_kind")) == "stay")),
                "executed_decision_backends": list(executed_backends),
                "final_logical_block_count": int(self.current_layout.logical_parameter_count),
                "final_runtime_parameter_count": int(self.current_layout.runtime_parameter_count),
            }
        )
        payload = {
            "status": str(status),
            "stage": str(stage),
            "mode": str(self.cfg.mode),
            "trajectory": [dict(row) for row in self._trajectory],
            "ledger": [dict(row) for row in self._ledger],
            "reference": {
                "controller_state": self._controller_state_payload(),
            },
            "summary": payload_summary,
        }
        tmp_path = self._partial_payload_path.with_suffix(f"{self._partial_payload_path.suffix}.tmp")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self._partial_payload_path)

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
        floor = float(max(1.0e-10, float(self.cfg.regularization_lambda)))
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

    def _exact_state_at(self, time_value: float) -> np.ndarray:
        if self._reference_states is not None:
            if len(self._reference_states) == 0:
                return np.asarray(self.psi_initial, dtype=complex).reshape(-1)
            idx = int(np.argmin(np.abs(self.times - float(time_value))))
            return np.asarray(self._reference_states[int(idx)], dtype=complex).reshape(-1)
        phases = np.exp(-1.0j * np.asarray(self._exact_evals, dtype=float) * float(time_value))
        return np.asarray(self._exact_evecs @ (phases * self._exact_coeffs0), dtype=complex)

    def _observable_snapshot(self, psi: np.ndarray) -> dict[str, Any]:
        snapshot = _site_resolved_number_observables(
            np.asarray(psi, dtype=complex).reshape(-1),
            num_sites=int(max(1, self._num_sites)),
            ordering=str(self._ordering),
        )
        return {
            "n_up_site": [float(x) for x in np.asarray(snapshot.n_up_site, dtype=float).tolist()],
            "n_dn_site": [float(x) for x in np.asarray(snapshot.n_dn_site, dtype=float).tolist()],
            "site_occupations": [float(x) for x in np.asarray(snapshot.n_site, dtype=float).tolist()],
            "doublon": float(snapshot.doublon),
            "staggered": float(snapshot.staggered),
        }

    def _exact_step_forecast(
        self,
        *,
        time_stop: float,
        executor: CompiledAnsatzExecutor,
        theta_runtime: np.ndarray | Sequence[float],
    ) -> dict[str, Any]:
        psi_pred = np.asarray(
            executor.prepare_state(
                np.asarray(theta_runtime, dtype=float).reshape(-1),
                self.replay_context.psi_ref,
            ),
            dtype=complex,
        ).reshape(-1)
        psi_exact = np.asarray(self._exact_state_at(float(time_stop)), dtype=complex).reshape(-1)
        step_hamiltonian = self._step_hamiltonian_artifacts(float(time_stop))
        energy_total_controller_next = float(
            np.real(np.vdot(psi_pred, step_hamiltonian.hmat @ psi_pred))
        )
        energy_total_exact_next = float(
            np.real(np.vdot(psi_exact, step_hamiltonian.hmat @ psi_exact))
        )
        pred_obs = self._observable_snapshot(psi_pred)
        exact_obs = self._observable_snapshot(psi_exact)
        pred_site = np.asarray(pred_obs["site_occupations"], dtype=float)
        exact_site = np.asarray(exact_obs["site_occupations"], dtype=float)
        site_occ_abs_error_max = float(
            np.max(np.abs(pred_site - exact_site)) if pred_site.size > 0 else float("nan")
        )
        return {
            "fidelity_exact_next": float(abs(np.vdot(psi_exact, psi_pred)) ** 2),
            "energy_total_controller_next": float(energy_total_controller_next),
            "energy_total_exact_next": float(energy_total_exact_next),
            "abs_energy_total_error_next": float(
                abs(float(energy_total_controller_next) - float(energy_total_exact_next))
            ),
            "abs_staggered_error_next": float(
                abs(float(pred_obs["staggered"]) - float(exact_obs["staggered"]))
            ),
            "abs_doublon_error_next": float(
                abs(float(pred_obs["doublon"]) - float(exact_obs["doublon"]))
            ),
            "site_occupations_abs_error_max_next": float(site_occ_abs_error_max),
        }

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

    def _exact_forecast_tracking_error_weights(self) -> tuple[float, float, float, float, float]:
        return (
            max(0.0, float(getattr(self.cfg, "exact_forecast_tracking_fidelity_defect_weight", 1.0))),
            max(0.0, float(getattr(self.cfg, "exact_forecast_tracking_staggered_error_weight", 1.0))),
            max(0.0, float(getattr(self.cfg, "exact_forecast_tracking_doublon_error_weight", 1.0))),
            max(
                0.0,
                float(getattr(self.cfg, "exact_forecast_tracking_site_occupations_error_weight", 1.0)),
            ),
            max(0.0, float(getattr(self.cfg, "exact_forecast_tracking_energy_total_error_weight", 1.0))),
        )

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
        _slope_weight, curvature_weight = self._exact_forecast_energy_shape_weights()
        excursion_under_weight = self._exact_forecast_energy_excursion_under_weight()
        excursion_over_weight = self._exact_forecast_energy_excursion_over_weight()
        curvature_anchor: dict[str, Any] | None = None
        if (
            (float(curvature_weight) > 0.0 and len(forecasts) >= 2)
            or float(excursion_under_weight) > 0.0
            or float(excursion_over_weight) > 0.0
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
        score = self._forecast_tracking_score(
            forecast=forecasts,
            curvature_anchor=curvature_anchor,
        )
        shape_terms = self._energy_shape_tracking_terms(
            forecasts=forecasts,
            weights=self._exact_forecast_tracking_horizon_weights(steps=len(forecasts)),
            curvature_anchor=curvature_anchor,
        )
        excursion_terms = self._energy_excursion_tracking_terms(
            forecasts=forecasts,
            weights=self._exact_forecast_tracking_horizon_weights(steps=len(forecasts)),
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
        first["tracking_horizon_weights_used"] = [
            float(x) for x in self._exact_forecast_tracking_horizon_weights(steps=len(forecasts))
        ]
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
        if mode != "dual_metric_v1":
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
        if (
            fidelity_delta < -float(fidelity_loss_tol)
            and energy_error_delta > float(energy_increase_tol)
        ):
            return "exact_forecast_dual_metric_regression"
        return None

    def _forecast_tracking_score(
        self,
        *,
        forecast: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        curvature_anchor: Mapping[str, Any] | None = None,
    ) -> float:
        if isinstance(forecast, Mapping):
            if "tracking_score_horizon" in forecast:
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
            staggered_error_weight,
            doublon_error_weight,
            site_occupations_error_weight,
            energy_total_error_weight,
        ) = self._exact_forecast_tracking_error_weights()
        score = 0.0
        for weight, item in zip(weights, forecasts):
            fidelity_defect = max(0.0, 1.0 - float(item["fidelity_exact_next"]))
            score += float(weight) * float(
                float(fidelity_defect_weight) * fidelity_defect
                + float(staggered_error_weight) * float(item["abs_staggered_error_next"])
                + float(doublon_error_weight) * float(item["abs_doublon_error_next"])
                + float(site_occupations_error_weight)
                * float(item["site_occupations_abs_error_max_next"])
                + float(energy_total_error_weight) * float(item["abs_energy_total_error_next"])
            )
        total = float(score / total_weight)
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

    def _stay_forecast_within_exact_v1_bounded_defect(
        self,
        *,
        forecast: Mapping[str, Any],
    ) -> bool:
        fidelity_defect = max(0.0, 1.0 - float(forecast["fidelity_exact_next"]))
        return bool(
            fidelity_defect <= 1.0e-3
            and float(forecast["abs_staggered_error_next"]) <= 2.0e-2
            and float(forecast["abs_doublon_error_next"]) <= 2.0e-3
            and float(forecast["site_occupations_abs_error_max_next"]) <= 2.0e-2
            and float(forecast["abs_energy_total_error_next"]) <= 2.0e-3
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
        selected_score = self._forecast_tracking_score(forecast=selected_forecast)
        stay_score = self._forecast_tracking_score(forecast=stay_forecast)
        if float(selected_score) >= float(stay_score) - 1.0e-12:
            return "exact_forecast_nonimproving_tracking_score"
        return None

    def _select_exact_v1_candidate_step_scale(
        self,
        *,
        baseline_theta_dot: np.ndarray | Sequence[float],
        selected: Mapping[str, Any],
        dt: float,
        time_stop: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        candidate_data = dict(selected["candidate_data"])
        best_selected: dict[str, Any] | None = None
        best_forecast: dict[str, Any] | None = None
        best_score: float | None = None
        best_scale: float | None = None
        for step_scale in self._candidate_step_scales():
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
            forecast, _forecast_rollout, score = self._exact_forecast_rollout(
                time_stop=float(time_stop),
                dt=float(dt),
                executor=candidate_data["aug_executor"],
                theta_runtime_start=theta_runtime,
                theta_dot_step=np.asarray(scaled_theta_dot_aug, dtype=float).reshape(-1),
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
            return False, "rho_miss_above_prune_threshold"
        direction_cosine = motion.direction_cosine
        if direction_cosine is None or float(direction_cosine) < float(self.cfg.motion_calm_direction_cosine_threshold):
            return False, "motion_not_calm_direction"
        rate_change_ratio = motion.rate_change_ratio
        if rate_change_ratio is None or float(rate_change_ratio) > float(self.cfg.motion_calm_rate_change_ratio_threshold):
            return False, "motion_not_calm_rate"
        return True, "prune_permitted"

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
                np.linalg.pinv(G_red, rcond=float(self.cfg.pinv_rcond))
                if G_red.size
                else np.zeros((0, 0), dtype=float)
            )
            reduced_objective = float(f_red @ (G_red_pinv @ f_red)) if f_red.size else 0.0
        norm_b_sq = float(max(float(baseline.get("norm_b_sq", 0.0)), 1.0e-14))
        return float(max(0.0, baseline_objective - reduced_objective) / norm_b_sq)

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
        rows: list[dict[str, Any]] = []
        for logical_index, (carrier, block) in enumerate(zip(self.current_terms, self.current_layout.blocks)):
            label = str(carrier.label)
            birth = int(self._block_birth_checkpoint.get(label, -protection_steps - 1))
            age = int(checkpoint_index) - int(birth)
            cooldown = int(self._block_cooldown.get(label, 0))
            runtime_indices = list(range(int(block.runtime_start), int(block.runtime_stop)))
            stagnation_score, motion_mean, fit_mean = self._block_stagnation_statistics(label)
            theta_block = np.asarray(self.current_theta[int(block.runtime_start) : int(block.runtime_stop)], dtype=float).reshape(-1)
            theta_block_norm = float(np.linalg.norm(theta_block))
            if age < int(protection_steps):
                continue
            if cooldown > 0:
                continue
            if float(stagnation_score) < float(stale_threshold):
                continue
            rows.append(
                {
                    "candidate_label": str(label),
                    "position_id": int(logical_index),
                    "runtime_block_indices": [int(x) for x in runtime_indices],
                    "cached_prune_loss": float(
                        self._cached_prune_loss(
                            baseline=baseline,
                            runtime_indices=runtime_indices,
                        )
                    ),
                    "stagnation_score": float(stagnation_score),
                    "motion_mean": float(motion_mean),
                    "fit_mean": float(fit_mean),
                    "theta_block_norm": float(theta_block_norm),
                    "burden": float(self._block_burden.get(label, max(1, len(runtime_indices)))),
                    "removed_carrier": carrier,
                }
            )
        rows.sort(
            key=lambda rec: (
                float(rec["cached_prune_loss"]),
                -float(rec["burden"]),
                int(rec["position_id"]),
            )
        )
        if not rows:
            return [], "no_prune_eligible_coordinates"
        return rows[: max(1, int(getattr(self.cfg, "prune_max_candidates", 1)))], "prune_candidates_available"

    def _build_pruned_runtime_state(self, *, logical_index: int) -> dict[str, Any]:
        idx = int(logical_index)
        removed_block = self.current_layout.blocks[idx]
        removed_carrier = self.current_terms[idx]
        reduced_terms = list(self.current_terms[:idx]) + list(self.current_terms[idx + 1 :])
        reduced_layout = _layout_from_carriers(reduced_terms, template=self.current_layout)
        reduced_theta = _delete_theta_block(
            self.current_theta,
            runtime_start=int(removed_block.runtime_start),
            runtime_stop=int(removed_block.runtime_stop),
        )
        reduced_executor = self._build_executor(reduced_terms, reduced_layout)
        reduced_psi = np.asarray(
            reduced_executor.prepare_state(reduced_theta, self.replay_context.psi_ref),
            dtype=complex,
        ).reshape(-1)
        return {
            "removed_label": str(removed_carrier.label),
            "removed_source_label": str(removed_carrier.source_label),
            "removed_carrier": removed_carrier,
            "removed_block": removed_block,
            "reduced_terms": reduced_terms,
            "reduced_layout": reduced_layout,
            "reduced_theta": np.asarray(reduced_theta, dtype=float).reshape(-1),
            "reduced_executor": reduced_executor,
            "reduced_psi": reduced_psi,
            "reduced_planning_audit": self._build_planning_audit_for_terms(reduced_terms),
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

    def _step_hamiltonian_artifacts(self, time_value: float) -> StepHamiltonianArtifacts:
        if self._drive_config is None or self._drive_coeff_provider_exyz is None:
            return StepHamiltonianArtifacts(
                physical_time=float(time_value),
                h_poly=self.h_poly,
                hmat=np.asarray(self.hmat, dtype=complex),
                compiled_h=self._compiled_h,
                oracle_observable=self._oracle_qop,
                drive_term_count=0,
            )

        from pipelines.exact_bench.noise_oracle_runtime import pauli_poly_to_sparse_pauli_op
        from pipelines.hardcoded.hh_fixed_manifold_measured import (
            FixedManifoldMeasuredConfig,
            _build_driven_hamiltonian,
        )

        physical_time = self._physical_time(float(time_value))
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
        current = np.asarray(theta_dot, dtype=float).reshape(-1)
        if not self._theta_dot_history:
            return MotionSchedulerTelemetry(
                regime="bootstrap",
                direction_cosine=None,
                rate_change_l2=None,
                rate_change_ratio=None,
                acceleration_l2=None,
                curvature_cosine=None,
                direction_reversal=False,
                curvature_sign_flip=False,
                kink_score=0.0,
            )
        previous = np.asarray(self._theta_dot_history[-1], dtype=float).reshape(-1)
        current_aligned, previous_aligned = _align_theta_vectors(current, previous)
        delta = np.asarray(current_aligned - previous_aligned, dtype=float)
        rate_change_l2 = float(np.linalg.norm(delta))
        previous_norm = float(np.linalg.norm(previous_aligned))
        current_norm = float(np.linalg.norm(current_aligned))
        rate_change_denom = max(previous_norm, current_norm, 1.0e-12)
        rate_change_ratio = float(rate_change_l2 / rate_change_denom)
        direction_cosine = _cosine_similarity(current_aligned, previous_aligned)
        direction_reversal = bool(
            direction_cosine is not None
            and float(direction_cosine) <= float(self.cfg.motion_direction_reversal_cosine_threshold)
            and float(rate_change_ratio) >= float(self.cfg.motion_calm_rate_change_ratio_threshold)
            and float(rate_change_l2) >= float(self.cfg.motion_acceleration_l2_threshold)
        )
        acceleration_l2 = float(np.linalg.norm(delta))
        curvature_cosine: float | None = None
        curvature_sign_flip = False
        if len(self._theta_dot_history) >= 2:
            previous_previous = np.asarray(self._theta_dot_history[-2], dtype=float).reshape(-1)
            max_width = max(int(current.size), int(previous.size), int(previous_previous.size))
            current_pad = np.zeros(int(max_width), dtype=float)
            previous_pad = np.zeros(int(max_width), dtype=float)
            previous_previous_pad = np.zeros(int(max_width), dtype=float)
            current_pad[: int(current.size)] = current
            previous_pad[: int(previous.size)] = previous
            previous_previous_pad[: int(previous_previous.size)] = previous_previous
            acceleration = np.asarray(current_pad - previous_pad, dtype=float)
            previous_acceleration = np.asarray(previous_pad - previous_previous_pad, dtype=float)
            acceleration_l2 = float(np.linalg.norm(acceleration))
            previous_acceleration_l2 = float(np.linalg.norm(previous_acceleration))
            curvature_cosine = _cosine_similarity(acceleration, previous_acceleration)
            curvature_sign_flip = bool(
                curvature_cosine is not None
                and float(curvature_cosine) <= float(self.cfg.motion_curvature_flip_cosine_threshold)
                and float(acceleration_l2) >= float(self.cfg.motion_acceleration_l2_threshold)
                and float(previous_acceleration_l2) >= float(self.cfg.motion_acceleration_l2_threshold)
            )
        calm = bool(
            direction_cosine is not None
            and float(direction_cosine) >= float(self.cfg.motion_calm_direction_cosine_threshold)
            and float(rate_change_ratio) <= float(self.cfg.motion_calm_rate_change_ratio_threshold)
            and not direction_reversal
            and not curvature_sign_flip
            and float(predicted_displacement) <= 0.05
        )
        kink_score = float(
            max(
                0.0,
                float(rate_change_ratio),
                0.0 if direction_cosine is None else float(max(0.0, -direction_cosine)),
                0.0 if curvature_cosine is None else float(max(0.0, -curvature_cosine)),
            )
        )
        large_change = float(rate_change_l2) >= float(self.cfg.motion_acceleration_l2_threshold)
        if bool(direction_reversal) or bool(curvature_sign_flip) or (
            bool(large_change)
            and float(rate_change_ratio) >= float(self.cfg.motion_kink_rate_change_ratio_threshold)
        ):
            regime = "kink"
        elif bool(calm):
            regime = "calm"
        else:
            regime = "steady"
        return MotionSchedulerTelemetry(
            regime=str(regime),
            direction_cosine=(None if direction_cosine is None else float(direction_cosine)),
            rate_change_l2=float(rate_change_l2),
            rate_change_ratio=float(rate_change_ratio),
            acceleration_l2=float(acceleration_l2),
            curvature_cosine=(None if curvature_cosine is None else float(curvature_cosine)),
            direction_reversal=bool(direction_reversal),
            curvature_sign_flip=bool(curvature_sign_flip),
            kink_score=float(kink_score),
        )

    def _effective_refresh_pressure(
        self,
        *,
        base_refresh_pressure: str,
        motion: MotionSchedulerTelemetry,
    ) -> str:
        order = {"low": 0, "medium": 1, "high": 2}
        base = str(base_refresh_pressure).strip().lower()
        motion_floor = (
            "high"
            if str(motion.regime) == "kink"
            else ("medium" if str(motion.regime) == "bootstrap" else "low")
        )
        return max((base, motion_floor), key=lambda item: int(order.get(str(item), 1)))

    def _shortlist_cfg_for_motion(self, motion: MotionSchedulerTelemetry) -> FullScoreConfig:
        base_size = int(self._shortlist_cfg.shortlist_size)
        base_fraction = float(self._shortlist_cfg.shortlist_fraction)
        if str(motion.regime) == "calm":
            return FullScoreConfig(
                shortlist_fraction=float(
                    max(0.05, base_fraction * float(self.cfg.motion_calm_shortlist_scale))
                ),
                shortlist_size=max(
                    1,
                    int(np.ceil(float(base_size) * float(self.cfg.motion_calm_shortlist_scale))),
                ),
            )
        if str(motion.regime) == "kink":
            return FullScoreConfig(
                shortlist_fraction=float(min(1.0, base_fraction * 1.5)),
                shortlist_size=max(1, int(base_size) + int(self.cfg.motion_kink_shortlist_bonus)),
            )
        return self._shortlist_cfg

    def _oracle_confirm_limit_for_motion(
        self,
        *,
        confirmed_count: int,
        refresh_pressure: str,
        motion: MotionSchedulerTelemetry,
    ) -> int:
        count = max(0, int(confirmed_count))
        if count <= 0:
            return 0
        if str(motion.regime) == "kink" or str(refresh_pressure) == "high":
            return int(count)
        if str(refresh_pressure) == "medium":
            return min(2, int(count))
        if str(motion.regime) == "calm":
            return min(1, int(count))
        return min(2, int(count))

    def _oracle_budget_scale_for_motion(
        self,
        *,
        refresh_pressure: str,
        motion: MotionSchedulerTelemetry,
    ) -> float:
        if str(motion.regime) == "kink" or str(refresh_pressure) == "high":
            return float(max(1.0, float(self.cfg.motion_kink_oracle_budget_scale)))
        if str(motion.regime) == "calm" and str(refresh_pressure) == "low":
            return float(max(0.25, float(self.cfg.motion_calm_oracle_budget_scale)))
        return 1.0

    """
    lane_k =
    append, if time_stop exists and rho_miss_k > tau_miss;
    stay,   otherwise.
    """
    def _controller_lane(
        self,
        *,
        time_stop: float | None,
        rho_miss: float,
        prune_candidates_available: bool = False,
        prune_reason: str | None = None,
    ) -> tuple[str, str]:
        if time_stop is None:
            return "stay", "terminal_checkpoint"
        if str(self.cfg.mode) == "off":
            return "stay", "controller_disabled"
        if float(rho_miss) > float(self.cfg.miss_threshold):
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

        return FixedManifoldMeasuredConfig(
            regularization_lambda=float(self.cfg.regularization_lambda),
            pinv_rcond=float(self.cfg.pinv_rcond),
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
        return bool(
            getattr(self.cfg, "exact_forecast_include_tangent_secant_proposal", False)
        )

    def _exact_forecast_tangent_secant_trust_radius(self) -> float:
        return max(
            0.0,
            float(
                getattr(self.cfg, "exact_forecast_tangent_secant_trust_radius", 0.0)
            ),
        )

    def _exact_forecast_tangent_secant_signed_energy_lead_limit(self) -> float:
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
        if baseline is None or float(dt) <= 0.0:
            return None
        psi_current = np.asarray(baseline.get("psi", np.zeros(0, dtype=complex)), dtype=complex).reshape(-1)
        tangents = np.asarray(
            baseline.get("T", np.zeros((psi_current.size, 0), dtype=complex)),
            dtype=complex,
        )
        K_pinv = np.asarray(
            baseline.get("K_pinv", np.zeros((0, 0), dtype=float)),
            dtype=float,
        )
        if (
            psi_current.size <= 0
            or tangents.ndim != 2
            or tangents.shape[0] != psi_current.size
            or tangents.shape[1] <= 0
            or K_pinv.ndim != 2
            or K_pinv.shape != (tangents.shape[1], tangents.shape[1])
        ):
            return None
        psi_exact_next = np.asarray(
            self._exact_state_at(float(time_stop)),
            dtype=complex,
        ).reshape(-1)
        if psi_exact_next.shape != psi_current.shape:
            return None
        overlap = complex(np.vdot(psi_current, psi_exact_next))
        if abs(overlap) > 1.0e-15:
            psi_exact_next = np.asarray(
                np.exp(-1.0j * float(np.angle(overlap))) * psi_exact_next,
                dtype=complex,
            ).reshape(-1)
        delta_sec = np.asarray(psi_exact_next - psi_current, dtype=complex).reshape(-1)
        delta_sec = np.asarray(
            delta_sec - complex(np.vdot(psi_current, delta_sec)) * psi_current,
            dtype=complex,
        ).reshape(-1)
        delta_norm = float(np.linalg.norm(delta_sec))
        if (not np.isfinite(delta_norm)) or delta_norm <= 1.0e-12:
            return None
        xi_sec = np.asarray(
            np.real(tangents.conj().T @ delta_sec),
            dtype=float,
        ).reshape(-1)
        if xi_sec.size != int(tangents.shape[1]):
            return None
        coeff_sec = np.asarray(K_pinv @ xi_sec, dtype=float).reshape(-1)
        if coeff_sec.size != xi_sec.size or not np.all(np.isfinite(coeff_sec)):
            return None
        theta_dot_sec = np.asarray(coeff_sec / float(dt), dtype=float).reshape(-1)
        current_energy_bias: float | None = None
        next_exact_energy_delta: float | None = None
        signed_energy_lead: float | None = None
        signed_energy_lead_taper = 1.0
        signed_energy_lead_limit = self._exact_forecast_tangent_secant_signed_energy_lead_limit()
        if signed_energy_lead_limit > 0.0:
            time_start = float(time_stop) - float(dt)
            current_step_hamiltonian = self._step_hamiltonian_artifacts(float(time_start))
            next_step_hamiltonian = self._step_hamiltonian_artifacts(float(time_stop))
            psi_exact_current = np.asarray(
                self._exact_state_at(float(time_start)),
                dtype=complex,
            ).reshape(-1)
            if psi_exact_current.shape == psi_current.shape:
                energy_current_controller = float(
                    np.real(np.vdot(psi_current, current_step_hamiltonian.hmat @ psi_current))
                )
                energy_current_exact = float(
                    np.real(np.vdot(psi_exact_current, current_step_hamiltonian.hmat @ psi_exact_current))
                )
                energy_next_exact = float(
                    np.real(np.vdot(psi_exact_next, next_step_hamiltonian.hmat @ psi_exact_next))
                )
                current_energy_bias = float(energy_current_controller - energy_current_exact)
                next_exact_energy_delta = float(energy_next_exact - energy_current_exact)
                if abs(float(next_exact_energy_delta)) > 1.0e-15:
                    signed_energy_lead = float(
                        max(
                            0.0,
                            float(np.sign(next_exact_energy_delta)) * float(current_energy_bias),
                        )
                    )
                    lead_cap = float(signed_energy_lead_limit) * abs(float(next_exact_energy_delta))
                    if lead_cap > 1.0e-15 and float(signed_energy_lead) > 0.0:
                        signed_energy_lead_taper = max(
                            0.0,
                            1.0 - float(signed_energy_lead) / float(lead_cap),
                        )
                        if signed_energy_lead_taper <= 1.0e-12:
                            return None
                        theta_dot_sec = np.asarray(
                            float(signed_energy_lead_taper) * theta_dot_sec,
                            dtype=float,
                        ).reshape(-1)
        raw_metric_norm = float(
            self._proposal_metric_norm(
                baseline=baseline,
                theta_dot=theta_dot_sec,
            )
        )
        trust_radius = self._exact_forecast_tangent_secant_trust_radius()
        if trust_radius > 0.0 and raw_metric_norm > float(trust_radius):
            theta_dot_sec = np.asarray(
                (float(trust_radius) / float(raw_metric_norm)) * theta_dot_sec,
                dtype=float,
            ).reshape(-1)
        clipped_metric_norm = float(
            self._proposal_metric_norm(
                baseline=baseline,
                theta_dot=theta_dot_sec,
            )
        )
        projected_delta = np.asarray(tangents @ coeff_sec, dtype=complex).reshape(-1)
        projection_quality = float(
            np.linalg.norm(projected_delta) / max(float(delta_norm), 1.0e-15)
        )
        return {
            "proposal_kind": "tangent_secant_exact_v1",
            "blend_weight": 0.0,
            "theta_dot_direction": np.asarray(theta_dot_sec, dtype=float).reshape(-1),
            "current_baseline_norm": None,
            "current_drive_norm": None,
            "lookahead_drive_norm": None,
            "tangent_secant_displacement_norm": float(delta_norm),
            "tangent_secant_projection_quality": float(projection_quality),
            "tangent_secant_raw_metric_norm": float(raw_metric_norm),
            "tangent_secant_metric_norm": float(clipped_metric_norm),
            "tangent_secant_current_energy_bias": (
                None if current_energy_bias is None else float(current_energy_bias)
            ),
            "tangent_secant_next_exact_energy_delta": (
                None if next_exact_energy_delta is None else float(next_exact_energy_delta)
            ),
            "tangent_secant_signed_energy_lead": (
                None if signed_energy_lead is None else float(signed_energy_lead)
            ),
            "tangent_secant_signed_energy_lead_limit": float(signed_energy_lead_limit),
            "tangent_secant_signed_energy_lead_taper": float(signed_energy_lead_taper),
        }

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
        K_drive_pinv = np.linalg.pinv(K_drive, rcond=float(self.cfg.pinv_rcond))
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
            forecast, _forecast_rollout, score = self._exact_forecast_rollout(
                time_stop=float(time_stop),
                dt=float(dt),
                executor=self.current_executor,
                theta_runtime_start=theta_runtime,
                theta_dot_step=np.asarray(scaled_theta_dot, dtype=float).reshape(-1),
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
        for proposal in proposal_records:
            blend_weight = float(proposal.get("blend_weight", 0.0) or 0.0)
            theta_dot = np.asarray(proposal["theta_dot_direction"], dtype=float).reshape(-1)
            for step_scale in self._drive_aligned_baseline_step_scales():
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
            forecast, _forecast_rollout, score = self._exact_forecast_rollout(
                time_stop=float(time_stop),
                dt=float(dt),
                executor=self.current_executor,
                theta_runtime_start=theta_runtime,
                theta_dot_step=np.asarray(gained_theta_dot, dtype=float).reshape(-1),
            )
            cached = (
                np.asarray(gained_theta_dot, dtype=float).reshape(-1),
                dict(forecast),
                float(score),
            )
            gain_evaluated[rounded] = cached
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
            state_key = self._measurement_state_key(
                layout=self.current_layout,
                theta_runtime=self.current_theta,
            )
            measured = estimate_grouped_raw_mclachlan_geometry(
                oracle=oracle,
                raw_group_pool=raw_group_pool,
                layout=self.current_layout,
                theta_runtime=self.current_theta,
                psi_ref=np.asarray(self.replay_context.psi_ref, dtype=complex).reshape(-1),
                h_poly=h_poly_step,
                geom_cfg=self._measured_geometry_config(),
                observable_family_prefix="baseline_geometry",
                candidate_label=None,
                position_id=None,
                state_key=str(state_key),
                min_total_shots=int(min_total_shots),
                min_samples=int(min_samples),
            )
            geometry = dict(measured["geometry"])
            summary = BaselineGeometrySummary(
                energy=float(geometry["energy"]),
                variance=float(geometry["variance"]),
                epsilon_proj_sq=float(geometry["epsilon_proj_sq"]),
                epsilon_step_sq=float(geometry["epsilon_step_sq"]),
                rho_miss=float(geometry["rho_miss"]),
                theta_dot_l2=float(np.linalg.norm(np.asarray(geometry["theta_dot_step"], dtype=float))),
                matrix_rank=int(geometry["matrix_rank"]),
                condition_number=float(geometry["condition_number"]),
                regularization_lambda=float(self.cfg.regularization_lambda),
                solve_mode="grouped_raw_measured",
                logical_block_count=int(self.current_layout.logical_parameter_count),
                runtime_parameter_count=int(self.current_layout.runtime_parameter_count),
                planning_summary=dict(self._planning_audit.summary()),
                exact_cache_summary=dict(cache.summary()),
            )
            return {
                **geometry,
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
                candidate_regularization_lambda=float(self.cfg.candidate_regularization_lambda),
                pinv_rcond=float(self.cfg.pinv_rcond),
                observable_family_prefix="candidate_incremental_block",
                candidate_label=str(candidate_identity),
                position_id=int(position_id),
                state_key=str(state_key),
                min_total_shots=int(min_total_shots),
                min_samples=int(min_samples),
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
                np.linalg.pinv(G_eval, rcond=float(self.cfg.pinv_rcond))
                if G_eval.size
                else np.zeros((0, 0), dtype=float)
            )
            K = np.asarray(
                G_eval
                + float(self.cfg.regularization_lambda) * np.eye(int(G_eval.shape[0])),
                dtype=float,
            )
            K_pinv = (
                np.linalg.pinv(K, rcond=float(self.cfg.pinv_rcond))
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
            rank = (
                int(np.linalg.matrix_rank(K, tol=float(self.cfg.pinv_rcond)))
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
                "summary": BaselineGeometrySummary(
                    energy=float(energy),
                    variance=float(variance),
                    epsilon_proj_sq=float(epsilon_proj_sq),
                    epsilon_step_sq=float(epsilon_step_sq),
                    rho_miss=float(rho_miss),
                    theta_dot_l2=float(theta_dot_l2),
                    matrix_rank=int(rank),
                    condition_number=float(cond),
                    regularization_lambda=float(self.cfg.regularization_lambda),
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

    def _candidate_pool_terms(self) -> list[tuple[int, AnsatzTerm]]:
        current_source_labels = self._current_source_labels()
        out: list[tuple[int, AnsatzTerm]] = []
        for pool_index, term in enumerate(self.replay_context.family_pool):
            source_label = str(term.label)
            if (not self.allow_repeats) and source_label in current_source_labels:
                continue
            out.append((int(pool_index), term))
        return out

    def _candidate_positions(self) -> list[int]:
        logical_count = int(self.current_layout.logical_parameter_count)
        active_window = range(max(0, logical_count - int(self.cfg.active_window_size)), logical_count)
        return allowed_positions(
            n_params=int(logical_count),
            append_position=int(logical_count),
            active_window_indices=active_window,
            max_positions=int(self.cfg.max_probe_positions),
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
        records: list[dict[str, Any]] = []
        current_terms_window = self.current_terms[
            max(0, len(self.current_terms) - int(self.cfg.active_window_size)) :
        ]
        for candidate_pool_index, candidate_term in self._candidate_pool_terms():
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
                    C + float(self.cfg.candidate_regularization_lambda) * np.eye(int(C.shape[0])),
                    dtype=float,
                )
                C_reg_pinv = (
                    np.linalg.pinv(C_reg, rcond=float(self.cfg.pinv_rcond))
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
        return shortlist_records(
            records,
            cfg=(self._shortlist_cfg if shortlist_cfg is None else shortlist_cfg),
            score_key="scout_score",
            tie_break_score_key="scout_score",
        )

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
            gain = float(max(0.0, fallback @ (np.linalg.pinv(C_mat + float(self.cfg.candidate_regularization_lambda) * np.eye(int(C_mat.shape[0])), rcond=float(self.cfg.pinv_rcond)) @ fallback))) if C_mat.size else 0.0
            ratio = float(gain / max(float(baseline.get("norm_b_sq", 0.0)), 1.0e-14))
            return float(gain), float(ratio), 0, 0
        if K.size == 0 or q_vec.size == 0:
            return 0.0, 0.0, 0, 0
        evals, evecs = np.linalg.eigh(K)
        if evals.size == 0:
            return 0.0, 0.0, 0, 0
        tol = max(1.0e-12, float(self.cfg.pinv_rcond) * float(np.max(np.abs(evals))))
        support = np.flatnonzero(evals > tol)
        if support.size <= 0:
            S_tilde = np.asarray(
                C_mat + float(self.cfg.candidate_regularization_lambda) * np.eye(int(C_mat.shape[0])),
                dtype=float,
            )
            gain = float(max(0.0, fallback @ (np.linalg.pinv(S_tilde, rcond=float(self.cfg.pinv_rcond)) @ fallback))) if fallback.size else 0.0
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
                C_mat + float(self.cfg.candidate_regularization_lambda) * np.eye(int(C_mat.shape[0])),
                dtype=float,
            )
            gain = float(max(0.0, w_vec @ (np.linalg.pinv(S_tilde, rcond=float(self.cfg.pinv_rcond)) @ w_vec))) if w_vec.size else 0.0
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
            + float(self.cfg.candidate_regularization_lambda) * np.eye(int(C_mat.shape[0]))
            - Gamma_selected.T @ Gamma_selected,
            dtype=float,
        )
        S_tilde_pinv = np.linalg.pinv(S_tilde, rcond=float(self.cfg.pinv_rcond)) if S_tilde.size else np.zeros((0, 0), dtype=float)
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
        groups_new: float,
        directional_change_l2: float | None,
    ) -> dict[str, Any]:
        directional_penalty = 0.0 if directional_change_l2 is None else float(directional_change_l2)
        adjusted_gain = float(
            float(gain_ratio)
            - float(self.cfg.directional_penalty_weight) * directional_penalty
            - float(self.cfg.measurement_penalty_weight) * float(groups_new)
        )
        mode = str(getattr(self.cfg, "confirm_score_mode", "exact_gain_ratio"))
        if mode != "compressed_whitened_v1":
            return {
                "adjusted_gain": float(adjusted_gain),
                "confirm_score": float(adjusted_gain),
                "confirm_score_kind": "geometry_gain_ratio_minus_penalties",
                "confirm_compress_modes_used": 0,
                "confirm_support_rank": 0,
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
            - float(self.cfg.directional_penalty_weight) * directional_penalty
            - float(self.cfg.measurement_penalty_weight) * float(groups_new)
        )
        return {
            "adjusted_gain": float(adjusted_gain),
            "confirm_score": float(confirm_score),
            "confirm_score_kind": "compressed_whitened_lower_gain_ratio_minus_penalties",
            "confirm_compress_modes_used": int(modes_used),
            "confirm_support_rank": int(support_rank),
            "confirm_compressed_gain_ratio": float(compressed_ratio),
            "confirm_compressed_gain_exact": float(compressed_gain),
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
        rec["confirm_compressed_gain_ratio"] = None
        rec["confirm_compressed_gain_exact"] = None
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
            + float(self.cfg.candidate_regularization_lambda) * np.eye(int(C.shape[0]))
            - B.T @ K_pinv @ B,
            dtype=float,
        )
        S_pinv = np.linalg.pinv(S, rcond=float(self.cfg.pinv_rcond)) if S.size else np.zeros((0, 0), dtype=float)
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
            if float(proposed.get("theta_block_norm", 0.0)) > float(theta_tol):
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

            reduced_state = self._build_pruned_runtime_state(logical_index=int(proposed["position_id"]))
            state_jump_l2 = float(
                np.linalg.norm(
                    np.asarray(reduced_state["reduced_psi"], dtype=complex).reshape(-1)
                    - np.asarray(baseline["psi"], dtype=complex).reshape(-1)
                )
            )
            proposed["post_prune_state_jump_l2"] = float(state_jump_l2)
            if float(state_jump_l2) > float(getattr(self.cfg, "prune_state_jump_l2_tol", 0.0)):
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
            if float(delta_rho) > float(getattr(self.cfg, "prune_safe_miss_increase_tol", 0.0)):
                self._block_cooldown[str(proposed["candidate_label"])] = int(self.cfg.prune_cooldown_steps)
                proposed["prune_accept"] = False
                proposed["pruned_baseline"] = reduced_baseline
                proposed["prune_rejection_reason"] = "rho_miss_increase_above_tol"
                evaluated[idx] = proposed
                if proposed_out is None:
                    proposed_out = dict(proposed)
                    rejection_reason_out = "prune_rejected_rho_miss_increase_above_tol"
                continue

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

    def _select_action_exact_v1(
        self,
        *,
        baseline: Mapping[str, Any],
        confirmed: Sequence[Mapping[str, Any]],
        dt: float,
        time_stop: float,
    ) -> tuple[str, Mapping[str, Any] | None]:
        if float(baseline["summary"].rho_miss) <= float(self.cfg.miss_threshold):
            return "stay", None
        if not confirmed:
            return "stay", None
        best_record: dict[str, Any] | None = None
        best_score: float | None = None
        for record in self._sorted_confirmed_by_gain(confirmed):
            if not self._passes_exact_confirm_thresholds(record):
                continue
            scaled_record, forecast = self._select_exact_v1_candidate_step_scale(
                baseline_theta_dot=np.asarray(baseline["theta_dot_step"], dtype=float).reshape(-1),
                selected=record,
                dt=float(dt),
                time_stop=float(time_stop),
            )
            score = self._forecast_tracking_score(forecast=forecast)
            if best_record is None or best_score is None or float(score) < float(best_score) - 1.0e-12:
                best_record = dict(scaled_record)
                best_score = float(score)
        if best_record is None:
            return "stay", None
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

    def run(self) -> ControllerRunArtifacts:
        self._run_wallclock_start = time.perf_counter()
        try:
            self._write_progress(stage="run_start", force=True)
            self._write_partial_payload(stage="run_start")
            for checkpoint_index, time_value in enumerate(self.times):
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
                baseline_exact = self._baseline_geometry(
                    checkpoint_ctx,
                    cache,
                    geometry_memo,
                    step_hamiltonian=step_hamiltonian,
                )
                baseline_for_decision = baseline_exact
                degraded_reason = baseline_exact.get("analytic_noise_degraded_reason")
                decision_backend = "exact"
                decision_noise_mode: str | None = None
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
                self._decrement_prune_cooldowns()
                self._record_prune_histories(baseline=baseline_for_decision)
                prune_candidates: list[dict[str, Any]] = []
                prune_reason = (
                    str(degraded_reason)
                    if degraded_reason is not None
                    else "exact_rho_miss_below_threshold"
                )
                if (
                    time_stop is not None
                    and str(self.cfg.mode) != "off"
                    and str(getattr(self.cfg, "prune_mode", "off")) != "off"
                    and float(baseline_exact["summary"].rho_miss) <= float(self.cfg.miss_threshold)
                    and degraded_reason is None
                ):
                    prune_candidates, prune_reason = self._prune_candidates(
                        checkpoint_index=int(checkpoint_index),
                        baseline=baseline_exact,
                        motion=motion_telemetry,
                    )
                if degraded_reason is not None:
                    controller_lane, controller_lane_reason = "stay", str(degraded_reason)
                else:
                    controller_lane, controller_lane_reason = self._controller_lane(
                        time_stop=time_stop,
                        rho_miss=float(baseline_exact["summary"].rho_miss),
                        prune_candidates_available=bool(prune_candidates),
                        prune_reason=str(prune_reason),
                    )
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
                proposed_selected_override: Mapping[str, Any] | None = None
                proposed_action_kind_override: str | None = None
                if time_stop is None:
                    shortlist = []
                    confirmed = []
                    oracle_confirm_limit = 0
                    oracle_budget_scale = 0.0
                    if str(self.cfg.mode) == "off":
                        decision_backend = "off"
                    action_kind, selected = "stay", None
                elif str(self.cfg.mode) == "off":
                    shortlist = []
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
                    elif prune_error is not None:
                        controller_override_reason = str(prune_error)
                else:
                    shortlist = self._scout_candidates(
                        checkpoint_ctx=checkpoint_ctx,
                        cache=cache,
                        geometry_memo=geometry_memo,
                        baseline=baseline_exact,
                        predicted_displacement=float(predicted_displacement),
                        shortlist_cfg=shortlist_cfg,
                    ) if str(controller_lane) == "append" else []
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
                        if str(self.cfg.mode) == "exact_v1":
                            selection_metric = "exact_forecast_tracking"
                            action_kind, selected = self._select_action_exact_v1(
                                baseline=baseline_for_decision,
                                confirmed=confirmed,
                                dt=float(dt),
                                time_stop=float(time_stop),
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
                    str(self.cfg.mode) in {"oracle_v1", "exact_v1"}
                    and time_stop is not None
                    and str(proposed_action_kind) == "append_candidate"
                    and proposed_selected is not None
                ):
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
                        if str(self.cfg.mode) == "exact_v1":
                            proposed_selected, forecast_selected = self._select_exact_v1_candidate_step_scale(
                                baseline_theta_dot=np.asarray(
                                    baseline_for_decision["theta_dot_step"], dtype=float
                                ).reshape(-1),
                                selected=proposed_selected,
                                dt=float(dt),
                                time_stop=float(time_stop),
                            )
                        else:
                            selected_theta_forecast = np.asarray(
                                proposed_selected["candidate_data"]["theta_aug"]
                                + float(dt) * np.asarray(proposed_selected["theta_dot_aug"], dtype=float),
                                dtype=float,
                            ).reshape(-1)
                            forecast_selected = self._exact_step_forecast(
                                time_stop=float(time_stop),
                                executor=proposed_selected["candidate_data"]["aug_executor"],
                                theta_runtime=selected_theta_forecast,
                            )
                    except Exception as exc:
                        exact_forecast_error = f"{type(exc).__name__}: {exc}"
                        forecast_stay = None
                        forecast_selected = None
                        msg = f"exact_forecast_error: {exact_forecast_error}"
                        degraded_reason = msg if degraded_reason is None else f"{degraded_reason}; {msg}"
                if (
                    decision_override_reason is None
                    and forecast_stay is not None
                    and forecast_selected is not None
                ):
                    forecast_override_reason = self._exact_forecast_override_reason(
                        stay_forecast=forecast_stay,
                        selected_forecast=forecast_selected,
                    )
                    if forecast_override_reason is None:
                        forecast_override_reason = self._exact_v1_forecast_override_reason(
                            stay_forecast=forecast_stay,
                            selected_forecast=forecast_selected,
                            action_kind=str(proposed_action_kind),
                            selected=proposed_selected,
                        )
                    if forecast_override_reason is not None:
                        decision_override_reason = str(forecast_override_reason)
                if decision_override_reason is not None:
                    action_kind, selected = "stay", None
                else:
                    action_kind = str(proposed_action_kind)
                    selected = proposed_selected

                if degraded_reason is not None:
                    self._degraded_checkpoint_count += 1
                logical_before = int(self.current_layout.logical_parameter_count)
                runtime_before = int(self.current_layout.runtime_parameter_count)
                selected_groups_new = 0.0
                selected_gain_ratio = 0.0
                selected_prune_cached_loss: float | None = None
                selected_prune_stagnation_score: float | None = None
                selected_post_prune_state_jump_l2: float | None = None
                selected_candidate_label: str | None = None
                selected_position_id: int | None = None
                if selected is not None:
                    selected_candidate_label = str(selected["candidate_label"])
                    selected_position_id = int(selected["position_id"])
                    selected_groups_new = float(selected.get("groups_new", 0.0))
                    selected_gain_ratio = float(selected.get("gain_ratio", 0.0))
                    if selected.get("cached_prune_loss", None) is not None:
                        selected_prune_cached_loss = float(selected["cached_prune_loss"])
                    if selected.get("stagnation_score", None) is not None:
                        selected_prune_stagnation_score = float(selected["stagnation_score"])
                    if selected.get("post_prune_state_jump_l2", None) is not None:
                        selected_post_prune_state_jump_l2 = float(selected["post_prune_state_jump_l2"])
                selected_step_scale = (
                    None
                    if selected is None or selected.get("candidate_step_scale", None) is None
                    else float(selected["candidate_step_scale"])
                )
                tier_reached = "scout"
                rate_change_l2 = _overlap_l2(np.asarray(baseline_for_decision["theta_dot_step"], dtype=float), self._previous_theta_dot)

                psi_exact = self._exact_state_at(float(time_value))
                energy_exact = float(
                    np.real(
                        np.vdot(
                            np.asarray(psi_exact, dtype=complex).reshape(-1),
                            step_hamiltonian.hmat @ np.asarray(psi_exact, dtype=complex).reshape(-1),
                        )
                    )
                )
                energy_controller = float(baseline_for_decision["summary"].energy)
                fidelity_exact = float(abs(np.vdot(psi_exact, baseline_exact["psi"])) ** 2)
                abs_energy_total_error = float(abs(float(energy_controller) - float(energy_exact)))
                controller_obs = self._observable_snapshot(
                    np.asarray(baseline_exact["psi"], dtype=complex).reshape(-1)
                )
                exact_obs = self._observable_snapshot(np.asarray(psi_exact, dtype=complex).reshape(-1))
                site_occ_controller = np.asarray(controller_obs["site_occupations"], dtype=float)
                site_occ_exact = np.asarray(exact_obs["site_occupations"], dtype=float)
                site_occ_abs_error = np.abs(site_occ_controller - site_occ_exact)
                abs_staggered_error = float(
                    abs(float(controller_obs["staggered"]) - float(exact_obs["staggered"]))
                )
                abs_doublon_error = float(
                    abs(float(controller_obs["doublon"]) - float(exact_obs["doublon"]))
                )
                post_prune_payload: dict[str, Any] | None = None
                fidelity_initial_controller = float(
                    abs(
                        np.vdot(
                            np.asarray(self.psi_initial, dtype=complex).reshape(-1),
                            np.asarray(baseline_exact["psi"], dtype=complex).reshape(-1),
                        )
                    )
                    ** 2
                )
                fidelity_initial_exact = float(
                    abs(
                        np.vdot(
                            np.asarray(self.psi_initial, dtype=complex).reshape(-1),
                            np.asarray(psi_exact, dtype=complex).reshape(-1),
                        )
                    )
                    ** 2
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
                        "stagnation_score": float(item.get("stagnation_score", 0.0)),
                        "motion_mean": float(item.get("motion_mean", 0.0)),
                        "fit_mean": float(item.get("fit_mean", 0.0)),
                        "theta_block_norm": float(item.get("theta_block_norm", 0.0)),
                        "burden": float(item.get("burden", 0.0)),
                        "post_prune_state_jump_l2": (
                            None if item.get("post_prune_state_jump_l2") is None else float(item.get("post_prune_state_jump_l2"))
                        ),
                        "prune_delta_rho_miss": (
                            None if item.get("prune_delta_rho_miss") is None else float(item.get("prune_delta_rho_miss"))
                        ),
                        "prune_accept": (None if item.get("prune_accept") is None else bool(item.get("prune_accept"))),
                        "prune_rejection_reason": item.get("prune_rejection_reason", None),
                    }
                    for item in prune_candidates
                ]

                self._trajectory.append(
                    {
                        "checkpoint_index": int(checkpoint_index),
                        "time": float(time_value),
                        "physical_time": float(step_hamiltonian.physical_time),
                        "action_kind": str(action_kind),
                        "candidate_label": selected_candidate_label,
                        "proposed_action_kind": str(proposed_action_kind),
                        "proposed_candidate_label": proposed_candidate_label,
                        "controller_lane": str(controller_lane),
                        "controller_lane_reason": str(controller_lane_reason),
                        "requested_mode": str(self.cfg.mode),
                        "decision_backend": str(decision_backend),
                        "decision_noise_mode": decision_noise_mode,
                        "oracle_attempted": bool(oracle_attempted),
                        "oracle_decision_used": bool(oracle_decision_used),
                        "oracle_estimate_kind": oracle_estimate_kind,
                        "selection_metric": str(selection_metric),
                        "decision_override_reason": decision_override_reason,
                        "exact_forecast_error": exact_forecast_error,
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
                        "forecast_stay_fidelity_exact_next": (
                            None if forecast_stay is None else float(forecast_stay["fidelity_exact_next"])
                        ),
                        "forecast_selected_fidelity_exact_next": (
                            None
                            if forecast_selected is None
                            else float(forecast_selected["fidelity_exact_next"])
                        ),
                        "forecast_stay_abs_energy_total_error_next": (
                            None
                            if forecast_stay is None
                            else float(forecast_stay["abs_energy_total_error_next"])
                        ),
                        "forecast_selected_abs_energy_total_error_next": (
                            None
                            if forecast_selected is None
                            else float(forecast_selected["abs_energy_total_error_next"])
                        ),
                        "forecast_stay_abs_staggered_error_next": (
                            None
                            if forecast_stay is None
                            else float(forecast_stay["abs_staggered_error_next"])
                        ),
                        "forecast_selected_abs_staggered_error_next": (
                            None
                            if forecast_selected is None
                            else float(forecast_selected["abs_staggered_error_next"])
                        ),
                        "forecast_stay_abs_doublon_error_next": (
                            None
                            if forecast_stay is None
                            else float(forecast_stay["abs_doublon_error_next"])
                        ),
                        "forecast_selected_abs_doublon_error_next": (
                            None
                            if forecast_selected is None
                            else float(forecast_selected["abs_doublon_error_next"])
                        ),
                        "forecast_stay_site_occupations_abs_error_max_next": (
                            None
                            if forecast_stay is None
                            else float(forecast_stay.get("site_occupations_abs_error_max_next", float("nan")))
                        ),
                        "forecast_selected_site_occupations_abs_error_max_next": (
                            None
                            if forecast_selected is None
                            else float(
                                forecast_selected.get("site_occupations_abs_error_max_next", float("nan"))
                            )
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
                        "temporal_refresh_pressure": str(refresh_pressure),
                        "oracle_confirm_limit": int(oracle_confirm_limit),
                        "oracle_budget_scale": float(oracle_budget_scale),
                        "rho_miss": float(baseline_for_decision["summary"].rho_miss),
                        "epsilon_proj_sq": float(baseline_for_decision["summary"].epsilon_proj_sq),
                        "epsilon_step_sq": float(baseline_for_decision["summary"].epsilon_step_sq),
                        "energy_total": float(energy_controller),
                        "energy_total_controller": float(energy_controller),
                        "energy_total_exact": float(energy_exact),
                        "abs_energy_total_error": float(abs_energy_total_error),
                        "fidelity_exact": float(fidelity_exact),
                        "fidelity_initial_controller": float(fidelity_initial_controller),
                        "fidelity_initial_exact": float(fidelity_initial_exact),
                        "staggered": float(controller_obs["staggered"]),
                        "staggered_exact": float(exact_obs["staggered"]),
                        "abs_staggered_error": float(abs_staggered_error),
                        "doublon": float(controller_obs["doublon"]),
                        "doublon_exact": float(exact_obs["doublon"]),
                        "abs_doublon_error": float(abs_doublon_error),
                        "site_occupations": list(controller_obs["site_occupations"]),
                        "site_occupations_exact": list(exact_obs["site_occupations"]),
                        "site_occupations_up": list(controller_obs["n_up_site"]),
                        "site_occupations_up_exact": list(exact_obs["n_up_site"]),
                        "site_occupations_dn": list(controller_obs["n_dn_site"]),
                        "site_occupations_dn_exact": list(exact_obs["n_dn_site"]),
                        "site_occupations_abs_error": [float(x) for x in site_occ_abs_error.tolist()],
                        "site_occupations_abs_error_max": (
                            float(np.max(site_occ_abs_error)) if site_occ_abs_error.size > 0 else float("nan")
                        ),
                        "logical_block_count": int(logical_before),
                        "runtime_parameter_count": int(runtime_before),
                        "selected_noisy_energy_mean": oracle_commit_payload.get("selected_noisy_energy_mean", None),
                        "selected_noisy_energy_stderr": oracle_commit_payload.get("selected_noisy_energy_stderr", None),
                        "selected_noisy_backend_info": oracle_commit_payload.get("selected_noisy_backend_info", None),
                        "stay_noisy_energy_mean": oracle_commit_payload.get("stay_noisy_energy_mean", None),
                        "stay_noisy_energy_stderr": oracle_commit_payload.get("stay_noisy_energy_stderr", None),
                        "stay_noisy_backend_info": oracle_commit_payload.get("stay_noisy_backend_info", None),
                        "selected_noisy_improvement_abs": oracle_commit_payload.get("selected_noisy_improvement_abs", None),
                        "selected_noisy_improvement_ratio": oracle_commit_payload.get("selected_noisy_improvement_ratio", None),
                        "selected_prune_cached_loss": selected_prune_cached_loss,
                        "selected_prune_stagnation_score": selected_prune_stagnation_score,
                        "selected_post_prune_state_jump_l2": selected_post_prune_state_jump_l2,
                        "drive_term_count": int(step_hamiltonian.drive_term_count),
                        "degraded_reason": degraded_reason,
                        "baseline_geometry": dataclass_to_payload(baseline_for_decision["summary"]),
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
                    self.current_theta = np.asarray(
                        candidate_data["theta_aug"] + float(dt) * np.asarray(selected["theta_dot_aug"], dtype=float),
                        dtype=float,
                    ).reshape(-1)
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
                    self._block_motion_history.setdefault(appended_label, [])
                    self._block_fit_history.setdefault(appended_label, [])
                    self._record_theta_dot_history(
                        np.asarray(selected["theta_dot_aug"], dtype=float).reshape(-1)
                    )
                elif str(action_kind) == "prune_coordinate" and selected is not None:
                    tier_reached = "commit"
                    reduced_state = dict(selected["reduced_state"])
                    reduced_baseline = dict(selected["pruned_baseline"])
                    removed_label = str(reduced_state["removed_label"])
                    reduced_psi = np.asarray(reduced_state["reduced_psi"], dtype=complex).reshape(-1)
                    reduced_obs = self._observable_snapshot(reduced_psi)
                    reduced_site_occ = np.asarray(reduced_obs["site_occupations"], dtype=float)
                    reduced_site_occ_abs_error = np.abs(reduced_site_occ - site_occ_exact)
                    post_prune_payload = {
                        "post_prune_energy_total": float(
                            np.real(np.vdot(reduced_psi, step_hamiltonian.hmat @ reduced_psi))
                        ),
                        "post_prune_fidelity_exact": float(abs(np.vdot(psi_exact, reduced_psi)) ** 2),
                        "post_prune_abs_energy_total_error": float(
                            abs(
                                float(np.real(np.vdot(reduced_psi, step_hamiltonian.hmat @ reduced_psi)))
                                - float(energy_exact)
                            )
                        ),
                        "post_prune_staggered": float(reduced_obs["staggered"]),
                        "post_prune_abs_staggered_error": float(
                            abs(float(reduced_obs["staggered"]) - float(exact_obs["staggered"]))
                        ),
                        "post_prune_doublon": float(reduced_obs["doublon"]),
                        "post_prune_abs_doublon_error": float(
                            abs(float(reduced_obs["doublon"]) - float(exact_obs["doublon"]))
                        ),
                        "post_prune_site_occupations": [float(x) for x in reduced_site_occ.tolist()],
                        "post_prune_site_occupations_abs_error": [
                            float(x) for x in reduced_site_occ_abs_error.tolist()
                        ],
                        "post_prune_site_occupations_abs_error_max": (
                            float(np.max(reduced_site_occ_abs_error))
                            if reduced_site_occ_abs_error.size > 0
                            else float("nan")
                        ),
                        "post_prune_baseline_geometry": dataclass_to_payload(reduced_baseline["summary"]),
                    }
                    self.current_terms = list(reduced_state["reduced_terms"])
                    self.current_layout = reduced_state["reduced_layout"]
                    self.current_executor = reduced_state["reduced_executor"]
                    self.current_theta = np.asarray(
                        np.asarray(reduced_state["reduced_theta"], dtype=float).reshape(-1)
                        + float(dt) * np.asarray(reduced_baseline["theta_dot_step"], dtype=float).reshape(-1),
                        dtype=float,
                    ).reshape(-1)
                    self._planning_audit = reduced_state["reduced_planning_audit"]
                    self._block_birth_checkpoint.pop(removed_label, None)
                    self._block_cooldown.pop(removed_label, None)
                    self._block_burden.pop(removed_label, None)
                    self._block_motion_history.pop(removed_label, None)
                    self._block_fit_history.pop(removed_label, None)
                    self._previous_block_theta_snapshot.pop(removed_label, None)
                    self._record_theta_dot_history(
                        np.asarray(reduced_baseline["theta_dot_step"], dtype=float).reshape(-1)
                    )
                else:
                    self.current_theta = np.asarray(
                        self.current_theta + float(dt) * np.asarray(baseline_for_decision["theta_dot_step"], dtype=float),
                        dtype=float,
                    ).reshape(-1)
                    self._record_theta_dot_history(
                        np.asarray(baseline_for_decision["theta_dot_step"], dtype=float).reshape(-1)
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
                    candidate_label=selected_candidate_label,
                    proposed_action_kind=str(proposed_action_kind),
                    proposed_candidate_label=proposed_candidate_label,
                    controller_lane=str(controller_lane),
                    controller_lane_reason=str(controller_lane_reason),
                    position_id=selected_position_id,
                    rho_miss=float(baseline_for_decision["summary"].rho_miss),
                    gain_ratio_selected=float(selected_gain_ratio),
                    prune_cached_loss_selected=(None if selected_prune_cached_loss is None else float(selected_prune_cached_loss)),
                    prune_stagnation_score_selected=(None if selected_prune_stagnation_score is None else float(selected_prune_stagnation_score)),
                    post_prune_state_jump_l2=(None if selected_post_prune_state_jump_l2 is None else float(selected_post_prune_state_jump_l2)),
                    shortlist_size=int(len(shortlist)),
                    tier_reached=str(tier_reached),
                    logical_block_count_before=int(logical_before),
                    logical_block_count_after=int(self.current_layout.logical_parameter_count),
                    runtime_parameter_count_before=int(runtime_before),
                    runtime_parameter_count_after=int(self.current_layout.runtime_parameter_count),
                    rate_change_l2=(None if rate_change_l2 is None else float(rate_change_l2)),
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
                    energy_total_exact=float(energy_exact),
                    abs_energy_total_error=float(abs_energy_total_error),
                    fidelity_exact=float(fidelity_exact),
                    requested_mode=str(self.cfg.mode),
                    decision_backend=str(decision_backend),
                    decision_noise_mode=decision_noise_mode,
                    oracle_decision_used=bool(oracle_decision_used),
                    oracle_attempted=bool(oracle_attempted),
                    oracle_estimate_kind=oracle_estimate_kind,
                    selection_metric=str(selection_metric),
                    decision_override_reason=decision_override_reason,
                    exact_forecast_error=exact_forecast_error,
                    baseline_step_scale=baseline_step_scale,
                    baseline_blend_weight=baseline_blend_weight,
                    baseline_gain_scale=baseline_gain_scale,
                    baseline_proposal_kind=(
                        None if baseline_proposal_kind is None else str(baseline_proposal_kind)
                    ),
                    selected_step_scale=selected_step_scale,
                    forecast_stay_fidelity_exact_next=(
                        None if forecast_stay is None else float(forecast_stay["fidelity_exact_next"])
                    ),
                    forecast_selected_fidelity_exact_next=(
                        None
                        if forecast_selected is None
                        else float(forecast_selected["fidelity_exact_next"])
                    ),
                    forecast_stay_abs_energy_total_error_next=(
                        None
                        if forecast_stay is None
                        else float(forecast_stay["abs_energy_total_error_next"])
                    ),
                    forecast_selected_abs_energy_total_error_next=(
                        None
                        if forecast_selected is None
                        else float(forecast_selected["abs_energy_total_error_next"])
                    ),
                    forecast_stay_abs_staggered_error_next=(
                        None
                        if forecast_stay is None
                        else float(forecast_stay["abs_staggered_error_next"])
                    ),
                    forecast_selected_abs_staggered_error_next=(
                        None
                        if forecast_selected is None
                        else float(forecast_selected["abs_staggered_error_next"])
                    ),
                    forecast_stay_abs_doublon_error_next=(
                        None
                        if forecast_stay is None
                        else float(forecast_stay["abs_doublon_error_next"])
                    ),
                    forecast_selected_abs_doublon_error_next=(
                        None
                        if forecast_selected is None
                        else float(forecast_selected["abs_doublon_error_next"])
                    ),
                    forecast_stay_site_occupations_abs_error_max_next=(
                        None
                        if forecast_stay is None
                        else float(forecast_stay.get("site_occupations_abs_error_max_next", float("nan")))
                    ),
                    forecast_selected_site_occupations_abs_error_max_next=(
                        None
                        if forecast_selected is None
                        else float(
                            forecast_selected.get("site_occupations_abs_error_max_next", float("nan"))
                        )
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
                )
                self._write_partial_payload(stage="checkpoint_done")

            append_count = int(sum(1 for row in self._ledger if str(row.get("action_kind")) == "append_candidate"))
            prune_count = int(sum(1 for row in self._ledger if str(row.get("action_kind")) == "prune_coordinate"))
            stay_count = int(sum(1 for row in self._ledger if str(row.get("action_kind")) == "stay"))
            exact_decision_checkpoints = int(sum(1 for row in self._ledger if str(row.get("decision_backend")) == "exact"))
            oracle_decision_checkpoints = int(sum(1 for row in self._ledger if str(row.get("decision_backend")) == "oracle"))
            oracle_attempted_checkpoints = int(sum(1 for row in self._ledger if bool(row.get("oracle_attempted", False))))
            decision_override_count = int(
                sum(1 for row in self._ledger if row.get("decision_override_reason") not in {None, ""})
            )
            exact_forecast_veto_count = int(
                sum(
                    1
                    for row in self._ledger
                    if str(row.get("decision_override_reason", "")).startswith("exact_forecast_")
                )
            )
            executed_backends = sorted({str(row.get("decision_backend", "exact")) for row in self._ledger}) or ["exact"]
            final_row = self._trajectory[-1] if self._trajectory else {}
            staggered_error_vals = [
                float(row.get("abs_staggered_error", float("nan"))) for row in self._trajectory
            ]
            doublon_error_vals = [
                float(row.get("abs_doublon_error", float("nan"))) for row in self._trajectory
            ]
            site_occupation_error_vals = [
                float(row.get("site_occupations_abs_error_max", float("nan"))) for row in self._trajectory
            ]
            baseline_step_scale_vals = [
                float(row.get("baseline_step_scale"))
                for row in self._trajectory
                if row.get("baseline_step_scale", None) is not None
            ]
            baseline_blend_weight_vals = [
                float(row.get("baseline_blend_weight"))
                for row in self._trajectory
                if row.get("baseline_blend_weight", None) is not None
            ]
            baseline_gain_scale_vals = [
                float(row.get("baseline_gain_scale"))
                for row in self._trajectory
                if row.get("baseline_gain_scale", None) is not None
            ]
            baseline_proposal_kind_vals = [
                str(row.get("baseline_proposal_kind"))
                for row in self._trajectory
                if row.get("baseline_proposal_kind", None) not in {None, ""}
            ]
            final_ledger_row = self._ledger[-1] if self._ledger else {}
            oracle_backend_infos: list[dict[str, Any]] = []
            for row in self._trajectory:
                for key in ("selected_noisy_backend_info", "stay_noisy_backend_info"):
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
                "requested_decision_backend": (
                    "oracle"
                    if str(self.cfg.mode) == "oracle_v1"
                    else ("off" if str(self.cfg.mode) == "off" else "exact")
                ),
                "status": ("completed_with_fallback" if int(self._degraded_checkpoint_count) > 0 else "completed"),
                "decision_backend": (
                    executed_backends[0]
                    if len(executed_backends) == 1
                    else "mixed"
                ),
                "executed_decision_backends": list(executed_backends),
                "decision_noise_mode": (
                    None
                    if oracle_attempted_checkpoints <= 0 or self._oracle_base_config is None
                    else str(self._oracle_base_config.noise_mode)
                ),
                "oracle_estimate_kind": (
                    None if oracle_attempted_checkpoints <= 0 else self._oracle_estimate_kind()
                ),
                "oracle_selection_policy": str(self.cfg.oracle_selection_policy),
                "confirm_score_mode": str(getattr(self.cfg, "confirm_score_mode", "exact_gain_ratio")),
                "prune_mode": str(getattr(self.cfg, "prune_mode", "off")),
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
                "exact_forecast_tracking_fidelity_defect_weight": float(
                    getattr(self.cfg, "exact_forecast_tracking_fidelity_defect_weight", 1.0)
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
                "decision_override_count": int(decision_override_count),
                "exact_forecast_veto_count": int(exact_forecast_veto_count),
                "append_count": int(append_count),
                "prune_count": int(prune_count),
                "stay_count": int(stay_count),
                "exact_decision_checkpoints": int(exact_decision_checkpoints),
                "oracle_decision_checkpoints": int(oracle_decision_checkpoints),
                "oracle_attempted_checkpoints": int(oracle_attempted_checkpoints),
                "degraded_checkpoints": int(self._degraded_checkpoint_count),
                "raw_group_cache_hits": int(final_ledger_row.get("raw_group_cache_hits", 0)),
                "raw_group_cache_misses": int(final_ledger_row.get("raw_group_cache_misses", 0)),
                "raw_group_cache_extensions": int(final_ledger_row.get("raw_group_cache_extensions", 0)),
                "final_logical_block_count": int(self.current_layout.logical_parameter_count),
                "final_runtime_parameter_count": int(self.current_layout.runtime_parameter_count),
                "final_fidelity_exact": float(final_row.get("fidelity_exact", float("nan"))),
                "final_abs_energy_total_error": float(final_row.get("abs_energy_total_error", float("nan"))),
                "final_staggered": float(final_row.get("staggered", float("nan"))),
                "final_staggered_exact": float(final_row.get("staggered_exact", float("nan"))),
                "final_abs_staggered_error": float(final_row.get("abs_staggered_error", float("nan"))),
                "max_abs_staggered_error": float(np.nanmax(np.asarray(staggered_error_vals, dtype=float))),
                "final_doublon": float(final_row.get("doublon", float("nan"))),
                "final_doublon_exact": float(final_row.get("doublon_exact", float("nan"))),
                "final_abs_doublon_error": float(final_row.get("abs_doublon_error", float("nan"))),
                "max_abs_doublon_error": float(np.nanmax(np.asarray(doublon_error_vals, dtype=float))),
                "final_site_occupations": list(final_row.get("site_occupations", [])),
                "final_site_occupations_exact": list(final_row.get("site_occupations_exact", [])),
                "final_site_occupations_abs_error_max": float(
                    final_row.get("site_occupations_abs_error_max", float("nan"))
                ),
                "max_abs_site_occupations_error": float(
                    np.nanmax(np.asarray(site_occupation_error_vals, dtype=float))
                ),
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
            }
            reference = {
                "kind": (
                    "driven_piecewise_constant_reference_from_replay_seed"
                    if self._drive_config is not None
                    else "static_exact_reference_from_replay_seed"
                ),
                "initial_state": "stage_result.psi_final",
                "times": [float(x) for x in self.times.tolist()],
                "drive_profile": (None if self._drive_profile is None else dict(self._drive_profile)),
                "reference_method": (
                    None
                    if self._drive_config is None
                    else str(reference_method_name(str(self._drive_config.drive_time_sampling)))
                ),
                "reference_steps_multiplier": (
                    1
                    if self._drive_config is None
                    else int(self._drive_config.exact_steps_multiplier)
                ),
                "projection_time_sampling": (
                    "left"
                    if self._drive_config is None
                    else str(self._drive_config.drive_time_sampling)
                ),
                "geometry_sample_time_policy": (
                    "checkpoint_time"
                    if self._drive_config is None
                    else f"interval_{str(self._drive_config.drive_time_sampling).strip().lower()}_plus_t0_with_final_endpoint_fallback"
                ),
            }
            self._write_progress(
                stage="run_complete",
                force=True,
                status="completed",
                summary=summary,
            )
            self._write_partial_payload(
                status="completed",
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
]
