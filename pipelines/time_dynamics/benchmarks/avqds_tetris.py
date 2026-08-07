#!/usr/bin/env python3
"""Published AVQDS(T) Method-3 TETRIS comparator.

This comparator keeps the continuous McLachlan right-hand side used by AVQDS.
The ``T`` denotes TETRIS layer growth: singleton pool scores are ranked and
mutually qubit-disjoint generators are appended together.  The historical
``dyn_avqds_t`` product-formula-target diagnostic is intentionally separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.benchmarks import common
from pipelines.time_dynamics.benchmarks.common import (
    _build_layout_for_terms,
    _compile_audit_from_resources,
    _compiled_executor_for_terms,
    _copy_theta_by_layout_blocks,
    _float_or_none,
    _generic_parameter_manifest,
    _max_or_none,
    _metadata_float,
    _metadata_optional_int,
    _native_hamiltonian_flow,
    _normalize_state,
    _prepare_scaffold_state,
    _scaffold_resources_for_layouts,
    _state_diagnostic_row,
    _trajectory_summary,
)
from pipelines.time_dynamics.normalized_pauli_pool import (
    NORMALIZED_POOL_FULL_META_CHILDREN,
    NORMALIZED_POOL_HAMILTONIAN_DRIVE,
    NormalizedPauliPoolContract,
    build_normalized_pauli_pool,
)
from pipelines.time_dynamics.redundancy_stress import (
    inject_zero_angle_redundancy_layers,
    redundancy_stress_config_from_metadata,
)
from pipelines.time_dynamics.tables.dynamics_benchmark_contract import (
    DynamicsBenchmarkCase,
    DynamicsBenchmarkRow,
    build_dynamics_tuning_provenance,
    json_safe,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


AVQDS_TETRIS_ALGORITHM_ID = "dyn_avqds_tetris"
AVQDS_TETRIS_METHOD = "method3_disjoint_layer"
AVQDS_TETRIS_DEFAULT_DISTANCE_SQ_THRESHOLD = 1.0e-3
AVQDS_TETRIS_DEFAULT_EIGENVALUE_CUTOFF = 1.0e-6
AVQDS_TETRIS_DEFAULT_MIN_DISTANCE_SQ_GAIN = 1.0e-12
AVQDS_TETRIS_POOL_SOURCES = frozenset({"hamiltonian_pauli", "runtime_candidate_pool"})


@dataclass(frozen=True)
class TetrisPoolAtom:
    """One normalized Pauli-string generator eligible for TETRIS packing."""

    pool_index: int
    pauli_exyz: str
    qubit_support: tuple[int, ...]
    source_labels: tuple[str, ...]
    nq: int
    repr_mode: str = "JW"


@dataclass(frozen=True)
class TetrisCandidateScore:
    """Checkpoint-local singleton reduction in the McLachlan distance."""

    atom: TetrisPoolAtom
    distance_sq: float
    distance_sq_gain: float
    retained_rank: int
    parameter_count: int


@dataclass(frozen=True)
class AVQDSTangentGeometry:
    """Continuous-RHS McLachlan geometry at one state and support."""

    state: np.ndarray
    theta_dot: np.ndarray
    horizontal_tangents: np.ndarray
    horizontal_rhs: np.ndarray
    metric: np.ndarray
    force: np.ndarray
    eigenvalues: np.ndarray
    retained_mask: np.ndarray
    distance_sq: float
    variance: float
    solve_residual_norm: float

    @property
    def retained_rank(self) -> int:
        return int(np.count_nonzero(self.retained_mask))

    @property
    def parameter_count(self) -> int:
        return int(self.theta_dot.size)

    def to_step_dict(self, *, eigenvalue_cutoff: float) -> dict[str, Any]:
        positive = [float(value) for value in self.eigenvalues if float(value) > 0.0]
        kept = [
            float(value)
            for value, retain in zip(self.eigenvalues, self.retained_mask)
            if bool(retain)
        ]
        condition = None
        if kept:
            condition = float(max(kept) / min(kept))
        rhs_norm = float(np.linalg.norm(self.horizontal_rhs))
        residual_norm = float(np.sqrt(max(0.0, 0.5 * self.distance_sq)))
        formula_distance_sq = float(
            max(0.0, 2.0 * (self.variance - float(np.dot(self.force, self.theta_dot))))
        )
        return {
            "theta_dot": [float(value) for value in self.theta_dot.tolist()],
            "mclachlan_distance_sq": float(self.distance_sq),
            "mclachlan_distance_sq_formula": float(formula_distance_sq),
            "mclachlan_distance_sq_identity_abs_delta": float(
                abs(self.distance_sq - formula_distance_sq)
            ),
            "rhs_norm": float(rhs_norm),
            "rhs_residual_norm": float(residual_norm),
            "rhs_residual_ratio": (
                0.0 if rhs_norm <= 1.0e-15 else float(residual_norm / rhs_norm)
            ),
            "variance": float(self.variance),
            "eigenvalue_cutoff": float(eigenvalue_cutoff),
            "retained_rank": int(self.retained_rank),
            "parameter_count": int(self.parameter_count),
            "metric_min_eigenvalue": None if not positive else float(min(positive)),
            "metric_max_eigenvalue": None if not positive else float(max(positive)),
            "retained_condition_estimate": condition,
            "metric_symmetry_max_abs": (
                float(np.max(np.abs(self.metric - self.metric.T)))
                if self.metric.size
                else 0.0
            ),
            "force_norm": float(np.linalg.norm(self.force)),
            "solve_residual_norm": float(self.solve_residual_norm),
            "linear_solve_status": (
                "no_parameters" if self.parameter_count == 0 else "absolute_eigenvalue_truncation"
            ),
            "linear_solve_count": 1,
            "state_prep_count": 1,
            "dense_reference_kind": "projective_qgt_absolute_eigenvalue_truncation",
            "success": True,
        }


def initial_avqds_tetris_variational_bundle(
    *,
    case: DynamicsBenchmarkCase,
    runtime_input: Any,
    flow: common.NativeHamiltonianFlow,
) -> tuple[tuple[Any, ...], Any, np.ndarray, np.ndarray, CompiledAnsatzExecutor, Any, dict[str, Any]]:
    """Build the comparator's seed ANZATS, including an optional shared stress fixture."""

    state = common._ap_state_for_runtime_input(runtime_input)
    drive_augmentation = common.augment_state_with_drive_aligned_generator(
        state,
        hamiltonian=flow.hamiltonian,
        enabled=bool(flow.drive_enabled),
    )
    state = drive_augmentation.state
    stress_config = redundancy_stress_config_from_metadata(case.metadata)
    stress_contract = None
    if stress_config.enabled:
        stress_contract = build_normalized_pauli_pool(
            profile=str(stress_config.pool_profile),
            static_poly=flow.hamiltonian.static_poly,
            drive_poly=flow.hamiltonian.drive_poly,
            candidate_pool_terms=tuple(
                getattr(runtime_input, "candidate_pool_terms", ()) or ()
            ),
        )
    stress_result = inject_zero_angle_redundancy_layers(
        state,
        pool_contract=stress_contract,
        config=stress_config,
    )
    state = stress_result.state
    terms = tuple(state.terms)
    if not terms:
        raise ValueError("AVQDS(T) requires selected seed ANZATS terms")
    layout = state.layout
    theta = np.asarray(state.theta_runtime, dtype=float).reshape(-1)
    if int(theta.size) != int(getattr(layout, "runtime_parameter_count")):
        raise ValueError(
            "AVQDS(T) redundancy-stress theta/layout mismatch: "
            f"{theta.size} vs {layout.runtime_parameter_count}"
        )
    psi_ref = _normalize_state(state.psi_ref)
    return (
        terms,
        layout,
        theta,
        psi_ref,
        state.executor,
        drive_augmentation,
        dict(stress_result.receipt),
    )


def solve_avqds_projective_geometry(
    *,
    executor: CompiledAnsatzExecutor,
    psi_ref: np.ndarray,
    theta_runtime: np.ndarray,
    hmat: np.ndarray,
    eigenvalue_cutoff: float,
) -> AVQDSTangentGeometry:
    """Solve the published projective McLachlan equations by truncation."""

    theta = np.asarray(theta_runtime, dtype=float).reshape(-1)
    psi_raw, tangent_rows = executor.prepare_state_with_runtime_tangents(theta, psi_ref)
    psi = _normalize_state(psi_raw)
    hpsi = np.asarray(hmat, dtype=complex) @ psi
    energy = float(np.real(np.vdot(psi, hpsi)))
    horizontal_rhs = -1.0j * (hpsi - energy * psi)
    variance = float(max(0.0, np.real(np.vdot(horizontal_rhs, horizontal_rhs))))

    if theta.size == 0:
        empty = np.zeros((psi.size, 0), dtype=complex)
        return AVQDSTangentGeometry(
            state=psi,
            theta_dot=np.zeros(0, dtype=float),
            horizontal_tangents=empty,
            horizontal_rhs=horizontal_rhs,
            metric=np.zeros((0, 0), dtype=float),
            force=np.zeros(0, dtype=float),
            eigenvalues=np.zeros(0, dtype=float),
            retained_mask=np.zeros(0, dtype=bool),
            distance_sq=float(2.0 * variance),
            variance=float(variance),
            solve_residual_norm=0.0,
        )

    tangent_matrix = np.column_stack(
        [np.asarray(tangent_rows[index], dtype=complex).reshape(-1) for index in range(theta.size)]
    )
    overlaps = psi.conj() @ tangent_matrix
    horizontal_tangents = tangent_matrix - psi[:, None] * overlaps[None, :]
    metric = np.real(horizontal_tangents.conj().T @ horizontal_tangents)
    metric = 0.5 * (metric + metric.T)
    force = np.real(horizontal_tangents.conj().T @ horizontal_rhs)
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    cutoff = float(max(0.0, eigenvalue_cutoff))
    retained = np.asarray(eigenvalues > cutoff, dtype=bool)
    projected_force = eigenvectors.T @ force
    solved_modes = np.zeros_like(projected_force, dtype=float)
    solved_modes[retained] = projected_force[retained] / eigenvalues[retained]
    theta_dot = np.asarray(eigenvectors @ solved_modes, dtype=float).reshape(-1)
    residual = horizontal_tangents @ theta_dot - horizontal_rhs
    distance_sq = float(max(0.0, 2.0 * np.real(np.vdot(residual, residual))))
    solve_residual = metric @ theta_dot - force
    return AVQDSTangentGeometry(
        state=psi,
        theta_dot=theta_dot,
        horizontal_tangents=horizontal_tangents,
        horizontal_rhs=horizontal_rhs,
        metric=metric,
        force=force,
        eigenvalues=np.asarray(eigenvalues, dtype=float),
        retained_mask=retained,
        distance_sq=distance_sq,
        variance=float(variance),
        solve_residual_norm=float(np.linalg.norm(solve_residual)),
    )


def select_avqds_method1_candidate(
    scores: Sequence[TetrisCandidateScore],
    *,
    min_distance_sq_gain: float,
) -> tuple[TetrisCandidateScore, ...]:
    """Original AVQDS growth: choose only the best useful singleton."""

    ranked = _rank_useful_scores(scores, min_distance_sq_gain=min_distance_sq_gain)
    return tuple(ranked[:1])


def select_tetris_method3_layer(
    scores: Sequence[TetrisCandidateScore],
    *,
    min_distance_sq_gain: float,
    max_layer_width: int | None = None,
) -> tuple[TetrisCandidateScore, ...]:
    """Greedily pack score-ranked generators with disjoint qubit support."""

    if max_layer_width is not None and int(max_layer_width) <= 0:
        return ()
    selected: list[TetrisCandidateScore] = []
    occupied: set[int] = set()
    for score in _rank_useful_scores(
        scores,
        min_distance_sq_gain=min_distance_sq_gain,
    ):
        support = set(int(qubit) for qubit in score.atom.qubit_support)
        if not support or occupied.intersection(support):
            continue
        selected.append(score)
        occupied.update(support)
        if max_layer_width is not None and len(selected) >= int(max_layer_width):
            break
    return tuple(selected)


def _rank_useful_scores(
    scores: Sequence[TetrisCandidateScore],
    *,
    min_distance_sq_gain: float,
) -> list[TetrisCandidateScore]:
    threshold = float(max(0.0, min_distance_sq_gain))
    useful = [score for score in scores if float(score.distance_sq_gain) >= threshold]
    useful = [score for score in useful if float(score.distance_sq_gain) > 0.0]
    useful.sort(
        key=lambda score: (
            -float(score.distance_sq_gain),
            str(score.atom.pauli_exyz),
            int(score.atom.pool_index),
        )
    )
    return useful


def _metadata_text(case: DynamicsBenchmarkCase, key: str, default: str) -> str:
    metadata = case.metadata if isinstance(case.metadata, Mapping) else {}
    value = str(metadata.get(key, default)).strip().lower()
    return value


def _pauli_support(pauli_exyz: str) -> tuple[int, ...]:
    return tuple(
        int(index)
        for index, letter in enumerate(str(pauli_exyz).lower())
        if letter not in {"e", "i"}
    )


def build_tetris_pool_contract(
    *,
    flow: common.NativeHamiltonianFlow,
    runtime_input: Any,
    pool_source: str,
    candidate_limit: int | None = None,
) -> NormalizedPauliPoolContract:
    """Build the shared normalized pool receipt used by AVQDS(T)."""

    source = str(pool_source).strip().lower()
    if source not in AVQDS_TETRIS_POOL_SOURCES:
        raise ValueError(
            f"avqds_tetris_pool_source must be one of {sorted(AVQDS_TETRIS_POOL_SOURCES)}, "
            f"got {pool_source!r}"
        )
    profile = (
        NORMALIZED_POOL_HAMILTONIAN_DRIVE
        if source == "hamiltonian_pauli"
        else NORMALIZED_POOL_FULL_META_CHILDREN
    )
    contract = build_normalized_pauli_pool(
        profile=profile,
        static_poly=flow.hamiltonian.static_poly,
        drive_poly=flow.hamiltonian.drive_poly,
        candidate_pool_terms=tuple(
            getattr(runtime_input, "candidate_pool_terms", ()) or ()
        ),
    )
    return contract.limited(candidate_limit)


def _tetris_atoms_from_contract(
    contract: NormalizedPauliPoolContract,
) -> tuple[TetrisPoolAtom, ...]:
    return tuple(
        TetrisPoolAtom(
            pool_index=int(index),
            pauli_exyz=str(atom.pauli_exyz),
            qubit_support=_pauli_support(atom.pauli_exyz),
            source_labels=tuple(str(label) for label in atom.source_labels),
            nq=int(atom.nq),
            repr_mode=str(atom.repr_mode),
        )
        for index, atom in enumerate(contract.atoms)
    )


def enumerate_tetris_pool_atoms(
    *,
    flow: common.NativeHamiltonianFlow,
    runtime_input: Any,
    pool_source: str,
    candidate_limit: int | None = None,
) -> tuple[TetrisPoolAtom, ...]:
    """Build the fixed normalized Pauli-string pool used by AVQDS(T)."""

    return _tetris_atoms_from_contract(
        build_tetris_pool_contract(
            flow=flow,
            runtime_input=runtime_input,
            pool_source=pool_source,
            candidate_limit=candidate_limit,
        )
    )


def _term_for_atom(atom: TetrisPoolAtom, *, label: str) -> AnsatzTerm:
    return AnsatzTerm(
        label=str(label),
        polynomial=PauliPolynomial(
            str(atom.repr_mode),
            [PauliTerm(int(atom.nq), ps=str(atom.pauli_exyz), pc=1.0)],
        ),
        execution_mode="termwise_product",
    )


def _trial_support(
    *,
    current_terms: Sequence[Any],
    layout: Any,
    theta_runtime: np.ndarray,
    psi_ref: np.ndarray,
    atoms: Sequence[TetrisPoolAtom],
    label_prefix: str,
) -> tuple[tuple[Any, ...], Any, np.ndarray, CompiledAnsatzExecutor]:
    appended = tuple(
        _term_for_atom(atom, label=f"{label_prefix}::{index}::{atom.pauli_exyz}")
        for index, atom in enumerate(atoms)
    )
    terms = tuple(current_terms) + appended
    new_layout = _build_layout_for_terms(terms, reference_layout=layout)
    new_theta = _copy_theta_by_layout_blocks(
        old_theta=theta_runtime,
        old_layout=layout,
        new_layout=new_layout,
    )
    executor = _compiled_executor_for_terms(terms, new_layout)
    if int(new_theta.size) != int(theta_runtime.size + len(atoms)):
        raise ValueError(
            "AVQDS(TETRIS) append did not add exactly one runtime parameter per Pauli atom"
        )
    return terms, new_layout, new_theta, executor


def _candidate_score_rows(
    *,
    atoms: Sequence[TetrisPoolAtom],
    current_terms: Sequence[Any],
    layout: Any,
    theta_runtime: np.ndarray,
    psi_ref: np.ndarray,
    hmat: np.ndarray,
    base_geometry: AVQDSTangentGeometry,
    eigenvalue_cutoff: float,
    interval_index: int,
    growth_iteration: int,
) -> tuple[list[TetrisCandidateScore], list[dict[str, Any]]]:
    scores: list[TetrisCandidateScore] = []
    rows: list[dict[str, Any]] = []
    for atom in atoms:
        _terms, _layout, trial_theta, trial_executor = _trial_support(
            current_terms=current_terms,
            layout=layout,
            theta_runtime=theta_runtime,
            psi_ref=psi_ref,
            atoms=(atom,),
            label_prefix=(
                f"avqds_tetris_trial_i{int(interval_index)}_g{int(growth_iteration)}_p{int(atom.pool_index)}"
            ),
        )
        geometry = solve_avqds_projective_geometry(
            executor=trial_executor,
            psi_ref=psi_ref,
            theta_runtime=trial_theta,
            hmat=hmat,
            eigenvalue_cutoff=eigenvalue_cutoff,
        )
        gain = float(base_geometry.distance_sq - geometry.distance_sq)
        score = TetrisCandidateScore(
            atom=atom,
            distance_sq=float(geometry.distance_sq),
            distance_sq_gain=float(gain),
            retained_rank=int(geometry.retained_rank),
            parameter_count=int(geometry.parameter_count),
        )
        scores.append(score)
        rows.append(
            {
                "interval_index": int(interval_index),
                "growth_iteration": int(growth_iteration),
                "candidate_pool_index": int(atom.pool_index),
                "candidate_pauli_exyz": str(atom.pauli_exyz),
                "candidate_qubit_support": [int(qubit) for qubit in atom.qubit_support],
                "candidate_source_labels": list(atom.source_labels),
                "mclachlan_distance_sq_base": float(base_geometry.distance_sq),
                "mclachlan_distance_sq": float(geometry.distance_sq),
                "mclachlan_distance_sq_gain": float(gain),
                "retained_rank": int(geometry.retained_rank),
                "runtime_parameter_count": int(geometry.parameter_count),
            }
        )
    return scores, rows


def _build_correctness_sidecar(
    *,
    case: DynamicsBenchmarkCase,
    trajectory: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    layer_events: Sequence[Mapping[str, Any]],
    candidate_evaluations: Sequence[Mapping[str, Any]],
    state_norms: Sequence[float],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    finite = common._json_numeric_values_are_finite(
        {
            "trajectory": list(trajectory),
            "steps": list(steps),
            "layer_events": list(layer_events),
            "candidate_evaluations": list(candidate_evaluations),
            "state_norms": list(state_norms),
        }
    )
    checks.append(
        common._check_payload(
            check_id="finite_trajectory_and_tetris_diagnostics",
            check_type="invariant_correctness",
            passed=bool(steps) and bool(finite),
            details={"step_count": len(steps), "trajectory_points": len(trajectory)},
        )
    )
    overlap_bad: list[dict[str, Any]] = []
    for event in layer_events:
        supports = [set(int(q) for q in support) for support in event.get("qubit_supports", ())]
        for left in range(len(supports)):
            for right in range(left + 1, len(supports)):
                overlap = sorted(supports[left].intersection(supports[right]))
                if overlap:
                    overlap_bad.append(
                        {
                            "interval_index": int(event.get("interval_index", -1)),
                            "growth_iteration": int(event.get("growth_iteration", -1)),
                            "overlap": overlap,
                        }
                    )
    checks.append(
        common._check_payload(
            check_id="tetris_layer_qubit_disjointness",
            check_type="method3_selection_correctness",
            passed=not overlap_bad,
            details={"bad_layers": overlap_bad, "layer_count": len(layer_events)},
        )
    )
    distance_bad: list[int] = []
    distance_identity_bad: list[int] = []
    for step in steps:
        distance = _float_or_none(step.get("mclachlan_distance_sq"))
        variance = _float_or_none(step.get("variance"))
        if distance is None or distance < -1.0e-12 or variance is None or variance < -1.0e-12:
            distance_bad.append(int(step.get("interval_index", -1)))
        identity_delta = _float_or_none(step.get("mclachlan_distance_sq_identity_abs_delta"))
        if identity_delta is None or identity_delta > 1.0e-9:
            distance_identity_bad.append(int(step.get("interval_index", -1)))
    checks.append(
        common._check_payload(
            check_id="projective_mclachlan_distance_and_truncated_solve",
            check_type="dense_reference_component_parity",
            passed=not distance_bad and not distance_identity_bad,
            details={
                "bad_interval_indices": distance_bad,
                "distance_identity_bad_interval_indices": distance_identity_bad,
                "solve_kind": "projective_qgt_absolute_eigenvalue_truncation",
            },
        )
    )
    norm_deviation = max((abs(float(value) - 1.0) for value in state_norms), default=0.0)
    checks.append(
        common._check_payload(
            check_id="state_norm_preservation",
            check_type="invariant_correctness",
            passed=bool(state_norms) and norm_deviation <= 1.0e-10,
            details={"state_norm_count": len(state_norms), "max_norm_deviation": norm_deviation},
        )
    )
    passed = common._checks_pass(checks)
    return json_safe(
        {
            "schema": "avqds_tetris_correctness_v1",
            "algorithm_id": AVQDS_TETRIS_ALGORITHM_ID,
            "family": str(case.family),
            "case_id": str(case.case_id),
            "sidecar_name": common.CORRECTNESS_SIDECAR_FILENAMES[AVQDS_TETRIS_ALGORITHM_ID],
            "support_scope": "continuous_rhs_projective_solve_tetris_method3_growth_and_invariants",
            "sidecar_kind": "dense_reference_component_parity_and_method3_invariant_correctness",
            "status": "ok" if passed else "failed",
            "passed": bool(passed),
            "required_status": "passed",
            "check_count": int(len(checks)),
            "checks": checks,
            "exact_data_policy": "benchmark_exact_fields_reporting_only_not_rhs_or_tetris_decision",
            "controller_decisions_modified": False,
            "exact_reference_controller_inputs": False,
        }
    )


def _build_avqds_tetris_payload(
    *,
    case: DynamicsBenchmarkCase,
    runtime_input: Any,
    command: Sequence[str],
) -> dict[str, Any]:
    flow = _native_hamiltonian_flow(case, runtime_input)
    hamiltonian_terms = flow.terms_for_interval(
        float(flow.times[0]),
        float(flow.times[min(1, len(flow.times) - 1)]),
    )
    (
        current_terms,
        layout,
        theta,
        psi_ref,
        executor,
        drive_aligned_ansatz,
        redundancy_stress,
    ) = initial_avqds_tetris_variational_bundle(
        case=case,
        runtime_input=runtime_input,
        flow=flow,
    )
    times = np.asarray(flow.times, dtype=float)
    if times.size < 2:
        raise ValueError("AVQDS(TETRIS) requires at least two time points")
    threshold = _metadata_float(
        case,
        "avqds_tetris_mclachlan_distance_sq_threshold",
        AVQDS_TETRIS_DEFAULT_DISTANCE_SQ_THRESHOLD,
        minimum=0.0,
    )
    eigenvalue_cutoff = _metadata_float(
        case,
        "avqds_tetris_eigenvalue_cutoff",
        AVQDS_TETRIS_DEFAULT_EIGENVALUE_CUTOFF,
        minimum=0.0,
    )
    min_gain = _metadata_float(
        case,
        "avqds_tetris_min_distance_sq_gain",
        AVQDS_TETRIS_DEFAULT_MIN_DISTANCE_SQ_GAIN,
        minimum=0.0,
    )
    candidate_limit = _metadata_optional_int(
        case,
        "avqds_tetris_candidate_limit",
        None,
        minimum=1,
    )
    max_layer_width = _metadata_optional_int(
        case,
        "avqds_tetris_max_layer_width",
        None,
        minimum=1,
    )
    max_growth_layers = _metadata_optional_int(
        case,
        "avqds_tetris_max_growth_layers_per_checkpoint",
        None,
        minimum=1,
    )
    pool_source = _metadata_text(
        case,
        "avqds_tetris_pool_source",
        "hamiltonian_pauli",
    )
    pool_contract = build_tetris_pool_contract(
        flow=flow,
        runtime_input=runtime_input,
        pool_source=pool_source,
        candidate_limit=candidate_limit,
    )
    pool = _tetris_atoms_from_contract(pool_contract)
    observable_context = dict(flow.observable_context or {})
    exact_states = flow.exact_states
    theta_current = np.asarray(theta, dtype=float).reshape(-1)
    current_state = _prepare_scaffold_state(executor, psi_ref, theta_current)
    initial_layout = layout
    trajectory: list[dict[str, Any]] = [
        _state_diagnostic_row(
            checkpoint_index=0,
            time_value=float(times[0]),
            method="generic_avqds_tetris",
            method_kind="avqds_tetris",
            state=current_state,
            exact_state=exact_states[0],
            hmat=flow.hmat_at_time(float(times[0])),
            **observable_context,
            extra={
                "runtime_parameter_count": int(theta_current.size),
                "logical_block_count": int(getattr(layout, "logical_parameter_count")),
                "mclachlan_distance_sq": None,
                "tetris_layers_added": 0,
                "tetris_generators_added": 0,
                "append_accepted": None,
            },
        )
    ]
    steps: list[dict[str, Any]] = []
    layer_events: list[dict[str, Any]] = []
    candidate_evaluations: list[dict[str, Any]] = []
    interval_layouts: list[Any] = []
    state_norms = [float(np.linalg.norm(current_state))]
    unsupported_checkpoints: list[dict[str, Any]] = []
    total_geometry_solves = 0
    global_layer_index = 0

    for interval_index, (left, right) in enumerate(zip(times[:-1], times[1:])):
        dt = float(right - left)
        hmat = flow.hmat_for_interval(float(left), float(right))
        geometry = solve_avqds_projective_geometry(
            executor=executor,
            psi_ref=psi_ref,
            theta_runtime=theta_current,
            hmat=hmat,
            eigenvalue_cutoff=eigenvalue_cutoff,
        )
        total_geometry_solves += 1
        interval_layer_count = 0
        interval_generator_count = 0
        checkpoint_supported = True
        unsupported_reason = None

        while float(geometry.distance_sq) >= float(threshold):
            if max_growth_layers is not None and interval_layer_count >= int(max_growth_layers):
                checkpoint_supported = False
                unsupported_reason = "diagnostic_growth_layer_budget_reached"
                break
            scores, rows = _candidate_score_rows(
                atoms=pool,
                current_terms=current_terms,
                layout=layout,
                theta_runtime=theta_current,
                psi_ref=psi_ref,
                hmat=hmat,
                base_geometry=geometry,
                eigenvalue_cutoff=eigenvalue_cutoff,
                interval_index=int(interval_index),
                growth_iteration=int(interval_layer_count),
            )
            candidate_evaluations.extend(rows)
            total_geometry_solves += int(len(scores))
            selected = select_tetris_method3_layer(
                scores,
                min_distance_sq_gain=min_gain,
                max_layer_width=max_layer_width,
            )
            if not selected:
                checkpoint_supported = False
                unsupported_reason = "no_positive_gain_disjoint_tetris_layer"
                break
            selected_atoms = tuple(score.atom for score in selected)
            new_terms, new_layout, new_theta, new_executor = _trial_support(
                current_terms=current_terms,
                layout=layout,
                theta_runtime=theta_current,
                psi_ref=psi_ref,
                atoms=selected_atoms,
                label_prefix=f"avqds_tetris_layer_{int(global_layer_index)}",
            )
            joint_geometry = solve_avqds_projective_geometry(
                executor=new_executor,
                psi_ref=psi_ref,
                theta_runtime=new_theta,
                hmat=hmat,
                eigenvalue_cutoff=eigenvalue_cutoff,
            )
            total_geometry_solves += 1
            joint_gain = float(geometry.distance_sq - joint_geometry.distance_sq)
            if not np.isfinite(joint_gain) or joint_gain < float(min_gain):
                checkpoint_supported = False
                unsupported_reason = "joint_tetris_layer_failed_minimum_gain"
                break
            event = {
                "event_kind": "append",
                "append_mechanism": "avqds_tetris_method3_layer",
                "interval_index": int(interval_index),
                "time": float(left),
                "growth_iteration": int(interval_layer_count),
                "global_layer_index": int(global_layer_index),
                "count": int(len(selected_atoms)),
                "type": "batched Pauli terms" if len(selected_atoms) > 1 else "singleton Pauli term",
                "pauli_terms": [str(atom.pauli_exyz) for atom in selected_atoms],
                "qubit_supports": [list(atom.qubit_support) for atom in selected_atoms],
                "singleton_scores": [float(score.distance_sq_gain) for score in selected],
                "mclachlan_distance_sq_before": float(geometry.distance_sq),
                "mclachlan_distance_sq_after": float(joint_geometry.distance_sq),
                "mclachlan_distance_sq_gain": float(joint_gain),
                "runtime_parameter_count": int(new_theta.size),
                "logical_block_count": int(getattr(new_layout, "logical_parameter_count")),
            }
            layer_events.append(event)
            current_terms = tuple(new_terms)
            layout = new_layout
            theta_current = np.asarray(new_theta, dtype=float)
            executor = new_executor
            geometry = joint_geometry
            interval_layer_count += 1
            interval_generator_count += int(len(selected_atoms))
            global_layer_index += 1

        if not checkpoint_supported:
            unsupported_checkpoints.append(
                {
                    "interval_index": int(interval_index),
                    "time": float(left),
                    "reason": str(unsupported_reason),
                    "mclachlan_distance_sq": float(geometry.distance_sq),
                    "threshold": float(threshold),
                }
            )

        theta_delta = float(dt) * np.asarray(geometry.theta_dot, dtype=float)
        theta_current = np.asarray(theta_current + theta_delta, dtype=float)
        current_state = _prepare_scaffold_state(executor, psi_ref, theta_current)
        state_norms.append(float(np.linalg.norm(current_state)))
        interval_layouts.append(layout)
        step = {
            "interval_index": int(interval_index),
            "time_start": float(left),
            "time_stop": float(right),
            "dt": float(dt),
            "growth_method": AVQDS_TETRIS_METHOD,
            "checkpoint_supported": bool(checkpoint_supported),
            "unsupported_reason": unsupported_reason,
            "tetris_layers_added": int(interval_layer_count),
            "tetris_generators_added": int(interval_generator_count),
            "append_accepted": bool(interval_generator_count > 0),
            "theta_update_l2": float(np.linalg.norm(theta_delta)),
            "state_norm_after": float(np.linalg.norm(current_state)),
            **geometry.to_step_dict(eigenvalue_cutoff=eigenvalue_cutoff),
        }
        steps.append(step)
        trajectory.append(
            _state_diagnostic_row(
                checkpoint_index=int(interval_index) + 1,
                time_value=float(right),
                method="generic_avqds_tetris",
                method_kind="avqds_tetris",
                state=current_state,
                exact_state=exact_states[int(interval_index) + 1],
                hmat=flow.hmat_at_time(float(right)),
                **observable_context,
                extra={
                    "runtime_parameter_count": int(theta_current.size),
                    "logical_block_count": int(getattr(layout, "logical_parameter_count")),
                    "mclachlan_distance_sq": float(geometry.distance_sq),
                    "rhs_residual_ratio": float(step["rhs_residual_ratio"]),
                    "tetris_layers_added": int(interval_layer_count),
                    "tetris_generators_added": int(interval_generator_count),
                    "append_accepted": bool(interval_generator_count > 0),
                    "checkpoint_supported": bool(checkpoint_supported),
                },
            )
        )

    summary = _trajectory_summary(trajectory)
    distance_values = [step.get("mclachlan_distance_sq") for step in steps]
    n_h = int(len(enumerate_tetris_pool_atoms(
        flow=flow,
        runtime_input=runtime_input,
        pool_source="hamiltonian_pauli",
    )))
    base_parameter_count = int(getattr(initial_layout, "runtime_parameter_count"))
    metric_element_evaluations = int(
        sum(
            int(step.get("parameter_count", 0)) * (int(step.get("parameter_count", 0)) + 1) // 2
            for step in steps
        )
    )
    force_pauli_component_evaluations = int(
        sum(int(step.get("parameter_count", 0)) * n_h for step in steps)
    )
    candidate_scan_new_column_evaluations = int(
        sum(max(0, int(row.get("runtime_parameter_count", 0)) - 1) + n_h for row in candidate_evaluations)
    )
    resources = _scaffold_resources_for_layouts(
        state_layout=layout,
        interval_layouts=interval_layouts,
        state_scope="generic_avqds_tetris_state_scaffold",
        horizon_scope="generic_avqds_tetris_scaffold_epoch_sum",
        extra={
            "measurement_model": "ideal_expectation_primitives_no_finite_shots",
            "shots_total": None,
            "hamiltonian_pauli_count": int(n_h),
            "tetris_pool_size": int(len(pool)),
            "normalized_pool_profile": str(pool_contract.profile),
            "normalized_pool_ordered_unique_pauli_sha256": str(
                pool_contract.ordered_unique_pauli_sha256
            ),
            "tetris_layer_count": int(len(layer_events)),
            "tetris_generators_added_total": int(sum(int(event["count"]) for event in layer_events)),
            "candidate_evaluations_total": int(len(candidate_evaluations)),
            "geometry_solves_total": int(total_geometry_solves),
            "metric_element_evaluations_total": int(metric_element_evaluations),
            "force_pauli_component_evaluations_total": int(force_pauli_component_evaluations),
            "candidate_scan_new_column_component_evaluations_total": int(candidate_scan_new_column_evaluations),
            "initial_runtime_parameter_count": int(base_parameter_count),
        },
    )
    metrics = {
        "method_kind": "avqds_tetris",
        "growth_method": AVQDS_TETRIS_METHOD,
        "decision_mode": "continuous_rhs_projective_mclachlan_tetris_method3",
        "decision_data_flow": "ideal_mclachlan_expectation_primitive_estimator",
        "pool_source": str(pool_source),
        "candidate_pool_complete": True,
        "candidate_pool_completeness": (
            "complete_hamiltonian_and_drive_pauli_union"
            if str(pool_source) == "hamiltonian_pauli"
            else "complete_runtime_candidate_pool_pauli_expansion"
        ),
        "candidate_pool_size": int(len(pool)),
        "normalized_candidate_pool": pool_contract.to_json_dict(
            include_atoms=False
        ),
        "tetris_layer_count": int(len(layer_events)),
        "tetris_generators_added_total": int(sum(int(event["count"]) for event in layer_events)),
        "candidate_evaluations_total": int(len(candidate_evaluations)),
        "final_runtime_parameter_count": int(theta_current.size),
        "final_logical_block_count": int(getattr(layout, "logical_parameter_count")),
        "mclachlan_distance_sq_final": _float_or_none(distance_values[-1]) if distance_values else None,
        "mclachlan_distance_sq_max": _max_or_none(distance_values),
        "unsupported_checkpoint_count": int(len(unsupported_checkpoints)),
        "uses_statevector_as_ideal_observable_estimator": True,
        "finite_shots_used": False,
        "exact_fields_reporting_only": True,
        "append_scoring_uses_exact_reference": False,
        "singleton_limit_contract": "max_layer_width_1_equals_method1_best_singleton_selection",
        "diagnostic_redundancy_stress": dict(redundancy_stress),
    }
    setting_keys = (
        "avqds_tetris_mclachlan_distance_sq_threshold",
        "avqds_tetris_eigenvalue_cutoff",
        "avqds_tetris_min_distance_sq_gain",
        "avqds_tetris_pool_source",
        "avqds_tetris_candidate_limit",
        "avqds_tetris_max_layer_width",
        "avqds_tetris_max_growth_layers_per_checkpoint",
    )
    tuning = build_dynamics_tuning_provenance(
        case=case,
        algorithm_id=AVQDS_TETRIS_ALGORITHM_ID,
        settings_kind="comparator",
        settings_payload={
            "growth_method": AVQDS_TETRIS_METHOD,
            "mclachlan_distance_sq_threshold": float(threshold),
            "eigenvalue_cutoff": float(eigenvalue_cutoff),
            "min_distance_sq_gain": float(min_gain),
            "pool_source": str(pool_source),
            "candidate_limit": candidate_limit,
            "max_layer_width": max_layer_width,
            "max_growth_layers_per_checkpoint": max_growth_layers,
        },
        settings_source=common.metadata_override_settings_source(case, setting_keys),
        locked=False,
    )
    parameter_manifest = _generic_parameter_manifest(
        case=case,
        runtime_input=runtime_input,
        algorithm_id=AVQDS_TETRIS_ALGORITHM_ID,
        times=times,
        terms=hamiltonian_terms,
        flow=flow,
    )
    parameter_manifest["tuning_provenance"] = dict(tuning)
    parameter_manifest["normalized_candidate_pool"] = (
        pool_contract.to_json_dict(include_atoms=False)
    )
    parameter_manifest["diagnostic_redundancy_stress"] = dict(
        redundancy_stress
    )
    correctness = _build_correctness_sidecar(
        case=case,
        trajectory=trajectory,
        steps=steps,
        layer_events=layer_events,
        candidate_evaluations=candidate_evaluations,
        state_norms=state_norms,
    )
    return json_safe(
        {
            "schema_version": "generic_avqds_tetris_benchmark_v1",
            "case": case.to_dict(),
            "drive_aligned_ansatz": drive_aligned_ansatz.to_json_dict(),
            "diagnostic_redundancy_stress": dict(redundancy_stress),
            "row_contract": {
                "qpu_faithful": True,
                "exact_assisted": False,
                "diagnostic": True,
            },
            "parameter_manifest": parameter_manifest,
            "tuning_provenance": tuning,
            "command": list(command),
            "trajectory": trajectory,
            "avqds_tetris_steps": steps,
            "tetris_layer_events": layer_events,
            "append_events": layer_events,
            "candidate_evaluations": candidate_evaluations,
            "unsupported_checkpoints": unsupported_checkpoints,
            "avqds_tetris_correctness": correctness,
            "summary": summary,
            "metrics": metrics,
            "resources": resources,
            "compile_audit": _compile_audit_from_resources(resources),
            "provenance": {
                "route_module": "pipelines.time_dynamics.benchmarks.avqds_tetris",
                "runner_module": "pipelines.time_dynamics.benchmarks.avqds_tetris",
                "benchmark_only": True,
                "literature_method": "AVQDS(T), Method 3 TETRIS",
                "literature_reference": "Zhang et al., Phys. Rev. B 111, 094310 (2025)",
                "comparator_kernel": "continuous_rhs_projective_avqds_tetris_method3",
                "exact_data_policy": "diagnostic_exact_reference_reporting_only_not_rhs_or_tetris_decision",
                "controller_paths_called": False,
                "controller_decisions_modified": False,
                "exact_reference_controller_inputs": False,
                "append_scoring_uses_exact_reference": False,
                "uses_statevector_as_ideal_observable_estimator": True,
                "finite_shots_used": False,
            },
        }
    )


def run_benchmark_row(
    *,
    case: DynamicsBenchmarkCase,
    output_dir: Path,
) -> DynamicsBenchmarkRow:
    return common.run_native_generic_comparator_row(
        case=case,
        algorithm_id=AVQDS_TETRIS_ALGORITHM_ID,
        output_dir=Path(output_dir),
        payload_builder=_build_avqds_tetris_payload,
    )


__all__ = [
    "AVQDS_TETRIS_ALGORITHM_ID",
    "AVQDS_TETRIS_DEFAULT_DISTANCE_SQ_THRESHOLD",
    "AVQDS_TETRIS_DEFAULT_EIGENVALUE_CUTOFF",
    "TetrisCandidateScore",
    "TetrisPoolAtom",
    "build_tetris_pool_contract",
    "enumerate_tetris_pool_atoms",
    "run_benchmark_row",
    "select_avqds_method1_candidate",
    "select_tetris_method3_layer",
    "solve_avqds_projective_geometry",
    "initial_avqds_tetris_variational_bundle",
]
