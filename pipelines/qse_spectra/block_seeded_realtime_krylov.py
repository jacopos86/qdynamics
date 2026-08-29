"""Deterministic block-seeded real-time Krylov comparator for Paper III.

The route uses the computational-probe branch of the direct-Davidson
initializer: lowest-diagonal computational states in the declared target
sector.  The first ``R`` independent excited directions form a fixed block.
The prepared ground is retained as a free reference direction, while every
admitted block source and every Hamiltonian propagation step is charged in
each independently prepared Krylov state.

Exact excited roots are not available to :func:`run_block_seeded_realtime_krylov`.
They enter only through :func:`audit_block_krylov_energies` after the fixed-cap
online trace has terminated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import expm_multiply

from pipelines.exact_bench.paper_iii_qse_measurement_cost import _polynomial_words
from pipelines.qse_spectra.adaptive_qse_benchmark import (
    _seed_action,
    orthogonalize_adaptive_direction,
)
from pipelines.qse_spectra.compiled_costs import (
    ORACLE_KIND_BACKEND_TRANSPILE,
    annotate_basis_with_compiled_costs,
    resolve_cost_weights_preset,
)
from pipelines.qse_spectra.core import QSEBasisElement, QSEPruningConfig, normalize_statevector
from pipelines.qse_spectra.davidson_initialization import (
    ORIGIN_COMPUTATIONAL_PROBE,
    ORIGIN_FIXED_RECORD,
    DavidsonSeedDirection,
    build_direct_davidson_start,
    target_sector_indices,
)
from pipelines.qse_spectra.seeded_realtime_krylov import (
    DEFAULT_OVERLAP_RELATIVE_CUTOFF,
    DEFAULT_TIME_STEP,
    _restricted_hamiltonian,
    _step_resource_receipt,
)


BLOCK_SEEDED_REALTIME_KRYLOV_SCHEMA_VERSION = (
    "paper_iii_block_seeded_realtime_krylov_v1"
)
BLOCK_SEEDED_REALTIME_KRYLOV_POLICY_ID = (
    "block_seeded_realtime_krylov_lowest_diagonal_b6_dt0p5_v1"
)
DEFAULT_BLOCK_SIZE = 6
DEFAULT_INDEPENDENCE_FLOOR = 1.0e-12


@dataclass(frozen=True)
class BlockSeededRealtimeKrylovConfig:
    """Predeclared online policy for one block-Krylov trace."""

    k_max: int
    target_root_count: int = 6
    block_size: int = DEFAULT_BLOCK_SIZE
    time_step: float = DEFAULT_TIME_STEP
    overlap_relative_cutoff: float = DEFAULT_OVERLAP_RELATIVE_CUTOFF
    independence_floor: float = DEFAULT_INDEPENDENCE_FLOOR
    include_qiskit: bool = True

    def __post_init__(self) -> None:
        if int(self.target_root_count) < 1:
            raise ValueError("target_root_count must be positive.")
        if int(self.block_size) < 1:
            raise ValueError("block_size must be positive.")
        if int(self.block_size) != int(self.target_root_count):
            raise ValueError(
                "the locked block policy requires block_size == target_root_count."
            )
        if int(self.k_max) < int(self.block_size) + 1:
            raise ValueError(
                "k_max must admit the free ground plus one complete block shell."
            )
        if not np.isfinite(float(self.time_step)) or float(self.time_step) <= 0.0:
            raise ValueError("time_step must be finite and positive.")
        cutoff = float(self.overlap_relative_cutoff)
        if not np.isfinite(cutoff) or cutoff <= 0.0:
            raise ValueError("overlap_relative_cutoff must be finite and positive.")
        floor = float(self.independence_floor)
        if not np.isfinite(floor) or floor < 0.0:
            raise ValueError("independence_floor must be finite and non-negative.")


@dataclass(frozen=True)
class BlockSeededRealtimeKrylovRun:
    """Online trace plus terminal vectors retained for post-run audits."""

    policy_receipt: Mapping[str, Any]
    source_receipt: Mapping[str, Any]
    resource_receipt: Mapping[str, Any]
    sector_indices: tuple[int, ...]
    trace: tuple[Mapping[str, Any], ...]
    terminal_ritz_vectors_sector: np.ndarray


def _seed_graph_resources(
    direction: DavidsonSeedDirection,
) -> dict[str, float]:
    raw = dict(direction.resources)
    n2q = float(raw.get("n2q", 0.0))
    d2q = float(raw.get("d2q", 0.0))
    dc = float(raw.get("dc", 0.0))
    return {
        "n2q": n2q,
        "d2q": d2q,
        "n1q": max(0.0, dc - d2q),
        "dc": dc,
    }


def _seed_qiskit_resources(
    direction: DavidsonSeedDirection, *, num_qubits: int, include_qiskit: bool
) -> dict[str, Any]:
    if not include_qiskit:
        return {"status": "NOT_REQUESTED"}
    if direction.origin == ORIGIN_COMPUTATIONAL_PROBE:
        graph = _seed_graph_resources(direction)
        return {
            "status": "COMPILED",
            "oracle_kind": ORACLE_KIND_BACKEND_TRANSPILE,
            "requested_backend_name": "FakeMarrakesh",
            "seed_transpiler": 7,
            "optimization_level": 1,
            "n2q": 0.0,
            "d2q": 0.0,
            "n1q": float(graph["n1q"]),
            "source": "computational_basis_bit_flips",
            "source_mode": "analytic_basis_preparation",
        }
    if direction.origin != ORIGIN_FIXED_RECORD or not isinstance(
        direction.payload, QSEBasisElement
    ):
        raise ValueError(f"unsupported block-source origin {direction.origin!r}.")
    row = annotate_basis_with_compiled_costs(
        (direction.payload,),
        num_qubits=int(num_qubits),
        oracle_kind=ORACLE_KIND_BACKEND_TRANSPILE,
        cost_weights=resolve_cost_weights_preset("two_qubit_only_v1"),
    )[0]
    return {
        "status": "COMPILED",
        "oracle_kind": ORACLE_KIND_BACKEND_TRANSPILE,
        "requested_backend_name": "FakeMarrakesh",
        "seed_transpiler": 7,
        "optimization_level": 1,
        "n2q": float(row.estimate.c_hat_2q),
        "d2q": float(row.estimate.c_hat_d),
        "n1q": float(row.estimate.c_hat_1q),
        "source": str(row.hardware_cost_source),
        "source_mode": str(row.source_mode),
    }


def _build_source_block(
    problem: Any,
    *,
    config: BlockSeededRealtimeKrylovConfig,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Admit the first fixed-order independent directions without references."""

    psi, _norm, nq = normalize_statevector(np.asarray(problem.ground, dtype=complex))
    start = build_direct_davidson_start(
        problem,
        target_root_count=int(config.target_root_count),
        fixed_record_indices=(),
    )
    frame: list[np.ndarray] = [np.asarray(psi, dtype=complex)]
    raw_sources: list[np.ndarray] = []
    admitted_directions: list[DavidsonSeedDirection] = []
    candidate_rows: list[dict[str, Any]] = []
    for candidate_index, direction in enumerate(start.seed_directions):
        raw, materialized_name = _seed_action(
            direction.payload,
            psi,
            nq=int(nq),
            config=QSEPruningConfig(),
        )
        raw_norm = float(np.linalg.norm(raw))
        q0 = raw - complex(np.vdot(psi, raw)) * psi
        admitted, novelty = orthogonalize_adaptive_direction(
            q0,
            np.column_stack(tuple(frame)),
            independence_floor=float(config.independence_floor),
        )
        accepted = admitted is not None and raw_norm > 1.0e-14
        candidate_rows.append(
            {
                "candidate_index": int(candidate_index),
                "label": str(direction.label),
                "materialized_name": str(materialized_name),
                "origin": str(direction.origin),
                "raw_norm": raw_norm,
                "q0_projected_norm": float(np.linalg.norm(q0)),
                "novelty_fraction": float(novelty),
                "admitted": bool(accepted),
            }
        )
        if not accepted:
            continue
        raw_sources.append(np.asarray(raw / raw_norm, dtype=complex))
        admitted_directions.append(direction)
        frame.append(np.asarray(admitted, dtype=complex))
        if len(raw_sources) == int(config.block_size):
            break
    if len(raw_sources) != int(config.block_size):
        raise ValueError(
            "BLOCK_START_INSUFFICIENT: deterministic non-oracular candidates "
            f"supplied {len(raw_sources)} independent directions; "
            f"required {int(config.block_size)}."
        )

    graph_rows = [_seed_graph_resources(direction) for direction in admitted_directions]
    qiskit_rows = [
        _seed_qiskit_resources(
            direction,
            num_qubits=int(nq),
            include_qiskit=bool(config.include_qiskit),
        )
        for direction in admitted_directions
    ]
    source_receipt = {
        **dict(start.receipt),
        "policy": "lowest_diagonal_target_sector_computational_probe_block",
        "block_policy": "first_R_independent_directions_in_declared_start_order",
        "block_size": int(config.block_size),
        "independence_floor": float(config.independence_floor),
        "orthogonalization": "q0_then_double_gram_schmidt_against_admitted_frame",
        "candidate_rows": candidate_rows,
        "admitted_seed_labels": [
            str(direction.label) for direction in admitted_directions
        ],
        "admitted_seed_origins": [
            str(direction.origin) for direction in admitted_directions
        ],
        "uses_exact_reference": False,
        "prepared_ground_included_as_free_reference_direction": True,
        "prepared_ground_state_cost_included": False,
        "raw_source_normalization_cost_included": False,
        "q0_projection_is_span_equivalence_not_a_separately_prepared_state": True,
    }
    source_resources = {
        "graph_span_per_source_preparation": graph_rows,
        "qiskit_compiled_per_source_preparation": qiskit_rows,
        "source_preparation_repetition": (
            "each_admitted_source_action_is_charged_once_per_time_shell"
        ),
    }
    return np.column_stack(tuple(raw_sources)), source_receipt, source_resources


def _sum_resource_rows(
    rows: Sequence[Mapping[str, Any]], *, multiplier: int
) -> dict[str, float]:
    return {
        key: float(
            sum(float(dict(row).get(key, 0.0)) for row in rows) * int(multiplier)
        )
        for key in ("n2q", "d2q", "n1q")
    }


def _aggregate_resources(
    resource_receipt: Mapping[str, Any],
    *,
    block_size: int,
    shell_count: int,
) -> dict[str, Any]:
    """Charge every independently prepared block state through this shell."""

    shells = int(shell_count)
    if shells < 1:
        raise ValueError("shell_count must be positive.")
    maximum_power = shells - 1
    h_steps = int(block_size) * maximum_power * (maximum_power + 1) // 2
    source_resources = dict(resource_receipt["source_preparations"])
    graph_sources = _sum_resource_rows(
        source_resources["graph_span_per_source_preparation"], multiplier=shells
    )
    step = dict(resource_receipt["per_hamiltonian_step"])
    graph_step = dict(step["graph_span"])
    graph = {
        key: float(graph_sources[key]) + float(graph_step[key]) * h_steps
        for key in ("n2q", "d2q", "n1q")
    }
    graph["dc"] = float(graph["d2q"] + graph["n1q"])

    qiskit_step = dict(step["qiskit_compiled"])
    qiskit_source_rows = tuple(
        dict(row)
        for row in source_resources["qiskit_compiled_per_source_preparation"]
    )
    if qiskit_step.get("status") == "COMPILED" and all(
        row.get("status") == "COMPILED" for row in qiskit_source_rows
    ):
        qiskit_sources = _sum_resource_rows(qiskit_source_rows, multiplier=shells)
        qiskit = {
            "status": "COMPILED",
            "oracle_kind": ORACLE_KIND_BACKEND_TRANSPILE,
            "requested_backend_name": "FakeMarrakesh",
            "seed_transpiler": 7,
            "optimization_level": 1,
            "n2q": float(qiskit_sources["n2q"])
            + float(qiskit_step["n2q"]) * h_steps,
            "d2q": float(qiskit_sources["d2q"])
            + float(qiskit_step["d2q"]) * h_steps,
            "n1q": float(qiskit_sources["n1q"])
            + float(qiskit_step["n1q"]) * h_steps,
            "aggregation": "sum_of_independently_prepared_basis_state_annotations",
        }
    else:
        qiskit = {"status": "NOT_REQUESTED"}
    return {
        "source_preparations_charged": int(block_size) * shells,
        "hamiltonian_steps_charged": h_steps,
        "K": float(graph["n2q"]),
        "graph_span": graph,
        "qiskit_compiled": qiskit,
    }


def run_block_seeded_realtime_krylov(
    problem: Any, *, config: BlockSeededRealtimeKrylovConfig
) -> BlockSeededRealtimeKrylovRun:
    """Run the locked block route without reading exact excited roots."""

    sector_indices = target_sector_indices(problem)
    ground_full, _ground_norm, _nq = normalize_statevector(
        np.asarray(problem.ground, dtype=complex)
    )
    sources_full, source_receipt, source_resources = _build_source_block(
        problem, config=config
    )
    sector = list(sector_indices)
    ground_sector = np.asarray(ground_full[sector], dtype=complex)
    ground_sector /= np.linalg.norm(ground_sector)
    sources_sector = np.asarray(sources_full[sector, :], dtype=complex)
    restricted_h = _restricted_hamiltonian(
        problem.hamiltonian,
        num_qubits=int(problem.num_qubits),
        sector_indices=sector_indices,
    )
    shell_count_max = (int(config.k_max) - 1) // int(config.block_size)
    if shell_count_max == 1:
        evolved = np.asarray(sources_sector[np.newaxis, :, :], dtype=complex)
    else:
        evolved = np.asarray(
            expm_multiply(
                -1.0j * restricted_h,
                sources_sector,
                start=0.0,
                stop=float(config.time_step) * (shell_count_max - 1),
                num=shell_count_max,
                endpoint=True,
            ),
            dtype=complex,
        )
    resource_receipt = {
        **_step_resource_receipt(
            problem, include_qiskit=bool(config.include_qiskit)
        ),
        "source_preparations": source_resources,
        "basis_state_costing": (
            "sum_seed_action_plus_hamiltonian_steps_for_every_independently_"
            "prepared_block_krylov_state"
        ),
        "source_preparation_cost_included": True,
        "prepared_ground_state_cost_included": False,
    }
    ham_terms = len(_polynomial_words(problem.hamiltonian))
    trace: list[dict[str, Any]] = []
    terminal_vectors = np.zeros((len(sector_indices), 0), dtype=complex)
    for shells in range(1, shell_count_max + 1):
        orbit = np.transpose(evolved[:shells], (1, 0, 2)).reshape(
            len(sector_indices), shells * int(config.block_size)
        )
        block = np.column_stack((ground_sector, orbit))
        h_block = np.asarray(restricted_h @ block, dtype=complex)
        overlap = block.conj().T @ block
        projected_h = block.conj().T @ h_block
        overlap = 0.5 * (overlap + overlap.conj().T)
        projected_h = 0.5 * (projected_h + projected_h.conj().T)
        overlap_values, overlap_vectors = np.linalg.eigh(overlap)
        maximum = float(max(float(np.max(overlap_values)), 0.0))
        keep = overlap_values > float(config.overlap_relative_cutoff) * maximum
        retained_rank = int(np.count_nonzero(keep))
        if retained_rank < 2:
            continue
        whitening = overlap_vectors[:, keep] / np.sqrt(overlap_values[keep])
        reduced = whitening.conj().T @ projected_h @ whitening
        reduced = 0.5 * (reduced + reduced.conj().T)
        all_roots, reduced_vectors = np.linalg.eigh(reduced)
        coefficients = whitening @ reduced_vectors
        all_vectors = np.asarray(block @ coefficients, dtype=complex)
        excitation_roots = np.asarray(all_roots[1:], dtype=float)
        terminal_vectors = np.asarray(all_vectors[:, 1:], dtype=complex)
        raw_dimension = 1 + shells * int(config.block_size)
        resources = _aggregate_resources(
            resource_receipt,
            block_size=int(config.block_size),
            shell_count=shells,
        )
        pairs = raw_dimension * (raw_dimension + 1) // 2
        trace.append(
            {
                "k": int(raw_dimension),
                "support_size": int(raw_dimension),
                "block_size": int(config.block_size),
                "time_shell_count": int(shells),
                "maximum_time_index": int(shells - 1),
                "retained_rank": retained_rank,
                "excited_retained_rank": max(0, retained_rank - 1),
                "time_step": float(config.time_step),
                "reference_root_energy": float(all_roots[0]),
                "pencil_root_energies": [
                    float(value) for value in excitation_roots
                ],
                "K": float(resources["K"]),
                "n2q": float(resources["graph_span"]["n2q"]),
                "d2q": float(resources["graph_span"]["d2q"]),
                "dc": float(resources["graph_span"]["dc"]),
                "resources": resources,
                "estimator": {
                    "pair_count": int(pairs),
                    "queries": int(pairs * (1 + ham_terms)),
                    "hamiltonian_pauli_terms": int(ham_terms),
                    "convention": "synthesized_basis_pencil_no_qwc_reuse",
                },
            }
        )
    if not trace:
        raise ValueError("block-Krylov route produced no supported pencil.")
    policy_receipt = {
        "schema_version": BLOCK_SEEDED_REALTIME_KRYLOV_SCHEMA_VERSION,
        "policy_id": BLOCK_SEEDED_REALTIME_KRYLOV_POLICY_ID,
        "time_step": float(config.time_step),
        "time_step_selection": "fixed_before_case_execution",
        "time_step_reference_tuning": False,
        "block_size": int(config.block_size),
        "block_size_rule": "equals_declared_target_root_count",
        "k_max": int(config.k_max),
        "terminal_raw_dimension": int(trace[-1]["k"]),
        "target_root_count": int(config.target_root_count),
        "overlap_relative_cutoff": float(config.overlap_relative_cutoff),
        "independence_floor": float(config.independence_floor),
        "exact_excited_roots_used_online": False,
        "root_matching_stage": "post_run_only",
    }
    return BlockSeededRealtimeKrylovRun(
        policy_receipt=policy_receipt,
        source_receipt=source_receipt,
        resource_receipt=resource_receipt,
        sector_indices=tuple(int(index) for index in sector_indices),
        trace=tuple(trace),
        terminal_ritz_vectors_sector=terminal_vectors,
    )


def audit_block_krylov_energies(
    run: BlockSeededRealtimeKrylovRun, exact_references: Sequence[float]
) -> tuple[dict[str, Any], ...]:
    """Attach one-to-one target-root errors after the online run completes."""

    references = np.asarray(tuple(float(value) for value in exact_references), dtype=float)
    target_count = int(run.policy_receipt["target_root_count"])
    if references.size < target_count:
        raise ValueError(
            f"energy audit needs {target_count} exact roots; got {references.size}."
        )
    references = references[:target_count]
    audited: list[dict[str, Any]] = []
    for raw in run.trace:
        row = dict(raw)
        roots = np.asarray(row["pencil_root_energies"], dtype=float)
        if roots.size < target_count:
            row.update(
                {
                    "post_run_exact_audit": True,
                    "target_window_status": "INSUFFICIENT_RETAINED_RANK",
                    "root_match_convention": (
                        "minimum_total_abs_energy_hungarian_one_to_one"
                    ),
                    "matched_pencil_root_indices": [],
                    "matched_root_energies": [],
                    "per_root_abs_error": [],
                    "max_root_abs_error": None,
                    "rth_root_energy": None,
                }
            )
            audited.append(row)
            continue
        target_indices, pencil_indices = linear_sum_assignment(
            np.abs(references[:, None] - roots[None, :])
        )
        assignment = {
            int(target): int(pencil)
            for target, pencil in zip(target_indices, pencil_indices)
        }
        indices = [assignment[target] for target in range(target_count)]
        matched = np.asarray([roots[index] for index in indices], dtype=float)
        errors = np.abs(matched - references)
        row.update(
            {
                "post_run_exact_audit": True,
                "target_window_status": "ONE_TO_ONE_MATCHED",
                "root_match_convention": (
                    "minimum_total_abs_energy_hungarian_one_to_one"
                ),
                "matched_pencil_root_indices": indices,
                "matched_root_energies": [float(value) for value in matched],
                "per_root_abs_error": [float(value) for value in errors],
                "max_root_abs_error": float(np.max(errors)),
                "rth_root_energy": float(matched[-1]),
            }
        )
        audited.append(row)
    return tuple(audited)


__all__ = [
    "BLOCK_SEEDED_REALTIME_KRYLOV_POLICY_ID",
    "BLOCK_SEEDED_REALTIME_KRYLOV_SCHEMA_VERSION",
    "BlockSeededRealtimeKrylovConfig",
    "BlockSeededRealtimeKrylovRun",
    "DEFAULT_BLOCK_SIZE",
    "audit_block_krylov_energies",
    "run_block_seeded_realtime_krylov",
]
