"""Fixed-policy seeded real-time Krylov comparator for Paper III.

The online route uses only the declared Hamiltonian, prepared ground state,
target sector, and a deterministic random seed.  Exact excited-state roots are
deliberately absent from :func:`run_seeded_realtime_krylov`; they enter only
through the separate post-run :func:`audit_krylov_energies` function.

This module supersedes the *policy* used by the older diagnostic
``trace_krylov`` without changing that function or its historical artifacts.
In particular, the time step is fixed before a case is run rather than chosen
from an exact-reference-scored grid.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.sparse.linalg import expm_multiply

from pipelines.exact_bench.paper_iii_qse_measurement_cost import _polynomial_words
from pipelines.qse_spectra.compiled_costs import (
    ORACLE_KIND_BACKEND_TRANSPILE,
    ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
    annotate_basis_with_compiled_costs,
    resolve_cost_weights_preset,
)
from pipelines.qse_spectra.core import polynomial_basis_element
from pipelines.qse_spectra.davidson_initialization import target_sector_indices
from src.quantum.compiled_polynomial import (
    apply_compiled_polynomial,
    compile_polynomial_action,
)


SEEDED_REALTIME_KRYLOV_SCHEMA_VERSION = "paper_iii_seeded_realtime_krylov_v1"
SEEDED_REALTIME_KRYLOV_POLICY_ID = "seeded_realtime_krylov_fixed_dt_0p5_v1"
DEFAULT_TIME_STEP = 0.5
DEFAULT_RNG_SEED = 20260826
DEFAULT_OVERLAP_RELATIVE_CUTOFF = 1.0e-12


@dataclass(frozen=True)
class SeededRealtimeKrylovConfig:
    """Predeclared online policy for one Krylov trace."""

    k_max: int
    target_root_count: int = 6
    time_step: float = DEFAULT_TIME_STEP
    rng_seed: int = DEFAULT_RNG_SEED
    overlap_relative_cutoff: float = DEFAULT_OVERLAP_RELATIVE_CUTOFF
    include_qiskit: bool = True

    def __post_init__(self) -> None:
        if int(self.k_max) < 2:
            raise ValueError("k_max must be at least 2 for a Krylov pencil.")
        if int(self.target_root_count) < 1:
            raise ValueError("target_root_count must be positive.")
        if not np.isfinite(float(self.time_step)) or float(self.time_step) <= 0.0:
            raise ValueError("time_step must be finite and positive.")
        cutoff = float(self.overlap_relative_cutoff)
        if not np.isfinite(cutoff) or cutoff <= 0.0:
            raise ValueError("overlap_relative_cutoff must be finite and positive.")


@dataclass(frozen=True)
class SeededRealtimeKrylovRun:
    """Online trace plus terminal vectors retained for post-run audits."""

    policy_receipt: Mapping[str, Any]
    source_receipt: Mapping[str, Any]
    resource_receipt: Mapping[str, Any]
    sector_indices: tuple[int, ...]
    trace: tuple[Mapping[str, Any], ...]
    terminal_ritz_vectors_sector: np.ndarray


def _restricted_hamiltonian(
    hamiltonian: Any, *, num_qubits: int, sector_indices: Sequence[int]
) -> np.ndarray:
    """Build ``H`` in a declared computational sector without diagonalizing it."""

    indices = tuple(int(index) for index in sector_indices)
    full_dimension = 1 << int(num_qubits)
    compiled = compile_polynomial_action(hamiltonian)
    restricted = np.zeros((len(indices), len(indices)), dtype=complex)
    for column, basis_index in enumerate(indices):
        basis = np.zeros(full_dimension, dtype=complex)
        basis[int(basis_index)] = 1.0
        restricted[:, column] = np.asarray(
            apply_compiled_polynomial(basis, compiled), dtype=complex
        )[list(indices)]
    return 0.5 * (restricted + restricted.conj().T)


def _deterministic_sector_kick(
    ground_sector: np.ndarray,
    *,
    sector_indices: Sequence[int],
    rng_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the historical seeded sector kick, orthogonal to the ground."""

    ground = np.asarray(ground_sector, dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(ground))
    if norm <= 1.0e-14:
        raise ValueError("prepared ground has zero norm in the declared target sector.")
    ground = ground / norm
    support = np.abs(ground) > 0.0
    support_positions = np.flatnonzero(support)
    if support_positions.size < 2:
        raise ValueError(
            "seeded sector kick needs at least two supported computational states."
        )
    rng = np.random.default_rng(int(rng_seed))
    source = np.zeros_like(ground)
    count = int(support_positions.size)
    source[support] = rng.normal(size=count) + 1.0j * rng.normal(size=count)
    source -= complex(np.vdot(ground, source)) * ground
    source_norm = float(np.linalg.norm(source))
    if source_norm <= 1.0e-14:
        raise ValueError("deterministic sector kick collapsed after ground projection.")
    source /= source_norm
    computational_indices = [
        int(sector_indices[int(position)]) for position in support_positions
    ]
    digest = hashlib.sha256(
        json.dumps(computational_indices, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return source, {
        "rng_seed": int(rng_seed),
        "construction": "random_complex_on_ground_computational_support",
        "orthogonalization": "subtract_prepared_ground_component",
        "prepared_ground_overlap_abs": float(abs(np.vdot(ground, source))),
        "computational_support_size": count,
        "computational_support_digest": digest,
        "source_preparation_cost_included": False,
    }


def _step_resource_receipt(problem: Any, *, include_qiskit: bool) -> dict[str, Any]:
    step_element = polynomial_basis_element(
        problem.hamiltonian, name="first_order_hamiltonian_trotter_step"
    )
    weights = resolve_cost_weights_preset("two_qubit_only_v1")
    graph_row = annotate_basis_with_compiled_costs(
        (step_element,),
        num_qubits=int(problem.num_qubits),
        oracle_kind=ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
        cost_weights=weights,
    )[0]
    graph = {
        "n2q": float(graph_row.estimate.c_hat_2q),
        "d2q": float(graph_row.estimate.c_hat_d),
        "n1q": float(graph_row.estimate.c_hat_1q),
        "source": str(graph_row.hardware_cost_source),
        "source_mode": str(graph_row.source_mode),
    }
    qiskit: dict[str, Any]
    if include_qiskit:
        qiskit_row = annotate_basis_with_compiled_costs(
            (step_element,),
            num_qubits=int(problem.num_qubits),
            oracle_kind=ORACLE_KIND_BACKEND_TRANSPILE,
            cost_weights=weights,
        )[0]
        qiskit = {
            "status": "COMPILED",
            "oracle_kind": ORACLE_KIND_BACKEND_TRANSPILE,
            "requested_backend_name": "FakeMarrakesh",
            "seed_transpiler": 7,
            "optimization_level": 1,
            "n2q": float(qiskit_row.estimate.c_hat_2q),
            "d2q": float(qiskit_row.estimate.c_hat_d),
            "n1q": float(qiskit_row.estimate.c_hat_1q),
            "source": str(qiskit_row.hardware_cost_source),
            "source_mode": str(qiskit_row.source_mode),
        }
    else:
        qiskit = {"status": "NOT_REQUESTED"}
    return {
        "K_coordinate": "aggregate_graph_span_n2q",
        "basis_state_costing": "sum_of_preparation_steps_for_all_krylov_states",
        "source_preparation_cost_included": False,
        "per_hamiltonian_step": {"graph_span": graph, "qiskit_compiled": qiskit},
    }


def _aggregate_resources(
    resource_receipt: Mapping[str, Any], *, dimension: int
) -> dict[str, Any]:
    steps = int(dimension) * (int(dimension) - 1) // 2
    per_step = dict(resource_receipt["per_hamiltonian_step"])
    graph_step = dict(per_step["graph_span"])
    graph = {
        key: float(graph_step[key]) * steps for key in ("n2q", "d2q", "n1q")
    }
    graph["dc"] = float(graph["d2q"] + graph["n1q"])
    qiskit_step = dict(per_step["qiskit_compiled"])
    if qiskit_step.get("status") == "COMPILED":
        qiskit = {
            **{
                key: qiskit_step[key]
                for key in (
                    "status",
                    "oracle_kind",
                    "requested_backend_name",
                    "seed_transpiler",
                    "optimization_level",
                    "source",
                    "source_mode",
                )
            },
            "n2q": float(qiskit_step["n2q"]) * steps,
            "d2q": float(qiskit_step["d2q"]) * steps,
            "n1q": float(qiskit_step["n1q"]) * steps,
        }
    else:
        qiskit = {"status": str(qiskit_step.get("status", "NOT_REQUESTED"))}
    return {
        "hamiltonian_steps_charged": steps,
        "K": float(graph["n2q"]),
        "graph_span": graph,
        "qiskit_compiled": qiskit,
    }


def run_seeded_realtime_krylov(
    problem: Any, *, config: SeededRealtimeKrylovConfig
) -> SeededRealtimeKrylovRun:
    """Run the fixed online Krylov route without reading exact excited roots."""

    sector_indices = target_sector_indices(problem)
    ground_full = np.asarray(problem.ground, dtype=complex).reshape(-1)
    expected_dimension = 1 << int(problem.num_qubits)
    if ground_full.size != expected_dimension:
        raise ValueError(
            f"prepared ground length {ground_full.size} != register dimension "
            f"{expected_dimension}."
        )
    ground_sector = ground_full[list(sector_indices)]
    source, source_receipt = _deterministic_sector_kick(
        ground_sector,
        sector_indices=sector_indices,
        rng_seed=int(config.rng_seed),
    )
    restricted_h = _restricted_hamiltonian(
        problem.hamiltonian,
        num_qubits=int(problem.num_qubits),
        sector_indices=sector_indices,
    )
    states = np.asarray(
        expm_multiply(
            -1.0j * restricted_h,
            source,
            start=0.0,
            stop=float(config.time_step) * (int(config.k_max) - 1),
            num=int(config.k_max),
            endpoint=True,
        ),
        dtype=complex,
    )
    h_states = np.asarray(states @ restricted_h.T, dtype=complex)
    resource_receipt = _step_resource_receipt(
        problem, include_qiskit=bool(config.include_qiskit)
    )
    ham_terms = len(_polynomial_words(problem.hamiltonian))
    trace: list[dict[str, Any]] = []
    terminal_vectors = np.zeros((len(sector_indices), 0), dtype=complex)
    for dimension in range(2, int(config.k_max) + 1):
        block = states[:dimension].T
        h_block = h_states[:dimension].T
        overlap = 0.5 * (block.conj().T @ block + (block.conj().T @ block).conj().T)
        projected_h = 0.5 * (
            block.conj().T @ h_block + (block.conj().T @ h_block).conj().T
        )
        overlap_values, overlap_vectors = np.linalg.eigh(overlap)
        maximum = float(max(float(np.max(overlap_values)), 0.0))
        keep = overlap_values > float(config.overlap_relative_cutoff) * maximum
        retained_rank = int(np.count_nonzero(keep))
        if retained_rank < 1:
            continue
        whitening = overlap_vectors[:, keep] / np.sqrt(overlap_values[keep])
        reduced = whitening.conj().T @ projected_h @ whitening
        reduced = 0.5 * (reduced + reduced.conj().T)
        roots, reduced_vectors = np.linalg.eigh(reduced)
        coefficients = whitening @ reduced_vectors
        terminal_vectors = np.asarray(block @ coefficients, dtype=complex)
        resources = _aggregate_resources(resource_receipt, dimension=dimension)
        pairs = dimension * (dimension + 1) // 2
        trace.append(
            {
                "k": int(dimension),
                "support_size": int(dimension),
                "retained_rank": retained_rank,
                "time_step": float(config.time_step),
                "pencil_root_energies": [float(value) for value in roots],
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
        raise ValueError("Krylov route produced no supported pencil.")
    policy_receipt = {
        "schema_version": SEEDED_REALTIME_KRYLOV_SCHEMA_VERSION,
        "policy_id": SEEDED_REALTIME_KRYLOV_POLICY_ID,
        "time_step": float(config.time_step),
        "time_step_selection": "fixed_before_case_execution",
        "time_step_reference_tuning": False,
        "rng_seed": int(config.rng_seed),
        "k_max": int(config.k_max),
        "target_root_count": int(config.target_root_count),
        "overlap_relative_cutoff": float(config.overlap_relative_cutoff),
        "exact_excited_roots_used_online": False,
        "root_matching_stage": "post_run_only",
    }
    return SeededRealtimeKrylovRun(
        policy_receipt=policy_receipt,
        source_receipt=source_receipt,
        resource_receipt=resource_receipt,
        sector_indices=tuple(int(index) for index in sector_indices),
        trace=tuple(trace),
        terminal_ritz_vectors_sector=terminal_vectors,
    )


def audit_krylov_energies(
    run: SeededRealtimeKrylovRun, exact_references: Sequence[float]
) -> tuple[dict[str, Any], ...]:
    """Attach best-matching target-root errors after the online run completes."""

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
        indices = [int(np.argmin(np.abs(roots - reference))) for reference in references]
        matched = np.asarray([roots[index] for index in indices], dtype=float)
        errors = np.abs(matched - references)
        row.update(
            {
                "post_run_exact_audit": True,
                "root_match_convention": "best_matching_pencil_root_per_target",
                "matched_pencil_root_indices": indices,
                "matched_root_energies": [float(value) for value in matched],
                "per_root_abs_error": [float(value) for value in errors],
                "max_root_abs_error": float(np.max(errors)),
                "rth_root_energy": float(matched[-1]),
            }
        )
        audited.append(row)
    return tuple(audited)


def matched_terminal_ritz_vectors(
    run: SeededRealtimeKrylovRun, audited_trace: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    """Return terminal Ritz vectors in target-root matching order."""

    if not audited_trace:
        raise ValueError("audited_trace is empty.")
    indices = [int(index) for index in audited_trace[-1]["matched_pencil_root_indices"]]
    vectors = np.asarray(run.terminal_ritz_vectors_sector, dtype=complex)
    return np.asarray(vectors[:, indices], dtype=complex)


__all__ = [
    "DEFAULT_OVERLAP_RELATIVE_CUTOFF",
    "DEFAULT_RNG_SEED",
    "DEFAULT_TIME_STEP",
    "SEEDED_REALTIME_KRYLOV_POLICY_ID",
    "SEEDED_REALTIME_KRYLOV_SCHEMA_VERSION",
    "SeededRealtimeKrylovConfig",
    "SeededRealtimeKrylovRun",
    "audit_krylov_energies",
    "matched_terminal_ritz_vectors",
    "run_seeded_realtime_krylov",
]
