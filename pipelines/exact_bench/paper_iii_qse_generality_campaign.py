#!/usr/bin/env python3
"""Acquire fresh cross-family Paper-III static-QSE evidence.

The campaign runs the locked full-pencil Ky-Fan-gain-per-K selector, certified
atomic exchange, an independently initialized direct-Davidson comparator, and
a declared fixed-class QSE comparator on three Peierls--Hubbard regimes and a
non-smoke ten-qubit SIAM (``N_b=4``) case.  Exact excited states are consulted
only after each route terminates, for energy and static-spectrum scoring.

The output is a self-contained JSON evidence artifact.  It includes the full
growth traces, terminal supports and statuses, graph-span costs, Qiskit-
transpiled costs, estimator-query conventions, and deterministic replay locks.
It does not edit the Paper-III manuscript or any report PDF.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench import paper_iii_growth_trace_campaign as growth
from pipelines.exact_bench.paper_iii_qse_peierls_pilot import (
    PEIERLS_REGIMES,
    _FIXED_CLASS as PEIERLS_FIXED_CLASS,
)
from pipelines.qse_spectra.compiled_costs import (
    ORACLE_KIND_BACKEND_TRANSPILE,
    annotate_basis_with_compiled_costs,
    resolve_cost_weights_preset,
)
from pipelines.qse_spectra.core import (
    compute_qse_spectra,
    polynomial_observable,
)
from pipelines.qse_spectra.peierls_static_adapter import (
    PEIERLS_FIXED_CLASS_POLICY_ID,
    PEIERLS_POOL_POLICY_ID,
    load_peierls_problem,
    peierls_sector_eigensystem,
)
from pipelines.qse_spectra.record_selection import select_static_qse_records
from pipelines.qse_spectra.selection_policy import (
    PAPER_III_LOCKED_SELECTOR_POLICY_ID,
    locked_paper_iii_selection_config,
)
from pipelines.qse_spectra.siam_static_adapter import (
    SIAM_POOL_GENERALIZED_SD_COMPLETION_V1,
    load_siam_problem,
)
from src.quantum.compiled_polynomial import (
    apply_compiled_polynomial,
    compile_polynomial_action,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm

SCHEMA_VERSION = "paper_iii_qse_generality_evidence_v1"
CAMPAIGN_ID = "paper_iii_qse_generality_20260829_v1"
DEFAULT_OUTPUT = (
    REPO_ROOT / "output/diagnostics" / CAMPAIGN_ID / "generality_evidence.json"
)
TARGET_ROOTS = 6
ERROR_TARGETS = (1.0e-2, 1.0e-4, 1.0e-6)
SIAM_FIXED_FAMILIES = frozenset({"K", "J", "Qd"})
SIAM_FIXED_CLASS_POLICY_ID = "siam_bare_one_body_plus_local_interaction_v1"
SPECTRAL_ETA = 0.05
SPECTRAL_GRID_POINTS = 801


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _selection(problem: Any):
    return select_static_qse_records(
        problem.basis,
        config=locked_paper_iii_selection_config(
            max_records=len(problem.basis),
            target_root_count=TARGET_ROOTS,
            residual_stop=growth.RESIDUAL_MARKER,
        ),
        hamiltonian=problem.hamiltonian,
        prepared_state=problem.ground,
        basis_vector_policy=growth._Q0_POLICY,
        compiled_costs=problem.costs,
    )


def _compact_selection_receipt(selection: Any) -> dict[str, Any]:
    stop = dict(selection.geometry_stop or {})
    return {
        "selector_policy_id": PAPER_III_LOCKED_SELECTOR_POLICY_ID,
        "selected_count": len(selection.selected_original_indices),
        "selected_original_indices": [
            int(index) for index in selection.selected_original_indices
        ],
        "stop_reason": stop.get("stop_reason"),
        "final_max_target_residual_norm": stop.get(
            "final_max_target_residual_norm"
        ),
        "residual_stop_threshold": stop.get("residual_stop_threshold"),
        "round_count": len(stop.get("round_trace", [])),
    }


def _fixed_support(problem: Any, family: str) -> tuple[list[int], str]:
    if family == "peierls_hubbard":
        allowed = set(PEIERLS_FIXED_CLASS)
        return (
            [
                index
                for index, element in enumerate(problem.basis)
                if str(element.name) in allowed
            ],
            PEIERLS_FIXED_CLASS_POLICY_ID,
        )
    indices = [
        index
        for index, element in enumerate(problem.basis)
        if str((element.metadata or {}).get("family")) in SIAM_FIXED_FAMILIES
    ]
    return indices, SIAM_FIXED_CLASS_POLICY_ID


def _fixed_row(problem: Any, indices: Sequence[int]) -> dict[str, Any]:
    return growth._row(problem, indices, len(indices), exchanged=False)


def _threshold_crossings(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for target in ERROR_TARGETS:
        crossing = next(
            (
                row
                for row in rows
                if row.get("max_root_abs_error") is not None
                and float(row["max_root_abs_error"]) <= float(target)
            ),
            None,
        )
        out[f"{target:.0e}"] = (
            {"status": "NOT_REACHED_WITHIN_CAP"}
            if crossing is None
            else {
                "status": "REACHED",
                "k": int(crossing["k"]),
                "support_size": int(crossing["support_size"]),
                "max_root_abs_error": float(crossing["max_root_abs_error"]),
                "graph_span": {
                    key: float(crossing[key]) for key in ("n2q", "d2q", "dc")
                },
                "estimator": crossing.get("estimator"),
            }
        )
    return out


def _structural_compile_elements(elements: Sequence[Any]) -> tuple[Any, ...]:
    """Real-coefficient Pauli-product surrogates for structural compilation.

    Some QSE records (for example ``b`` and current quadratures) are not
    themselves Hermitian ansatz generators.  The backend oracle compiles a
    product of Pauli rotations, so for this offline resource coordinate we
    preserve every nonzero Pauli word and replace its coefficient by its
    magnitude.  Circuit topology and Pauli support are retained; physical
    amplitudes are deliberately not interpreted as variational angles.
    """

    from pipelines.qse_spectra.core import polynomial_basis_element

    out = []
    for element in elements:
        if str(element.kind) == "pauli_string":
            out.append(element)
            continue
        polynomial = PauliPolynomial("JW")
        for term in element.polynomial.return_polynomial():
            coefficient = abs(complex(term.p_coeff))
            if coefficient <= 1.0e-12:
                continue
            polynomial.add_term(
                PauliTerm(
                    len(str(term.pw2strng())),
                    ps=str(term.pw2strng()),
                    pc=float(coefficient),
                )
            )
        out.append(
            polynomial_basis_element(
                polynomial,
                name=str(element.name),
                metadata=dict(element.metadata or {}),
            )
        )
    return tuple(out)


def _record_qiskit_cost(problem: Any, indices: Sequence[int]) -> dict[str, Any]:
    elements = tuple(problem.basis[int(index)] for index in indices)
    rows = annotate_basis_with_compiled_costs(
        _structural_compile_elements(elements),
        num_qubits=int(problem.num_qubits),
        oracle_kind=ORACLE_KIND_BACKEND_TRANSPILE,
        cost_weights=resolve_cost_weights_preset("two_qubit_only_v1"),
    )
    records = []
    for support_index, row in zip(indices, rows):
        records.append(
            {
                "support_index": int(support_index),
                "name": str(problem.basis[int(support_index)].name),
                "n2q": float(row.estimate.c_hat_2q),
                "d2q": float(row.estimate.c_hat_d),
                "n1q": float(row.estimate.c_hat_1q),
                "source": str(row.hardware_cost_source),
                "source_mode": str(row.source_mode),
            }
        )
    return {
        "status": "COMPILED",
        "oracle_kind": ORACLE_KIND_BACKEND_TRANSPILE,
        "requested_backend_name": "FakeMarrakesh",
        "seed_transpiler": 7,
        "optimization_level": 1,
        "coefficient_policy": "preserve_pauli_words_replace_coefficients_by_magnitude",
        "n2q": float(sum(row["n2q"] for row in records)),
        "d2q": float(sum(row["d2q"] for row in records)),
        "n1q": float(sum(row["n1q"] for row in records)),
        "records": records,
    }


def _direct_qiskit_cost(problem: Any, terminal: Mapping[str, Any]) -> dict[str, Any]:
    from pipelines.qse_spectra.core import polynomial_basis_element

    initialization = dict(terminal["initialization_receipt"])
    fixed_indices = [
        int(label.rsplit("::", 1)[1])
        for label in initialization.get("admitted_seed_labels", [])
        if str(label).startswith("fixed_record::")
    ]
    fixed = _record_qiskit_cost(problem, fixed_indices)
    step_element = polynomial_basis_element(
        problem.hamiltonian, name="davidson_residual_step"
    )
    step = annotate_basis_with_compiled_costs(
        _structural_compile_elements((step_element,)),
        num_qubits=int(problem.num_qubits),
        oracle_kind=ORACLE_KIND_BACKEND_TRANSPILE,
        cost_weights=resolve_cost_weights_preset("two_qubit_only_v1"),
    )[0]
    count = int(terminal["synthesized_direction_count"])
    return {
        "status": "COMPILED",
        "oracle_kind": ORACLE_KIND_BACKEND_TRANSPILE,
        "requested_backend_name": "FakeMarrakesh",
        "seed_transpiler": 7,
        "optimization_level": 1,
        "coefficient_policy": "preserve_pauli_words_replace_coefficients_by_magnitude",
        "costing_convention": "compiled_fixed_seeds_plus_repeated_compiled_hamiltonian_step",
        "fixed_seed_support_indices": fixed_indices,
        "fixed_seed": {key: fixed[key] for key in ("n2q", "d2q", "n1q")},
        "synthesized_direction_count": count,
        "per_synthesized_direction": {
            "n2q": float(step.estimate.c_hat_2q),
            "d2q": float(step.estimate.c_hat_d),
            "n1q": float(step.estimate.c_hat_1q),
        },
        "n2q": float(fixed["n2q"] + count * step.estimate.c_hat_2q),
        "d2q": float(fixed["d2q"] + count * step.estimate.c_hat_d),
        "n1q": float(fixed["n1q"] + count * step.estimate.c_hat_1q),
    }


def _exact_eigensystem(problem: Any, family: str) -> tuple[np.ndarray, np.ndarray]:
    if family == "peierls_hubbard":
        return peierls_sector_eigensystem(problem.key)
    from src.quantum.siam_model import embed_sector_vector
    from src.quantum.siam_reference import sector_hamiltonian

    restricted = sector_hamiltonian(
        int(problem.key.n_bath), u=float(problem.key.u), v=float(problem.key.v)
    )
    energies, vectors = np.linalg.eigh(0.5 * (restricted + restricted.conj().T))
    embedded = np.column_stack(
        tuple(
            embed_sector_vector(vectors[:, root], int(problem.key.n_bath))
            for root in range(vectors.shape[1])
        )
    )
    return np.asarray(energies, dtype=float), np.asarray(embedded, dtype=complex)


def _observables(problem: Any, family: str) -> tuple[tuple[str, Any], ...]:
    if family == "peierls_hubbard":
        by_name = {str(element.name): element for element in problem.basis}
        return (
            ("site0_density", by_name["site0_density"].polynomial),
            ("bond_displacement", by_name["bond_X"].polynomial),
        )
    from src.quantum.siam_model import SIAMLayout
    from src.quantum.siam_observables import (
        impurity_double_occupancy_operator,
        impurity_magnetization_operator,
    )

    layout = SIAMLayout(int(problem.key.n_bath))
    return (
        ("impurity_magnetization", impurity_magnetization_operator(layout)),
        ("impurity_double_occupancy", impurity_double_occupancy_operator(layout)),
    )


def _broadened_spectrum(
    omega: np.ndarray, energies: Sequence[float], strengths: Sequence[float]
) -> np.ndarray:
    values = np.zeros_like(omega, dtype=float)
    for energy, strength in zip(energies, strengths):
        offset = omega - float(energy)
        values += float(strength) * (SPECTRAL_ETA / np.pi) / (
            offset * offset + SPECTRAL_ETA * SPECTRAL_ETA
        )
    return values


def _spectral_comparison(
    *,
    exact_energies: Sequence[float],
    exact_strengths: Sequence[float],
    approximate_energies: Sequence[float],
    approximate_strengths: Sequence[float],
) -> dict[str, Any]:
    upper = max(
        max((float(x) for x in exact_energies), default=1.0),
        max((float(x) for x in approximate_energies), default=1.0),
    ) + 0.5
    omega = np.linspace(0.0, upper, SPECTRAL_GRID_POINTS)
    exact = _broadened_spectrum(omega, exact_energies, exact_strengths)
    approximate = _broadened_spectrum(
        omega, approximate_energies, approximate_strengths
    )
    normalization = max(float(np.trapezoid(np.abs(exact), omega)), 1.0e-15)
    return {
        "kernel": "lorentzian",
        "eta": SPECTRAL_ETA,
        "grid_points": SPECTRAL_GRID_POINTS,
        "omega_min": 0.0,
        "omega_max": float(upper),
        "normalized_l1_error": float(
            np.trapezoid(np.abs(approximate - exact), omega) / normalization
        ),
        "integrated_exact_weight": float(np.trapezoid(exact, omega)),
        "integrated_approximate_weight": float(np.trapezoid(approximate, omega)),
    }


def _static_spectral_audit(
    problem: Any,
    family: str,
    *,
    support: Sequence[int] | None = None,
    direct_ritz_vectors: np.ndarray | None = None,
) -> dict[str, Any]:
    exact_values, exact_vectors = _exact_eigensystem(problem, family)
    exact_frequencies = np.asarray(
        exact_values[1 : TARGET_ROOTS + 1] - exact_values[0], dtype=float
    )
    observables = _observables(problem, family)
    exact_excited = exact_vectors[:, 1 : TARGET_ROOTS + 1]
    observable_payload: dict[str, Any] = {}

    if support is not None:
        qse_observables = tuple(
            polynomial_observable(polynomial, name=name)
            for name, polynomial in observables
        )
        qse = compute_qse_spectra(
            problem.hamiltonian,
            problem.ground,
            tuple(problem.basis[int(index)] for index in support),
            basis_vector_policy=growth._Q0_POLICY,
            transition_observables=qse_observables,
        )
        approximate_values = np.asarray(qse.eigenvalues[:TARGET_ROOTS], dtype=float)
        approximate_frequencies = approximate_values - float(problem.ground_energy)
        approximate_strengths = {
            str(row.observable.name): np.asarray(
                row.transition_strengths[:TARGET_ROOTS], dtype=float
            )
            for row in qse.transition_observables
        }
        retained_rank = int(qse.retained_rank)
    elif direct_ritz_vectors is not None:
        vectors = np.asarray(direct_ritz_vectors, dtype=complex)
        approximate_values = np.asarray(
            [
                np.vdot(vectors[:, root], _apply(problem.hamiltonian, vectors[:, root])).real
                for root in range(1, min(TARGET_ROOTS + 1, vectors.shape[1]))
            ],
            dtype=float,
        )
        approximate_frequencies = approximate_values - float(problem.ground_energy)
        approximate_strengths = {}
        for name, polynomial in observables:
            source = _apply(polynomial, problem.ground)
            approximate_strengths[name] = np.abs(
                vectors[:, 1 : TARGET_ROOTS + 1].conj().T @ source
            ) ** 2
        retained_rank = int(vectors.shape[1])
    else:  # pragma: no cover - internal contract
        raise ValueError("spectral audit needs a record support or Davidson Ritz vectors.")

    for name, polynomial in observables:
        source = _apply(polynomial, problem.ground)
        exact_strengths = np.abs(exact_excited.conj().T @ source) ** 2
        approx = np.asarray(approximate_strengths[name], dtype=float)
        count = min(len(exact_frequencies), len(approximate_frequencies), len(approx))
        observable_payload[name] = {
            "exact_transition_strengths": [float(x) for x in exact_strengths[:count]],
            "approximate_transition_strengths": [float(x) for x in approx[:count]],
            "strength_l1_error": float(
                np.sum(np.abs(approx[:count] - exact_strengths[:count]))
            ),
            "broadened_spectrum": _spectral_comparison(
                exact_energies=exact_frequencies[:count],
                exact_strengths=exact_strengths[:count],
                approximate_energies=approximate_frequencies[:count],
                approximate_strengths=approx[:count],
            ),
        }
    return {
        "status": "EVALUATED",
        "post_run_only": True,
        "root_matching": "ascending_index; broadened spectrum is primary near degeneracy",
        "retained_rank": retained_rank,
        "exact_frequencies": [float(x) for x in exact_frequencies],
        "approximate_frequencies": [float(x) for x in approximate_frequencies],
        "observables": observable_payload,
    }


def _apply(polynomial: Any, state: np.ndarray) -> np.ndarray:
    return np.asarray(
        apply_compiled_polynomial(
            np.asarray(state, dtype=complex), compile_polynomial_action(polynomial)
        ),
        dtype=complex,
    )


def _resource_payload(
    problem: Any,
    terminal: Mapping[str, Any],
    *,
    support: Sequence[int] | None,
    include_qiskit: bool,
    direct: bool = False,
) -> dict[str, Any]:
    estimator = (
        dict(terminal["estimator"])
        if direct
        else {
            **dict(problem.estimator_cost(tuple(support or ()))),
            "convention": "record_pencil_qwc_basis_cover",
        }
    )
    if not direct:
        estimator["queries"] = int(estimator["qwc_groups"])
    qiskit = (
        {"status": "NOT_REQUESTED"}
        if not include_qiskit
        else (
            _direct_qiskit_cost(problem, terminal)
            if direct
            else _record_qiskit_cost(problem, tuple(support or ()))
        )
    )
    return {
        "online_K_coordinate": "graph_span_n2q",
        "graph_span": {key: float(terminal[key]) for key in ("n2q", "d2q", "dc")},
        "qiskit_compiled": qiskit,
        "estimator": estimator,
    }


def _run_case(
    case_id: str,
    *,
    trace_stride: int,
    include_qiskit: bool,
    verify_determinism: bool,
) -> dict[str, Any]:
    if case_id.startswith("peierls_"):
        family = "peierls_hubbard"
        problem = load_peierls_problem(case_id, target_roots=TARGET_ROOTS)
        direct_cap = 16
    elif case_id == "siam_nb4_pilot":
        family = "siam_finite_box"
        problem = load_siam_problem(
            case_id,
            allow_exact_benchmark_seed=True,
            pool_policy_id=SIAM_POOL_GENERALIZED_SD_COMPLETION_V1,
            target_roots=TARGET_ROOTS,
        )
        direct_cap = 100
    else:
        raise ValueError(f"unsupported generality case {case_id!r}.")

    started = time.monotonic()
    selection = _selection(problem)
    order = [int(index) for index in selection.selected_original_indices]
    stop = dict(selection.geometry_stop or {"stop_reason": "missing"})
    replay = _selection(problem) if verify_determinism else None
    deterministic = (
        None
        if replay is None
        else {
            "selected_support_identical": tuple(order)
            == tuple(int(index) for index in replay.selected_original_indices),
            "stop_reason_identical": str(stop.get("stop_reason"))
            == str((replay.geometry_stop or {}).get("stop_reason")),
            "final_residual_identical": stop.get("final_max_target_residual_norm")
            == (replay.geometry_stop or {}).get("final_max_target_residual_norm"),
        }
    )
    selected_rows, selected_completion = growth.trace_selected_qse_fixed_pool(
        problem,
        order,
        k_max=len(order),
        stride=int(trace_stride),
        record_stop=stop,
        target_root_count=TARGET_ROOTS,
    )
    selected_completion = dict(selected_completion)
    selected_completion["record_stop"] = _compact_selection_receipt(selection)
    selected_terminal = selected_rows[-1]
    selected_support = [int(index) for index in selected_terminal["support_indices"]]

    direct_rows, direct_completion = growth.trace_direct_davidson_with_completion(
        problem,
        k_max=direct_cap,
        target_root_count=TARGET_ROOTS,
        include_terminal_ritz_vectors=True,
    )
    direct_vectors = np.asarray(
        direct_completion.pop("terminal_ritz_vectors"), dtype=complex
    )
    direct_terminal = direct_rows[-1]

    fixed_indices, fixed_policy = _fixed_support(problem, family)
    fixed_terminal = _fixed_row(problem, fixed_indices)
    fixed_rows = [fixed_terminal]

    return {
        "case_id": str(case_id),
        "model_family": family,
        "problem_receipt": problem.arm_receipt(),
        "target_root_count": TARGET_ROOTS,
        "reference_roots": [float(value) for value in problem.references],
        "selector_replay_lock": deterministic,
        "routes": {
            "selected_qse": {
                "method_id": "record_first_adaptive_qse",
                "selection": _compact_selection_receipt(selection),
                "completion": selected_completion,
                "trace": selected_rows,
                "threshold_crossings": _threshold_crossings(selected_rows),
                "terminal_support_indices": selected_support,
                "resources": _resource_payload(
                    problem,
                    selected_terminal,
                    support=selected_support,
                    include_qiskit=include_qiskit,
                ),
                "static_spectral_audit": _static_spectral_audit(
                    problem, family, support=selected_support
                ),
            },
            "direct_davidson": {
                "method_id": "direct_davidson_qse",
                "completion": direct_completion,
                "trace": direct_rows,
                "threshold_crossings": _threshold_crossings(direct_rows),
                "resources": _resource_payload(
                    problem,
                    direct_terminal,
                    support=None,
                    include_qiskit=include_qiskit,
                    direct=True,
                ),
                "static_spectral_audit": _static_spectral_audit(
                    problem, family, direct_ritz_vectors=direct_vectors
                ),
            },
            "fixed_qse": {
                "method_id": "fixed_class_qse",
                "fixed_class_policy_id": fixed_policy,
                "completion": {
                    "terminal_status": "FIXED_CLASS_EVALUATED",
                    "terminal_support_size": len(fixed_indices),
                },
                "trace": fixed_rows,
                "threshold_crossings": _threshold_crossings(fixed_rows),
                "terminal_support_indices": fixed_indices,
                "resources": _resource_payload(
                    problem,
                    fixed_terminal,
                    support=fixed_indices,
                    include_qiskit=include_qiskit,
                ),
                "static_spectral_audit": _static_spectral_audit(
                    problem, family, support=fixed_indices
                ),
            },
        },
        "elapsed_seconds": float(time.monotonic() - started),
    }


def run_campaign(
    case_ids: Sequence[str],
    *,
    trace_stride: int = 2,
    include_qiskit: bool = True,
    verify_determinism: bool = True,
) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    for case_id in case_ids:
        print(f"acquiring {case_id} ...", flush=True)
        cases[str(case_id)] = _run_case(
            str(case_id),
            trace_stride=int(trace_stride),
            include_qiskit=bool(include_qiskit),
            verify_determinism=bool(verify_determinism),
        )
        routes = cases[str(case_id)]["routes"]
        print(
            f"  selected={routes['selected_qse']['completion']['terminal_status']} "
            f"err={routes['selected_qse']['trace'][-1]['max_root_abs_error']:.3e}; "
            f"direct={routes['direct_davidson']['completion']['terminal_status']} "
            f"err={routes['direct_davidson']['trace'][-1]['max_root_abs_error']:.3e}",
            flush=True,
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "source_revision": _git_revision(),
        "controller_boundary": {
            "exact_excited_states_used_online": False,
            "exact_audits_are_post_run_only": True,
            "implicit_fallback_enabled": False,
        },
        "method_contract": {
            "selector_policy_id": PAPER_III_LOCKED_SELECTOR_POLICY_ID,
            "selected_exchange": "certified_atomic_joint_delete_insert",
            "selected_pool_policy_ids": {
                "peierls_hubbard": PEIERLS_POOL_POLICY_ID,
                "siam_finite_box": SIAM_POOL_GENERALIZED_SD_COMPLETION_V1,
            },
            "direct_comparator": "robustly_initialized_quantum_davidson",
            "cost_symbol": "K",
        },
        "trace_stride": int(trace_stride),
        "qiskit_compile_requested": bool(include_qiskit),
        "deterministic_replay_requested": bool(verify_determinism),
        "cases": cases,
    }


def main(argv: Sequence[str] | None = None) -> int:
    default_cases = [str(row[0]) for row in PEIERLS_REGIMES] + ["siam_nb4_pilot"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case", action="append", choices=default_cases)
    parser.add_argument("--trace-stride", type=int, default=2)
    parser.add_argument("--skip-qiskit", action="store_true")
    parser.add_argument("--skip-determinism-replay", action="store_true")
    args = parser.parse_args(argv)
    cases = default_cases if not args.case else list(args.case)
    payload = run_campaign(
        cases,
        trace_stride=int(args.trace_stride),
        include_qiskit=not bool(args.skip_qiskit),
        verify_determinism=not bool(args.skip_determinism_replay),
    )
    path = Path(args.output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt = {
        "schema_version": "paper_iii_qse_generality_run_receipt_v1",
        "campaign_id": CAMPAIGN_ID,
        "output_json": str(path),
        "output_sha256": _sha256(path),
        "case_ids": cases,
        "source_revision": payload["source_revision"],
    }
    receipt_path = path.with_name("run_receipt.json")
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"output_json: {path}", flush=True)
    print(f"run_receipt: {receipt_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
