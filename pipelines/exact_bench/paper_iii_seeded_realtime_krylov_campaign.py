#!/usr/bin/env python3
"""Acquire the fixed-policy seeded real-time-Krylov comparator matrix.

The online Krylov construction is identical across Hubbard--Holstein,
Peierls--Hubbard, and finite-box SIAM.  Exact roots and eigenvectors are hidden
from the route object and are supplied only after the trace terminates, when
energy errors, root/subspace overlaps, and static transition spectra are
audited.

This is a candidate-evidence driver.  It neither edits the Paper-III manuscript
nor silently substitutes Davidson.  Hamiltonian-power QKSD remains an explicit
implementation-pending comparator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.paper_iii_qse_generality_campaign import (
    _spectral_comparison,
)
from pipelines.exact_bench.paper_iii_qse_peierls_pilot import PEIERLS_REGIMES
from pipelines.qse_spectra.hh_response_observables import (
    HHResponseLayout,
    phonon_displacement_operator,
    site_density_operator,
)
from pipelines.qse_spectra.paper_iii_problem import load_problem
from pipelines.qse_spectra.peierls_static_adapter import load_peierls_problem
from pipelines.qse_spectra.seeded_realtime_krylov import (
    SEEDED_REALTIME_KRYLOV_POLICY_ID,
    SeededRealtimeKrylovConfig,
    SeededRealtimeKrylovRun,
    audit_krylov_energies,
    matched_terminal_ritz_vectors,
    run_seeded_realtime_krylov,
)
from pipelines.qse_spectra.siam_static_adapter import (
    SIAM_POOL_GENERALIZED_SD_COMPLETION_V1,
    load_siam_problem,
)
from pipelines.qse_spectra.static_campaign import HH_CUTOFFS, HH_REGIMES
from src.quantum.compiled_polynomial import (
    apply_compiled_polynomial,
    compile_polynomial_action,
)


SCHEMA_VERSION = "paper_iii_seeded_realtime_krylov_matrix_v1"
CAMPAIGN_ID = "paper_iii_seeded_realtime_krylov_matrix_20260829_v1"
DEFAULT_OUTPUT = REPO_ROOT / "output/diagnostics" / CAMPAIGN_ID / "matrix.json"
TARGET_ROOTS = 6
REFERENCE_ROOTS = TARGET_ROOTS + 1
ERROR_TARGETS = (1.0e-2, 1.0e-4, 1.0e-6)
HH_K_MAX = 100
PEIERLS_K_MAX = 40
SIAM_K_MAX = 100


@dataclass(frozen=True)
class _OnlineProblemView:
    """Narrow online interface with no exact excited-state properties."""

    num_qubits: int
    hamiltonian: Any
    ground: np.ndarray
    n_ph_max: int | None = None
    key: Any = None


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


def _online_view(problem: Any) -> _OnlineProblemView:
    return _OnlineProblemView(
        num_qubits=int(problem.num_qubits),
        hamiltonian=problem.hamiltonian,
        ground=np.asarray(problem.ground, dtype=complex),
        n_ph_max=(
            None if not hasattr(problem, "n_ph_max") else int(problem.n_ph_max)
        ),
        key=getattr(problem, "key", None),
    )


def _load_case(case_id: str) -> tuple[str, Any, int]:
    if case_id.startswith("hh::"):
        _, regime, cutoff = case_id.split("::")
        n_ph_max = int(cutoff.removeprefix("nph"))
        try:
            _name, u, g_ep = next(row for row in HH_REGIMES if row[0] == regime)
        except StopIteration as exc:  # pragma: no cover - CLI validation
            raise ValueError(f"unknown Hubbard--Holstein regime {regime!r}.") from exc
        return (
            "hubbard_holstein",
            load_problem(
                regime=regime,
                u=float(u),
                g_ep=float(g_ep),
                n_ph_max=n_ph_max,
                target_roots=REFERENCE_ROOTS,
            ),
            HH_K_MAX,
        )
    if case_id.startswith("peierls_"):
        return (
            "peierls_hubbard",
            load_peierls_problem(case_id, target_roots=REFERENCE_ROOTS),
            PEIERLS_K_MAX,
        )
    if case_id == "siam_nb4_pilot":
        return (
            "siam_finite_box",
            load_siam_problem(
                case_id,
                allow_exact_benchmark_seed=True,
                pool_policy_id=SIAM_POOL_GENERALIZED_SD_COMPLETION_V1,
                target_roots=REFERENCE_ROOTS,
            ),
            SIAM_K_MAX,
        )
    raise ValueError(f"unknown Krylov comparator case {case_id!r}.")


def _all_case_ids() -> tuple[str, ...]:
    hh = tuple(
        f"hh::{regime}::nph{n_ph_max}"
        for n_ph_max in HH_CUTOFFS
        for regime, _u, _g_ep in HH_REGIMES
    )
    peierls = tuple(str(row[0]) for row in PEIERLS_REGIMES)
    return hh + peierls + ("siam_nb4_pilot",)


def _apply(polynomial: Any, state: np.ndarray) -> np.ndarray:
    return np.asarray(
        apply_compiled_polynomial(
            np.asarray(state, dtype=complex), compile_polynomial_action(polynomial)
        ),
        dtype=complex,
    )


def _exact_sector_eigensystem(
    problem: Any, sector_indices: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = tuple(int(index) for index in sector_indices)
    full_dimension = 1 << int(problem.num_qubits)
    compiled = compile_polynomial_action(problem.hamiltonian)
    restricted = np.zeros((len(indices), len(indices)), dtype=complex)
    for column, basis_index in enumerate(indices):
        basis = np.zeros(full_dimension, dtype=complex)
        basis[basis_index] = 1.0
        restricted[:, column] = np.asarray(
            apply_compiled_polynomial(basis, compiled), dtype=complex
        )[list(indices)]
    restricted = 0.5 * (restricted + restricted.conj().T)
    values, vectors = np.linalg.eigh(restricted)
    return (
        np.asarray(values, dtype=float),
        np.asarray(vectors, dtype=complex),
        restricted,
    )


def _observables(problem: Any, family: str) -> tuple[tuple[str, Any], ...]:
    if family == "hubbard_holstein":
        layout = HHResponseLayout(
            num_sites=2,
            n_ph_max=int(problem.n_ph_max),
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
            total_qubits=int(problem.num_qubits),
            num_particles=(1, 1),
        )
        return (
            ("site0_density", site_density_operator(layout, site=0)),
            (
                "site0_displacement",
                phonon_displacement_operator(layout, site=0),
            ),
        )
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
        (
            "impurity_double_occupancy",
            impurity_double_occupancy_operator(layout),
        ),
    )


def _append_window_audit(
    rows: Sequence[Mapping[str, Any]], exact_references: Sequence[float]
) -> list[dict[str, Any]]:
    references = np.asarray(exact_references, dtype=float)
    if references.size < REFERENCE_ROOTS:
        raise ValueError(
            f"R-to-R+1 audit needs {REFERENCE_ROOTS} excited roots; "
            f"got {references.size}."
        )
    out: list[dict[str, Any]] = []
    exact_gap = float(references[TARGET_ROOTS] - references[TARGET_ROOTS - 1])
    for raw in rows:
        row = dict(raw)
        roots = np.asarray(row["pencil_root_energies"], dtype=float)
        if row["target_window_status"] != "ONE_TO_ONE_MATCHED":
            row.update(
                {
                    "r_plus_1_pencil_root_index": None,
                    "r_plus_1_root_energy": None,
                    "r_to_r_plus_1_gap": None,
                    "exact_r_to_r_plus_1_gap": exact_gap,
                }
            )
            out.append(row)
            continue
        used = {int(index) for index in row["matched_pencil_root_indices"]}
        available = [index for index in range(len(roots)) if index not in used]
        if not available:
            row.update(
                {
                    "r_plus_1_pencil_root_index": None,
                    "r_plus_1_root_energy": None,
                    "r_to_r_plus_1_gap": None,
                    "exact_r_to_r_plus_1_gap": exact_gap,
                }
            )
            out.append(row)
            continue
        r_plus_1_index = int(
            min(
                available,
                key=lambda index: abs(
                    float(roots[index]) - float(references[TARGET_ROOTS])
                ),
            )
        )
        r_plus_1 = float(roots[r_plus_1_index])
        row.update(
            {
                "r_plus_1_pencil_root_index": r_plus_1_index,
                "r_plus_1_root_energy": r_plus_1,
                "r_to_r_plus_1_gap": float(r_plus_1 - row["rth_root_energy"]),
                "exact_r_to_r_plus_1_gap": exact_gap,
            }
        )
        out.append(row)
    return out


def _threshold_crossings(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    crossings: dict[str, Any] = {}
    for target in ERROR_TARGETS:
        crossing = next(
            (
                row
                for row in rows
                if row["max_root_abs_error"] is not None
                and float(row["max_root_abs_error"]) <= float(target)
            ),
            None,
        )
        if crossing is None:
            crossings[f"{target:.0e}"] = {"status": "NOT_REACHED_WITHIN_CAP"}
            continue
        resources = dict(crossing["resources"])
        crossings[f"{target:.0e}"] = {
            "status": "REACHED",
            "k": int(crossing["k"]),
            "retained_rank": int(crossing["retained_rank"]),
            "K": float(crossing["K"]),
            "max_root_abs_error": float(crossing["max_root_abs_error"]),
            "per_root_abs_error": list(crossing["per_root_abs_error"]),
            "matched_root_energies": list(crossing["matched_root_energies"]),
            "rth_root_energy": float(crossing["rth_root_energy"]),
            "r_to_r_plus_1_gap": float(crossing["r_to_r_plus_1_gap"]),
            "graph_span": dict(resources["graph_span"]),
            "qiskit_compiled": dict(resources["qiskit_compiled"]),
            "estimator": dict(crossing["estimator"]),
        }
    return crossings


def _static_spectral_audit(
    problem: Any,
    family: str,
    run: SeededRealtimeKrylovRun,
    audited_trace: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    exact_values, exact_vectors, restricted_h = _exact_sector_eigensystem(
        problem, run.sector_indices
    )
    declared = np.asarray(problem.spectrum, dtype=float)
    comparison_count = min(len(declared), REFERENCE_ROOTS + 1)
    if not np.allclose(
        exact_values[:comparison_count], declared[:comparison_count], atol=1.0e-10
    ):
        raise ValueError("post-run exact eigensystem disagrees with declared references.")
    exact_excited = exact_vectors[:, 1 : TARGET_ROOTS + 1]
    approximate = matched_terminal_ritz_vectors(run, audited_trace)
    overlap_matrix = np.abs(exact_excited.conj().T @ approximate) ** 2
    singular_values = np.linalg.svd(
        exact_excited.conj().T @ approximate, compute_uv=False
    )
    terminal = dict(audited_trace[-1])
    approximate_energies = np.asarray(terminal["matched_root_energies"], dtype=float)
    exact_frequencies = exact_values[1 : TARGET_ROOTS + 1] - exact_values[0]
    approximate_frequencies = approximate_energies - exact_values[0]
    residual_norms = [
        float(np.linalg.norm(restricted_h @ approximate[:, root] - approximate_energies[root] * approximate[:, root]))
        for root in range(TARGET_ROOTS)
    ]
    ground_full = np.asarray(problem.ground, dtype=complex)
    sector = list(run.sector_indices)
    observable_payload: dict[str, Any] = {}
    for name, polynomial in _observables(problem, family):
        source = _apply(polynomial, ground_full)[sector]
        exact_strengths = np.abs(exact_excited.conj().T @ source) ** 2
        approximate_strengths = np.abs(approximate.conj().T @ source) ** 2
        observable_payload[str(name)] = {
            "exact_transition_strengths": [float(value) for value in exact_strengths],
            "approximate_transition_strengths": [
                float(value) for value in approximate_strengths
            ],
            "strength_l1_error": float(
                np.sum(np.abs(approximate_strengths - exact_strengths))
            ),
            "broadened_spectrum": _spectral_comparison(
                exact_energies=exact_frequencies,
                exact_strengths=exact_strengths,
                approximate_energies=approximate_frequencies,
                approximate_strengths=approximate_strengths,
            ),
        }
    return {
        "status": "EVALUATED",
        "post_run_only": True,
        "root_matching": "minimum_total_abs_energy_hungarian_one_to_one",
        "matched_root_overlap_squared": [
            float(overlap_matrix[index, index]) for index in range(TARGET_ROOTS)
        ],
        "root_overlap_squared_matrix": [
            [float(value) for value in row] for row in overlap_matrix
        ],
        "target_subspace_principal_cosines_squared": [
            float(value * value) for value in singular_values
        ],
        "residual_norms": residual_norms,
        "exact_frequencies": [float(value) for value in exact_frequencies],
        "approximate_frequencies": [
            float(value) for value in approximate_frequencies
        ],
        "observables": observable_payload,
    }


def _problem_receipt(problem: Any, family: str) -> dict[str, Any]:
    base = dict(problem.arm_receipt())
    if family == "hubbard_holstein":
        base.update(
            {
                "model_family": family,
                "case_id": str(problem.regime),
                "n_ph_max": int(problem.n_ph_max),
            }
        )
    return base


def _run_case(case_id: str, *, include_qiskit: bool) -> dict[str, Any]:
    family, problem, k_max = _load_case(case_id)
    started = time.monotonic()
    run = run_seeded_realtime_krylov(
        _online_view(problem),
        config=SeededRealtimeKrylovConfig(
            k_max=int(k_max),
            target_root_count=TARGET_ROOTS,
            include_qiskit=bool(include_qiskit),
        ),
    )
    exact_references = tuple(float(value) for value in problem.references)
    energy_trace = _append_window_audit(
        audit_krylov_energies(run, exact_references), exact_references
    )
    terminal = energy_trace[-1]
    return {
        "case_id": str(case_id),
        "model_family": family,
        "problem_receipt": _problem_receipt(problem, family),
        "method_id": "seeded_realtime_krylov",
        "policy_id": SEEDED_REALTIME_KRYLOV_POLICY_ID,
        "policy_receipt": dict(run.policy_receipt),
        "source_receipt": dict(run.source_receipt),
        "resource_receipt": dict(run.resource_receipt),
        "reference_boundary": {
            "exact_excited_roots_available_to_online_route": False,
            "exact_roots_supplied_after_route_termination": True,
            "exact_eigenvectors_supplied_after_route_termination": True,
        },
        "target_root_count": TARGET_ROOTS,
        "reference_roots": [
            float(value) for value in exact_references[:TARGET_ROOTS]
        ],
        "reference_r_plus_1_root": float(exact_references[TARGET_ROOTS]),
        "trace": energy_trace,
        "threshold_crossings": _threshold_crossings(energy_trace),
        "terminal": {
            "status": "CAP_REACHED",
            "k": int(terminal["k"]),
            "retained_rank": int(terminal["retained_rank"]),
            "K": float(terminal["K"]),
            "matched_root_energies": list(terminal["matched_root_energies"]),
            "per_root_abs_error": list(terminal["per_root_abs_error"]),
            "max_root_abs_error": float(terminal["max_root_abs_error"]),
            "rth_root_energy": float(terminal["rth_root_energy"]),
            "r_to_r_plus_1_gap": float(terminal["r_to_r_plus_1_gap"]),
            "graph_span": dict(terminal["resources"]["graph_span"]),
            "qiskit_compiled": dict(
                terminal["resources"]["qiskit_compiled"]
            ),
            "estimator": dict(terminal["estimator"]),
        },
        "static_spectral_audit": _static_spectral_audit(
            problem, family, run, energy_trace
        ),
        "elapsed_seconds": float(time.monotonic() - started),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cases",
        default=None,
        help="Comma-separated case IDs; default is the full 16-case matrix.",
    )
    parser.add_argument(
        "--include-qiskit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile the repeated Hamiltonian step on FakeMarrakesh.",
    )
    args = parser.parse_args(argv)

    declared = _all_case_ids()
    cases = (
        declared
        if args.cases is None
        else tuple(token.strip() for token in str(args.cases).split(",") if token.strip())
    )
    unknown = sorted(set(cases) - set(declared))
    if unknown:
        raise ValueError(f"undeclared cases: {unknown!r}.")

    started = time.monotonic()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "evidence_class": "candidate",
        "promotion_decision": "UNRESOLVED_USER_DECISION",
        "git_revision": _git_revision(),
        "route_contract": {
            "method_id": "seeded_realtime_krylov",
            "policy_id": SEEDED_REALTIME_KRYLOV_POLICY_ID,
            "same_policy_across_model_families": True,
            "target_root_count": TARGET_ROOTS,
            "matched_error_thresholds": list(ERROR_TARGETS),
            "online_exact_reference_use": False,
            "exact_audit_stage": "strictly_post_run",
            "root_matching": "minimum_total_abs_energy_hungarian_one_to_one",
            "K": "aggregate_graph_span_n2q",
            "qiskit_cost": "aggregate_FakeMarrakesh_transpiled_hamiltonian_steps",
            "estimator_queries": "all_independent_S_and_H_pencil_elements_no_qwc_reuse",
        },
        "cases": {},
        "comparators": {
            "hamiltonian_power_qksd": {
                "status": "IMPLEMENTATION_PENDING",
                "substituted_method": None,
                "reason": "No faithful Hamiltonian-power QKSD implementation is admitted.",
            }
        },
    }
    for case_id in cases:
        print(f"== {case_id}", flush=True)
        payload["cases"][case_id] = _run_case(
            case_id, include_qiskit=bool(args.include_qiskit)
        )
        crossing = payload["cases"][case_id]["threshold_crossings"]["1e-06"]
        print(
            f"   1e-6={crossing['status']} "
            f"k={crossing.get('k')} K={crossing.get('K')}",
            flush=True,
        )
    payload["elapsed_seconds"] = float(time.monotonic() - started)

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    artifact_label = (
        str(output.relative_to(REPO_ROOT))
        if output.is_relative_to(REPO_ROOT)
        else str(output)
    )
    receipt = {
        "schema_version": f"{SCHEMA_VERSION}_receipt",
        "campaign_id": CAMPAIGN_ID,
        "artifact": artifact_label,
        "sha256": _sha256(output),
        "case_count": len(payload["cases"]),
        "qksd_status": payload["comparators"]["hamiltonian_power_qksd"]["status"],
        "all_1e_minus_6_reached": all(
            case["threshold_crossings"]["1e-06"]["status"] == "REACHED"
            for case in payload["cases"].values()
        ),
    }
    receipt_path = output.with_name("receipt.json")
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
