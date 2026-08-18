#!/usr/bin/env python3
"""Paper III comparator arms: standard fixed-class QSE and real-time Krylov.

Benchmarks the cost-selected QSE construction against the two families a
referee will ask for, on one shared compiled-2Q cost axis:

1. **Standard QSE (fixed class, deterministic)** — McClean-style: choose an
   operator class a priori, use every member, no selection. Two canonical
   classes are drawn from the same stored pool so the comparison isolates
   *selection* rather than pool content:
   - ``ham_terms``: identity + every Hamiltonian unit-term generator;
   - ``linear_response``: identity + fermionic UCCSD-lifted excitations +
     phonon ladder operators.
   Cost = sum of per-element compiled 2Q insertions (same oracle as the
   selection arms).

2. **Real-time Krylov QSE** — states ``|phi_k> = exp(-i H k dt)|psi>`` on a
   time grid (the hardware-honest, unitary Krylov variant; cf. quantum
   Krylov treatments of Hubbard--Holstein). States are evaluated
   statevector-exactly; the compiled cost charges one first-order Trotter
   step of H per grid interval, priced by the same Paper I oracle, so state
   ``k`` costs ``k`` steps and a dimension-``K`` basis costs
   ``c_step * K(K-1)/2``. Measurement (Hadamard-test) overhead is excluded
   on both families alike.

Accuracy on both arms is the absolute first-excited-energy error against the
full-158-basis q0 root-0 reference stored with the 20260802 advisor demo.
Diagnostic evidence driver; writes one JSON summary and never feeds
controller decisions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.qse_spectra.compiled_costs import (
    ORACLE_KIND_BACKEND_TRANSPILE,
    ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
    annotate_basis_with_compiled_costs,
    resolve_cost_weights_preset,
)
from pipelines.qse_spectra.core import (
    QSEBasisVectorPolicy,
    compute_qse_spectra,
    polynomial_basis_element,
)
from pipelines.qse_spectra.io import (
    load_operator_basis_json,
    load_polynomial_json,
    load_state_json,
)
from src.quantum.compiled_polynomial import apply_compiled_polynomial, compile_polynomial_action

GOLDEN_DIR = REPO_ROOT / "output/diagnostics/paper_iii_hh_advisor_demo_20260802_a005"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "output/diagnostics/paper_iii_cost_frontier_arms_20260818_v1/comparator_arms_summary.json"
)

FIXED_CLASSES: dict[str, tuple[str, ...]] = {
    "ham_terms": ("identity", "hh_termwise_ham_unit_term"),
    "linear_response": ("identity", "uccsd_ferm_lifted", "hh_phonon"),
}
_Q0_POLICY = QSEBasisVectorPolicy(
    reference_projection="q0", basis_vector_normalization="raw_projected"
)
_COST_PRESET = "two_qubit_only_v1"


def _dense_hamiltonian(hamiltonian: Any, dim: int) -> np.ndarray:
    compiled = compile_polynomial_action(hamiltonian)
    matrix = np.zeros((dim, dim), dtype=complex)
    for column in range(dim):
        unit = np.zeros(dim, dtype=complex)
        unit[column] = 1.0
        matrix[:, column] = apply_compiled_polynomial(unit, compiled)
    return 0.5 * (matrix + matrix.conj().T)


def _element_family(name: str) -> str:
    return str(name).split("(")[0].split("::")[0]


def _annotate_2q(elements: Sequence[Any], *, oracle_kind: str) -> list[float]:
    rows = annotate_basis_with_compiled_costs(
        elements,
        num_qubits=8,
        oracle_kind=oracle_kind,
        cost_weights=resolve_cost_weights_preset(_COST_PRESET),
    )
    return [float(row.estimate.c_hat_2q) for row in rows]


def run_fixed_class_arms(
    *, hamiltonian: Any, state: np.ndarray, basis: Sequence[Any], reference_energy: float
) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for class_name, families in FIXED_CLASSES.items():
        subset = [element for element in basis if _element_family(element.name) in families]
        span_2q = _annotate_2q(subset, oracle_kind=ORACLE_KIND_MARRAKESH_GRAPH_SPAN)
        transpile_2q = _annotate_2q(subset, oracle_kind=ORACLE_KIND_BACKEND_TRANSPILE)
        result = compute_qse_spectra(
            hamiltonian, state, tuple(subset), basis_vector_policy=_Q0_POLICY
        )
        root0 = float(np.asarray(result.eigenvalues, dtype=float).reshape(-1)[0])
        arms[class_name] = {
            "families": list(families),
            "class_size": len(subset),
            "retained_rank": int(result.retained_rank),
            "total_2q_graph_span": float(sum(span_2q)),
            "total_2q_transpile": float(sum(transpile_2q)),
            "root0_energy": root0,
            "abs_err_vs_reference": abs(root0 - float(reference_energy)),
        }
    return arms


def run_krylov_arm(
    *,
    hamiltonian: Any,
    state: np.ndarray,
    reference_energy: float,
    dt_grid: Sequence[float],
    max_dimension: int,
    overlap_cutoff: float = 1.0e-12,
) -> dict[str, Any]:
    psi = np.asarray(state, dtype=complex).reshape(-1)
    psi = psi / np.linalg.norm(psi)
    dim = int(psi.size)
    dense = _dense_hamiltonian(hamiltonian, dim)
    energies, vectors = np.linalg.eigh(dense)

    # Real-time Krylov from the (near-eigenstate) reference collapses to rank
    # one, so excited-state Krylov schemes evolve a kicked source. Use the
    # normalized Hamiltonian-residual kick (H - <H>)|psi>; source-state
    # preparation cost is excluded on both families alike (the QSE arms
    # likewise exclude measurement-circuit overhead), so the cost axis
    # charges propagation only.
    h_psi = dense @ psi
    source = h_psi - complex(np.vdot(psi, h_psi)) * psi
    source_norm = float(np.linalg.norm(source))
    if source_norm <= 1.0e-14:
        raise ValueError("Hamiltonian-residual kick is zero; reference is an exact eigenstate.")
    source = source / source_norm
    amplitudes = vectors.conj().T @ source

    step_element = polynomial_basis_element(hamiltonian, name="first_order_trotter_step")
    step_2q_span = _annotate_2q([step_element], oracle_kind=ORACLE_KIND_MARRAKESH_GRAPH_SPAN)[0]
    step_2q_transpile = _annotate_2q([step_element], oracle_kind=ORACLE_KIND_BACKEND_TRANSPILE)[0]

    rows: list[dict[str, Any]] = []
    for dt in dt_grid:
        states = []
        for k in range(int(max_dimension)):
            phases = np.exp(-1j * energies * float(dt) * k)
            states.append(vectors @ (phases * amplitudes))
        for dimension in range(1, int(max_dimension) + 1):
            block = states[:dimension]
            overlap = np.array(
                [[np.vdot(a, b) for b in block] for a in block], dtype=complex
            )
            ham = np.array(
                [[np.vdot(a, dense @ b) for b in block] for a in block], dtype=complex
            )
            overlap = 0.5 * (overlap + overlap.conj().T)
            ham = 0.5 * (ham + ham.conj().T)
            eigvals, eigvecs = np.linalg.eigh(overlap)
            retained = eigvals > float(overlap_cutoff) * float(max(eigvals.max(), 0.0))
            if int(retained.sum()) < 1:
                continue
            transform = eigvecs[:, retained] / np.sqrt(eigvals[retained])
            reduced = transform.conj().T @ ham @ transform
            roots = np.sort(np.linalg.eigvalsh(0.5 * (reduced + reduced.conj().T)))
            # The kicked source is (near-)orthogonal to the ground state, so
            # the root approximating the first excitation is whichever pencil
            # root lands closest to it; taking the best-matching root is the
            # Krylov-favoring (conservative) convention.
            errors = np.abs(roots - float(reference_energy))
            best = int(np.argmin(errors))
            steps_total = dimension * (dimension - 1) // 2
            rows.append(
                {
                    "dt": float(dt),
                    "krylov_dimension": int(dimension),
                    "retained_rank": int(retained.sum()),
                    "trotter_steps_total": int(steps_total),
                    "cum_2q_graph_span": float(step_2q_span * steps_total),
                    "cum_2q_transpile": float(step_2q_transpile * steps_total),
                    "best_matching_root_energy": float(roots[best]),
                    "abs_err_vs_reference": float(errors[best]),
                }
            )
    # Best-accuracy-per-cost envelope across dt values (favors Krylov).
    envelope: list[dict[str, Any]] = []
    best_err = float("inf")
    for row in sorted(rows, key=lambda item: (item["cum_2q_graph_span"], item["abs_err_vs_reference"])):
        if row["abs_err_vs_reference"] < best_err:
            best_err = row["abs_err_vs_reference"]
            envelope.append(row)
    return {
        "state_preparation_model": "first_order_trotter_one_step_per_grid_interval",
        "trotter_step_2q_graph_span": float(step_2q_span),
        "trotter_step_2q_transpile": float(step_2q_transpile),
        "propagation": "statevector_exact",
        "rows": rows,
        "best_per_cost_envelope": envelope,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-dir", type=Path, default=GOLDEN_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dt-grid", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    parser.add_argument("--max-krylov-dimension", type=int, default=12)
    args = parser.parse_args(argv)

    source_seed = args.golden_dir / "source_seed.json"
    qse_result = args.golden_dir / "qse_result.json"
    hamiltonian, _ = load_polynomial_json(source_seed)
    state, _ = load_state_json(source_seed, state_key="auto")
    basis, _ = load_operator_basis_json(qse_result, nq=8)
    reference_energy = float(
        json.loads(qse_result.read_text(encoding="utf-8"))["eigenvalues"][0]["energy"]
    )

    payload = {
        "schema_version": "paper_iii_qse_comparator_arms_v1",
        "policy": "diagnostic_only_comparator_benchmark",
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "post_run_diagnostic_only": True,
        },
        "cost_weights_preset": _COST_PRESET,
        "reference_full_basis_root0_energy": reference_energy,
        "fixed_class_arms": run_fixed_class_arms(
            hamiltonian=hamiltonian,
            state=state,
            basis=basis,
            reference_energy=reference_energy,
        ),
        "krylov_arm": run_krylov_arm(
            hamiltonian=hamiltonian,
            state=state,
            reference_energy=reference_energy,
            dt_grid=list(args.dt_grid),
            max_dimension=int(args.max_krylov_dimension),
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"reference root0: {reference_energy:.9f}")
    for name, arm in payload["fixed_class_arms"].items():
        print(
            f"fixed[{name}]: size={arm['class_size']} 2Q(span)={arm['total_2q_graph_span']:.0f} "
            f"|err|={arm['abs_err_vs_reference']:.2e}"
        )
    for row in payload["krylov_arm"]["best_per_cost_envelope"]:
        print(
            f"krylov dt={row['dt']} K={row['krylov_dimension']}: "
            f"2Q(span)={row['cum_2q_graph_span']:.0f} |err|={row['abs_err_vs_reference']:.2e}"
        )
    print(f"output_json: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
