#!/usr/bin/env python3
"""Paper III multi-regime accuracy-vs-compiled-2Q frontier sweep (HH dimer).

Runs the cost-selected QSE construction and its baselines across the
canonical Hubbard--Holstein dimer coupling regimes at a fixed
``L=2, n_ph_max=3`` layout (8 qubits), so the same structural operator pool
and per-element compiled costs apply everywhere and only the physics
changes. Regime parameters follow the Table-I three-model points
(``pipelines/exact_bench/table_i_canonical_cases.py``), which canonically
assign different phonon truncations to the strong-phonon regimes; this
sweep holds the truncation at nph3 by design so the pool is shared, and
records that choice.

Per regime: the reference is the exact ground state of the half-filled
``(n_up, n_dn) = (1, 1)`` sector (dense diagonalization; fermion numbers
are conserved so eigenstates classify exactly), and the accuracy target is
the exact first excited energy of that sector. Arms: input-order, cheapest
first, geometry selection (subtractive and utility/cost^alpha for alpha in
the sweep), each with the admitted-prefix frontier; plus the complete
fixed linear-response class and the full-pool q0 root as the manifold
limit. Costs are compiled 2Q (two_qubit_only_v1) from the Marrakesh
graph-span oracle, annotated once for the shared pool. Diagnostic evidence
driver; never feeds controller decisions.
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
    ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
    annotate_basis_with_compiled_costs,
    build_accuracy_cost_frontier,
    resolve_cost_weights_preset,
)
from pipelines.qse_spectra.core import (
    QSEBasisVectorPolicy,
    compute_qse_spectra,
)
from pipelines.qse_spectra.io import load_operator_basis_json, load_polynomial_json
from pipelines.qse_spectra.record_selection import (
    StaticRecordSelectionConfig,
    select_static_qse_records,
)
from src.quantum.compiled_polynomial import apply_compiled_polynomial, compile_polynomial_action

GOLDEN_QSE_RESULT = (
    REPO_ROOT / "output/diagnostics/paper_iii_hh_advisor_demo_20260802_a005/qse_result.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "output/diagnostics/paper_iii_regime_frontier_sweep_20260818_v1/regime_frontier_sweep.json"
)

# (regime, u, g_ep); t=1, omega0=1, dv=0, nph3 binary blocked open throughout.
REGIMES: tuple[tuple[str, float, float], ...] = (
    ("demo_weak", 0.25, 0.353553390593),
    ("weak_weak", 0.5, 0.5),
    ("strong_weak", 1.5, 0.5),
    ("weak_strong", 0.5, 0.8660254037844386),
    ("strong_strong", 1.5, 0.8660254037844386),
    ("strong_weak_u8", 8.0, 0.5),
)
_Q0_POLICY = QSEBasisVectorPolicy(
    reference_projection="q0", basis_vector_normalization="raw_projected"
)
_LINEAR_RESPONSE_FAMILIES = ("identity", "uccsd_ferm_lifted", "hh_phonon")
_FERMION_QUBITS_UP = (0, 1)
_FERMION_QUBITS_DN = (2, 3)
_NQ = 8
_BUDGET = 40


def _settings_payload(u: float, g_ep: float) -> dict[str, Any]:
    return {
        "settings": {
            "problem": "hh",
            "L": 2,
            "t": 1.0,
            "u": float(u),
            "omega0": 1.0,
            "g_ep": float(g_ep),
            "dv": 0.0,
            "n_ph_max": 3,
            "boson_encoding": "binary",
            "ordering": "blocked",
            "boundary": "open",
        },
        "adapt_vqe": {"num_particles": {"n_up": 1, "n_dn": 1}},
    }


def _dense_hamiltonian(hamiltonian: Any, dim: int) -> np.ndarray:
    compiled = compile_polynomial_action(hamiltonian)
    matrix = np.zeros((dim, dim), dtype=complex)
    for column in range(dim):
        unit = np.zeros(dim, dtype=complex)
        unit[column] = 1.0
        matrix[:, column] = apply_compiled_polynomial(unit, compiled)
    return 0.5 * (matrix + matrix.conj().T)


def _sector_reference(dense: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Exact ground state and first two energies of the (1,1) fermion sector.

    The sector restriction is exact (fermion numbers are diagonal in the
    computational basis), never expectation-based: eigenstates that are
    degenerate across number sectors (e.g. the u=8 spin triplet, exactly
    degenerate with its S_z = +-1 partners in the (2,0)/(0,2) sectors) would
    otherwise be mixed by the dense eigensolver and silently dropped,
    corrupting the reference.
    """

    dim = int(dense.shape[0])
    occupations_up = np.array(
        [sum((index >> qubit) & 1 for qubit in _FERMION_QUBITS_UP) for index in range(dim)]
    )
    occupations_dn = np.array(
        [sum((index >> qubit) & 1 for qubit in _FERMION_QUBITS_DN) for index in range(dim)]
    )
    sector = np.where((occupations_up == 1) & (occupations_dn == 1))[0]
    if int(sector.size) < 2:
        raise ValueError("(1,1) sector has fewer than two basis states; check the Hamiltonian.")
    restricted = dense[np.ix_(sector, sector)]
    energies, vectors = np.linalg.eigh(0.5 * (restricted + restricted.conj().T))
    ground = np.zeros(dim, dtype=complex)
    ground[sector] = vectors[:, 0]
    return ground, float(energies[0]), float(energies[1])


def _sector_spectrum(dense: np.ndarray, *, count: int = 8) -> tuple[np.ndarray, list[float]]:
    """Exact (1,1)-sector ground state and lowest ``count`` sector energies."""

    dim = int(dense.shape[0])
    occupations_up = np.array(
        [sum((index >> qubit) & 1 for qubit in _FERMION_QUBITS_UP) for index in range(dim)]
    )
    occupations_dn = np.array(
        [sum((index >> qubit) & 1 for qubit in _FERMION_QUBITS_DN) for index in range(dim)]
    )
    sector = np.where((occupations_up == 1) & (occupations_dn == 1))[0]
    restricted = dense[np.ix_(sector, sector)]
    energies, vectors = np.linalg.eigh(0.5 * (restricted + restricted.conj().T))
    ground = np.zeros(dim, dtype=complex)
    ground[sector] = vectors[:, 0]
    return ground, [float(value) for value in energies[: int(count)]]


def _element_family(name: str) -> str:
    return str(name).split("(")[0].split("::")[0]


def _frontier_summary(frontier: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in frontier["rows"]:
        if row.get("solve_status") != "solved":
            continue
        errors = row.get("root_abs_errors_vs_reference") or []
        rows.append(
            {
                "prefix_size": row["prefix_size"],
                "cum_2q": row["cumulative_c_hat_2q"],
                "abs_err_E1": errors[0] if errors else None,
            }
        )
    return rows


def run_sweep(
    *, alphas: Sequence[float], budget: int, output_json: Path
) -> dict[str, Any]:
    basis, _ = load_operator_basis_json(GOLDEN_QSE_RESULT, nq=_NQ)
    cost_rows = annotate_basis_with_compiled_costs(
        basis,
        num_qubits=_NQ,
        oracle_kind=ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
        cost_weights=resolve_cost_weights_preset("two_qubit_only_v1"),
    )
    costs = tuple(row.scalarized_canonical_cost for row in cost_rows)
    linear_response = [
        element for element in basis if _element_family(element.name) in _LINEAR_RESPONSE_FAMILIES
    ]
    linear_response_indices = [
        index
        for index, element in enumerate(basis)
        if _element_family(element.name) in _LINEAR_RESPONSE_FAMILIES
    ]

    scratch = output_json.parent / "settings"
    scratch.mkdir(parents=True, exist_ok=True)
    regimes_payload: dict[str, Any] = {}
    for regime, u, g_ep in REGIMES:
        settings_path = scratch / f"hh_{regime}_settings.json"
        settings_path.write_text(
            json.dumps(_settings_payload(u, g_ep), indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        hamiltonian, _prov = load_polynomial_json(settings_path)
        dense = _dense_hamiltonian(hamiltonian, 1 << _NQ)
        ground, e0_exact, e1_exact = _sector_reference(dense)

        full_result = compute_qse_spectra(
            hamiltonian, ground, tuple(basis), basis_vector_policy=_Q0_POLICY
        )
        manifold_root0 = float(np.asarray(full_result.eigenvalues, dtype=float).reshape(-1)[0])

        arms: dict[str, Any] = {}

        def _run_arm(arm_name: str, config: StaticRecordSelectionConfig, **kwargs: Any) -> None:
            selection = select_static_qse_records(basis, config=config, **kwargs)
            selected_rows = tuple(
                cost_rows[int(index)] for index in selection.selected_original_indices
            )
            frontier = build_accuracy_cost_frontier(
                selection.selected_basis_elements,
                selected_rows,
                hamiltonian=hamiltonian,
                prepared_state=ground,
                basis_vector_policy=_Q0_POLICY,
                reference_energies=[e1_exact],
            )
            arms[arm_name] = {"frontier": _frontier_summary(frontier)}

        _run_arm(
            "input_order",
            StaticRecordSelectionConfig(mode="input_order", max_records=int(budget)),
        )
        _run_arm(
            "compiled_cost",
            StaticRecordSelectionConfig(mode="compiled_cost", max_records=int(budget)),
            compiled_costs=costs,
        )
        geometry_kwargs = dict(
            hamiltonian=hamiltonian,
            prepared_state=ground,
            basis_vector_policy=_Q0_POLICY,
            compiled_costs=costs,
        )
        _run_arm(
            "geometry_subtractive",
            StaticRecordSelectionConfig(mode="geometry_selected", max_records=int(budget)),
            **geometry_kwargs,
        )
        for alpha in alphas:
            _run_arm(
                f"geometry_alpha{alpha:g}",
                StaticRecordSelectionConfig(
                    mode="geometry_selected",
                    max_records=int(budget),
                    geometry_cost_discount_alpha=float(alpha),
                ),
                **geometry_kwargs,
            )

        lr_result = compute_qse_spectra(
            hamiltonian, ground, tuple(linear_response), basis_vector_policy=_Q0_POLICY
        )
        lr_root0 = float(np.asarray(lr_result.eigenvalues, dtype=float).reshape(-1)[0])
        arms["fixed_linear_response_complete"] = {
            "class_size": len(linear_response),
            "total_2q": float(sum(costs[index] for index in linear_response_indices)),
            "abs_err_E1": abs(lr_root0 - e1_exact),
        }

        regimes_payload[regime] = {
            "u": float(u),
            "g_ep": float(g_ep),
            "e0_exact_sector": e0_exact,
            "e1_exact_sector": e1_exact,
            "exact_gap": e1_exact - e0_exact,
            "manifold_limit_abs_err_E1": abs(manifold_root0 - e1_exact),
            "arms": arms,
        }

    payload = {
        "schema_version": "paper_iii_qse_regime_frontier_sweep_v1",
        "policy": "diagnostic_only_regime_frontier_sweep",
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "post_run_diagnostic_only": True,
        },
        "layout": {"L": 2, "n_ph_max": 3, "num_qubits": _NQ, "note": "fixed nph3 shared-pool sweep"},
        "reference": "exact (1,1)-sector ground state and first excited energy (dense ED)",
        "cost_weights_preset": "two_qubit_only_v1",
        "oracle_kind": ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
        "budget": int(budget),
        "pool_size": len(basis),
        "regimes": regimes_payload,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--budget", type=int, default=_BUDGET)
    args = parser.parse_args(argv)

    payload = run_sweep(alphas=args.alphas, budget=args.budget, output_json=args.output_json)
    for regime, record in payload["regimes"].items():
        print(
            f"\n== {regime} (u={record['u']}, g={record['g_ep']:.3f}) "
            f"gap={record['exact_gap']:.4f} manifold-limit err={record['manifold_limit_abs_err_E1']:.2e}"
        )
        for arm_name, arm in record["arms"].items():
            if "frontier" in arm:
                rows = arm["frontier"]
                picks = [row for row in rows if row["prefix_size"] in (20, len(rows))]
                text = "  ".join(
                    f"k={row['prefix_size']}: {row['abs_err_E1']:.2e}@{row['cum_2q']:.0f}2Q"
                    for row in picks
                    if row["abs_err_E1"] is not None
                )
                print(f"  {arm_name:<24} {text}")
            else:
                print(
                    f"  {arm_name:<24} complete({arm['class_size']}): "
                    f"{arm['abs_err_E1']:.2e}@{arm['total_2q']:.0f}2Q"
                )
    print(f"\noutput_json: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
