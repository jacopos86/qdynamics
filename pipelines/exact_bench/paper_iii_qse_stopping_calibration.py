#!/usr/bin/env python3
"""Calibration of the residual-norm stopping rule for QSE selection.

Sweeps the ``geometry_residual_stop`` tolerance over the six Paper-I
Hubbard--Holstein regimes (nph3/nph7 conventions) and the three
Peierls--Hubbard coupling points. For each (case, epsilon) the greedy runs
with the residual-norm convergence rule (budget demoted to a large safety
cap) and records: stop reason, final max target-root residual norm,
support size, compiled 2Q cost, and the achieved per-root errors against
exact sector-restricted references. The paper-facing product is the
calibration relation between the declared tolerance, the internal residual
norm (measurable without exact references), and the realized accuracy —
replacing any arbitrary fixed support cap. Statevector diagnostics; never
feeds controller decisions.
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

from pipelines.exact_bench.paper_iii_qse_paper_i_convention_sweep import (
    PAPER_I_REGIMES,
    _build_regime_pool,
    _num_qubits,
)
from pipelines.exact_bench.paper_iii_qse_peierls_pilot import (
    PEIERLS_REGIMES,
    build_peierls_hamiltonian,
    build_peierls_pool,
)
from pipelines.exact_bench.paper_iii_qse_peierls_pilot import (
    _NQ as PEIERLS_NQ,
)
from pipelines.exact_bench.paper_iii_qse_peierls_pilot import (
    _sector_spectrum as peierls_sector_spectrum,
)
from pipelines.exact_bench.paper_iii_qse_regime_frontier_sweep import (
    _dense_hamiltonian,
    _sector_spectrum,
)
from pipelines.qse_spectra.compiled_costs import (
    ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
    annotate_basis_with_compiled_costs,
    resolve_cost_weights_preset,
)
from pipelines.qse_spectra.core import QSEBasisVectorPolicy, compute_qse_spectra
from pipelines.qse_spectra.record_selection import (
    StaticRecordSelectionConfig,
    select_static_qse_records,
)

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "output/diagnostics/paper_iii_stopping_calibration_20260819_v1/stopping_calibration.json"
)
_Q0_POLICY = QSEBasisVectorPolicy(
    reference_projection="q0", basis_vector_normalization="raw_projected"
)
_TARGET_ROOTS = 6
_EPSILONS = (1.0e-2, 1.0e-3, 1.0e-4, 1.0e-6, 1.0e-8)


def _cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for regime, u, g_ep, n_ph_max in PAPER_I_REGIMES:
        cases.append(
            {"case": regime, "family": "hh", "u": u, "g_ep": g_ep, "n_ph_max": n_ph_max}
        )
    for regime, u, g_ep in PEIERLS_REGIMES:
        cases.append({"case": regime, "family": "peierls", "u": u, "g_ep": g_ep})
    return cases


def _prepare(case: dict[str, Any]) -> dict[str, Any]:
    if case["family"] == "hh":
        nq = _num_qubits(case["n_ph_max"])
        hamiltonian, basis, _meta = _build_regime_pool(
            u=case["u"], g_ep=case["g_ep"], n_ph_max=case["n_ph_max"]
        )
        dense = _dense_hamiltonian(hamiltonian, 1 << nq)
        ground, spectrum = _sector_spectrum(dense, count=_TARGET_ROOTS + 1)
    else:
        nq = PEIERLS_NQ
        hamiltonian = build_peierls_hamiltonian(u=case["u"], g_ep=case["g_ep"])
        basis = build_peierls_pool()
        dense = _dense_hamiltonian(hamiltonian, 1 << nq)
        ground, spectrum = peierls_sector_spectrum(dense, count=_TARGET_ROOTS + 1)
    cost_rows = annotate_basis_with_compiled_costs(
        basis,
        num_qubits=nq,
        oracle_kind=ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
        cost_weights=resolve_cost_weights_preset("two_qubit_only_v1"),
    )
    return {
        "nq": nq,
        "hamiltonian": hamiltonian,
        "basis": basis,
        "ground": ground,
        "references": spectrum[1 : _TARGET_ROOTS + 1],
        "costs": tuple(row.scalarized_canonical_cost for row in cost_rows),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epsilons", type=float, nargs="+", default=list(_EPSILONS))
    args = parser.parse_args(argv)

    payload_cases: dict[str, Any] = {}
    for case in _cases():
        prepared = _prepare(case)
        rows = []
        for epsilon in args.epsilons:
            selection = select_static_qse_records(
                prepared["basis"],
                config=StaticRecordSelectionConfig(
                    mode="geometry_selected",
                    max_records=len(prepared["basis"]),
                    geometry_target_roots=_TARGET_ROOTS,
                    geometry_cost_discount_alpha=1.0,
                    geometry_residual_stop=float(epsilon),
                ),
                hamiltonian=prepared["hamiltonian"],
                prepared_state=prepared["ground"],
                basis_vector_policy=_Q0_POLICY,
                compiled_costs=prepared["costs"],
            )
            result = compute_qse_spectra(
                prepared["hamiltonian"],
                prepared["ground"],
                selection.selected_basis_elements,
                basis_vector_policy=_Q0_POLICY,
            )
            energies = np.asarray(result.eigenvalues, dtype=float).reshape(-1)
            errors = [
                abs(float(energies[root]) - reference) if root < energies.size else None
                for root, reference in enumerate(prepared["references"])
            ]
            finite = [error for error in errors if error is not None]
            stop = dict(selection.geometry_stop or {})
            rows.append(
                {
                    "epsilon": float(epsilon),
                    "stop_reason": stop.get("stop_reason"),
                    "final_max_target_residual_norm": stop.get(
                        "final_max_target_residual_norm"
                    ),
                    "support_size": len(selection.selected_original_indices),
                    "total_2q": float(
                        sum(
                            prepared["costs"][int(index)]
                            for index in selection.selected_original_indices
                        )
                    ),
                    "max_root_abs_error": max(finite) if finite else None,
                    "root_abs_errors": errors,
                }
            )
        payload_cases[case["case"]] = {**{k: v for k, v in case.items() if k != "case"}, "rows": rows}
        print(f"\n== {case['case']} ({case['family']})")
        for row in rows:
            err = row["max_root_abs_error"]
            resid = row["final_max_target_residual_norm"]
            print(
                f"  eps={row['epsilon']:.0e}  {row['stop_reason']:<19} "
                f"k={row['support_size']:3d} @{row['total_2q']:.0f}2Q  "
                f"residual={resid:.1e}  max_err={err:.1e}"
                if err is not None and resid is not None
                else f"  eps={row['epsilon']:.0e}  {row['stop_reason']} (incomplete)"
            )

    payload = {
        "schema_version": "paper_iii_qse_stopping_calibration_v1",
        "policy": "diagnostic_only_stopping_calibration",
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "post_run_diagnostic_only": True,
        },
        "target_roots": _TARGET_ROOTS,
        "epsilons": [float(value) for value in args.epsilons],
        "cost_weights_preset": "two_qubit_only_v1",
        "stopping_rule": "max_target_root_ritz_residual_norm_below_epsilon",
        "cases": payload_cases,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\noutput_json: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
