#!/usr/bin/env python3
"""Paper III frontier sweep at the Paper I regime conventions.

Runs the cost-selected QSE arms over the canonical Paper I six-regime HH
dimer matrix, recovered from the stationary-core run artifacts:

    weak_weak            u=0.25  g=0.353553390593  nph=3  (8 qubits)
    intermediate_weak    u=1.25  g=0.353553390593  nph=3
    strong_weak_u8       u=8.00  g=0.353553390593  nph=3
    weak_strong          u=0.25  g=0.790569415042  nph=7  (10 qubits)
    intermediate_strong  u=1.25  g=0.790569415042  nph=7
    strong_strong_u8     u=8.00  g=0.790569415042  nph=7

with t=1, omega0=1, dv=0, binary boson encoding, blocked ordering, open
boundary — i.e. nph3 for the weak-phonon sector and nph7 for the
strong-phonon sector, matching Paper I. The full_meta macro pool is built
per regime with the production builder (paop_r=1, no Pauli splitting,
prune_eps=0, normalization none), so pool contents and coefficients follow
each regime's Hamiltonian. References are exact sector-restricted
eigenproblems of the encoded Hamiltonian in the (1,1) fermion sector.
Compiled costs are Marrakesh graph-span 2Q under ``two_qubit_only_v1``
(the embedding table covers both 8- and 10-qubit layouts). Macro records
conserve fermion number, so no sector projection is needed in the pencil.
Diagnostic evidence driver; never feeds controller decisions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.paper_iii_qse_regime_frontier_sweep import (
    _dense_hamiltonian,
    _sector_reference,
)
from pipelines.qse_spectra.compiled_costs import (
    ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
    annotate_basis_with_compiled_costs,
    build_accuracy_cost_frontier,
    resolve_cost_weights_preset,
)
from pipelines.qse_spectra.core import (
    QSEBasisVectorPolicy,
    compute_qse_spectra,
    pauli_string_basis_element,
    polynomial_basis_element,
)
from pipelines.qse_spectra.record_selection import (
    StaticRecordSelectionConfig,
    select_static_qse_records,
)
from pipelines.qse_spectra.static_adapt_adapter import (
    build_artifact_problem_hamiltonian,
    build_hh_full_meta_pool_for_qse,
)

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "output/diagnostics/paper_iii_paper_i_convention_sweep_20260818_v1/paper_i_convention_sweep.json"
)

_G_WEAK = 0.353553390593
_G_STRONG = 0.790569415042
PAPER_I_REGIMES: tuple[tuple[str, float, float, int], ...] = (
    ("weak_weak", 0.25, _G_WEAK, 3),
    ("intermediate_weak", 1.25, _G_WEAK, 3),
    ("strong_weak_u8", 8.0, _G_WEAK, 3),
    ("weak_strong", 0.25, _G_STRONG, 7),
    ("intermediate_strong", 1.25, _G_STRONG, 7),
    ("strong_strong_u8", 8.0, _G_STRONG, 7),
)
_Q0_POLICY = QSEBasisVectorPolicy(
    reference_projection="q0", basis_vector_normalization="raw_projected"
)
_LINEAR_RESPONSE_FAMILIES = ("identity", "uccsd_ferm_lifted", "hh_phonon")
_BUDGET = 40


def _num_qubits(n_ph_max: int) -> int:
    qubits_per_boson = max(1, math.ceil(math.log2(int(n_ph_max) + 1)))
    return 4 + 2 * qubits_per_boson


def _element_family(name: str) -> str:
    return str(name).split("(")[0].split("::")[0]


def _build_regime_pool(
    *, u: float, g_ep: float, n_ph_max: int
) -> tuple[Any, list[Any], dict[str, Any]]:
    physics = dict(
        num_sites=2,
        t=1.0,
        u=float(u),
        omega0=1.0,
        g_ep=float(g_ep),
        dv=0.0,
        n_ph_max=int(n_ph_max),
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
    )
    hamiltonian = build_artifact_problem_hamiltonian(
        problem_key="hh", include_zero_point=True, **physics
    )
    terms, pool_meta = build_hh_full_meta_pool_for_qse(
        h_poly=hamiltonian,
        paop_r=1,
        paop_split_paulis=False,
        paop_prune_eps=0.0,
        paop_normalization="none",
        num_particles=(1, 1),
        **physics,
    )
    nq = _num_qubits(n_ph_max)
    basis = [pauli_string_basis_element("e" * nq, nq=nq, name="identity")]
    for term in terms:
        basis.append(polynomial_basis_element(term.polynomial, name=str(term.label)))
    return hamiltonian, basis, dict(pool_meta or {})


def _frontier_rows(frontier: dict[str, Any]) -> list[dict[str, Any]]:
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budget", type=int, default=_BUDGET)
    args = parser.parse_args(argv)

    regimes_payload: dict[str, Any] = {}
    for regime, u, g_ep, n_ph_max in PAPER_I_REGIMES:
        nq = _num_qubits(n_ph_max)
        hamiltonian, basis, pool_meta = _build_regime_pool(u=u, g_ep=g_ep, n_ph_max=n_ph_max)
        dense = _dense_hamiltonian(hamiltonian, 1 << nq)
        ground, e0_exact, e1_exact = _sector_reference(dense)

        cost_rows = annotate_basis_with_compiled_costs(
            basis,
            num_qubits=nq,
            oracle_kind=ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
            cost_weights=resolve_cost_weights_preset("two_qubit_only_v1"),
        )
        costs = tuple(row.scalarized_canonical_cost for row in cost_rows)

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
            arms[arm_name] = {"frontier": _frontier_rows(frontier)}

        budget = int(args.budget)
        _run_arm("input_order", StaticRecordSelectionConfig(mode="input_order", max_records=budget))
        geometry_kwargs = dict(
            hamiltonian=hamiltonian,
            prepared_state=ground,
            basis_vector_policy=_Q0_POLICY,
            compiled_costs=costs,
        )
        _run_arm(
            "geometry_subtractive",
            StaticRecordSelectionConfig(mode="geometry_selected", max_records=budget),
            **geometry_kwargs,
        )
        _run_arm(
            "geometry_alpha1",
            StaticRecordSelectionConfig(
                mode="geometry_selected", max_records=budget, geometry_cost_discount_alpha=1.0
            ),
            **geometry_kwargs,
        )

        linear_indices = [
            index
            for index, element in enumerate(basis)
            if _element_family(element.name) in _LINEAR_RESPONSE_FAMILIES
        ]
        lr_result = compute_qse_spectra(
            hamiltonian,
            ground,
            tuple(basis[index] for index in linear_indices),
            basis_vector_policy=_Q0_POLICY,
        )
        lr_root0 = float(np.asarray(lr_result.eigenvalues, dtype=float).reshape(-1)[0])
        arms["fixed_linear_response_complete"] = {
            "class_size": len(linear_indices),
            "total_2q": float(sum(costs[index] for index in linear_indices)),
            "abs_err_E1": abs(lr_root0 - e1_exact),
        }

        regimes_payload[regime] = {
            "u": float(u),
            "g_ep": float(g_ep),
            "n_ph_max": int(n_ph_max),
            "num_qubits": nq,
            "pool_size": len(basis),
            "e0_exact_sector": e0_exact,
            "e1_exact_sector": e1_exact,
            "exact_gap": e1_exact - e0_exact,
            "manifold_limit_abs_err_E1": abs(manifold_root0 - e1_exact),
            "arms": arms,
        }
        record = regimes_payload[regime]
        print(
            f"\n== {regime} (u={u}, g={g_ep:.4f}, nph{n_ph_max}, {nq}q, pool {len(basis)}) "
            f"gap={record['exact_gap']:.4f} manifold err={record['manifold_limit_abs_err_E1']:.2e}"
        )
        for arm_name, arm in arms.items():
            if "frontier" in arm:
                rows = arm["frontier"]
                picks = [row for row in rows if row["prefix_size"] in (20, len(rows))]
                text = "  ".join(
                    f"k={row['prefix_size']}: {row['abs_err_E1']:.2e}@{row['cum_2q']:.0f}2Q"
                    for row in picks
                    if row["abs_err_E1"] is not None
                )
                print(f"  {arm_name:<26} {text}")
            else:
                print(
                    f"  {arm_name:<26} complete({arm['class_size']}): "
                    f"{arm['abs_err_E1']:.2e}@{arm['total_2q']:.0f}2Q"
                )

    payload = {
        "schema_version": "paper_iii_qse_paper_i_convention_sweep_v1",
        "policy": "diagnostic_only_paper_i_convention_sweep",
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "post_run_diagnostic_only": True,
        },
        "conventions": {
            "regime_source": "paper_i_stationary_core_v7_run_artifacts",
            "weak_phonon_nph": 3,
            "strong_phonon_nph": 7,
            "granularity": "macro_records",
            "pool_builder": "full_meta paop_r=1 split=False prune_eps=0 norm=none",
        },
        "cost_weights_preset": "two_qubit_only_v1",
        "budget": int(args.budget),
        "regimes": regimes_payload,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\noutput_json: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
