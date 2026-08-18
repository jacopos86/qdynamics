#!/usr/bin/env python3
"""Multi-root Paper III sweep: six Paper I regimes, lowest six excitations.

Extends the Paper-I-convention sweep to the full low excitation window.
References are the exact lowest six sector excitation energies (E1..E6 of
the (1,1) sector). Geometry selection scores residual capture against the
lowest ``geometry_target_roots=6`` Ritz states of the growing subspace, and
exchange runs under the Ky Fan trace objective (``target_root_count=6``).
Reported per regime and arm: absolute error of every tracked root at the
budget-40 endpoint, plus the compiled-2Q cost. Arms:

- input_order (baseline ordering),
- fixed linear-response class (complete, deterministic),
- geometry alpha=1 (multi-root residual scoring),
- geometry alpha=1 + exchange (dominance, R=6),
- geometry alpha=1 + exchange (budgeted at fixed-class cost parity, R=6).

Statevector diagnostics at the Paper I conventions (nph3 weak-phonon /
nph7 strong-phonon, per-regime full_meta macro pools); never feeds
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

from pipelines.exact_bench.paper_iii_qse_paper_i_convention_sweep import (
    PAPER_I_REGIMES,
    _LINEAR_RESPONSE_FAMILIES,
    _build_regime_pool,
    _element_family,
    _num_qubits,
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
from pipelines.qse_spectra.exchange_maintenance import (
    QSEExchangeConfig,
    run_qse_exchange_maintenance,
)
from pipelines.qse_spectra.record_selection import (
    StaticRecordSelectionConfig,
    select_static_qse_records,
)

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "output/diagnostics/paper_iii_multiroot_sweep_20260818_v1/multiroot_sweep.json"
)
_Q0_POLICY = QSEBasisVectorPolicy(
    reference_projection="q0", basis_vector_normalization="raw_projected"
)
_TARGET_ROOTS = 6
_BUDGET = 40


def _root_errors(
    pool: Sequence[Any],
    indices: Sequence[int],
    references: Sequence[float],
    *,
    hamiltonian: Any,
    ground: np.ndarray,
) -> tuple[list[float | None], int]:
    result = compute_qse_spectra(
        hamiltonian,
        ground,
        tuple(pool[int(index)] for index in indices),
        basis_vector_policy=_Q0_POLICY,
    )
    energies = np.asarray(result.eigenvalues, dtype=float).reshape(-1)
    errors: list[float | None] = []
    for root, reference in enumerate(references):
        if root < energies.size:
            errors.append(abs(float(energies[root]) - float(reference)))
        else:
            errors.append(None)
    return errors, int(result.retained_rank)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budget", type=int, default=_BUDGET)
    parser.add_argument("--target-roots", type=int, default=_TARGET_ROOTS)
    parser.add_argument("--max-rounds", type=int, default=30)
    parser.add_argument(
        "--residual-stop",
        type=float,
        default=None,
        help="Residual-norm stopping tolerance; when set, budget becomes a safety cap only.",
    )
    args = parser.parse_args(argv)
    target_roots = int(args.target_roots)

    regimes_payload: dict[str, Any] = {}
    for regime, u, g_ep, n_ph_max in PAPER_I_REGIMES:
        nq = _num_qubits(n_ph_max)
        hamiltonian, basis, _meta = _build_regime_pool(u=u, g_ep=g_ep, n_ph_max=n_ph_max)
        dense = _dense_hamiltonian(hamiltonian, 1 << nq)
        ground, spectrum = _sector_spectrum(dense, count=target_roots + 1)
        references = spectrum[1 : target_roots + 1]

        cost_rows = annotate_basis_with_compiled_costs(
            basis,
            num_qubits=nq,
            oracle_kind=ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
            cost_weights=resolve_cost_weights_preset("two_qubit_only_v1"),
        )
        costs = tuple(row.scalarized_canonical_cost for row in cost_rows)
        linear_indices = [
            index
            for index, element in enumerate(basis)
            if _element_family(element.name) in _LINEAR_RESPONSE_FAMILIES
        ]
        linear_total = float(sum(costs[index] for index in linear_indices))

        arms: dict[str, Any] = {}

        def _record_arm(arm_name: str, indices: Sequence[int]) -> None:
            errors, rank = _root_errors(
                basis, indices, references, hamiltonian=hamiltonian, ground=ground
            )
            arms[arm_name] = {
                "support_size": len(indices),
                "retained_rank": rank,
                "total_2q": float(sum(costs[int(index)] for index in indices)),
                "root_abs_errors": errors,
                "max_root_abs_error": max((e for e in errors if e is not None), default=None),
            }

        _record_arm("input_order", list(range(min(int(args.budget), len(basis)))))
        _record_arm("fixed_linear_response_complete", linear_indices)

        safety_cap = len(basis) if args.residual_stop is not None else int(args.budget)
        selection = select_static_qse_records(
            basis,
            config=StaticRecordSelectionConfig(
                mode="geometry_selected",
                max_records=safety_cap,
                geometry_target_roots=target_roots,
                geometry_cost_discount_alpha=1.0,
                geometry_residual_stop=(
                    float(args.residual_stop) if args.residual_stop is not None else None
                ),
            ),
            hamiltonian=hamiltonian,
            prepared_state=ground,
            basis_vector_policy=_Q0_POLICY,
            compiled_costs=costs,
        )
        _record_arm("geometry_alpha1_R6", selection.selected_original_indices)
        if selection.geometry_stop is not None:
            arms["geometry_alpha1_R6"]["geometry_stop"] = dict(selection.geometry_stop)

        for arm_name, exchange_config in (
            (
                "exchange_dominance_R6",
                QSEExchangeConfig(max_rounds=int(args.max_rounds), target_root_count=target_roots, insertion_shortlist_size=16),
            ),
            (
                "exchange_budgeted_R6",
                QSEExchangeConfig(
                    max_rounds=int(args.max_rounds),
                    target_root_count=target_roots,
                    cost_budget=linear_total,
                    insertion_shortlist_size=16,
                ),
            ),
        ):
            exchange = run_qse_exchange_maintenance(
                basis,
                selection.selected_original_indices,
                costs,
                hamiltonian=hamiltonian,
                prepared_state=ground,
                basis_vector_policy=_Q0_POLICY,
                config=exchange_config,
            )
            _record_arm(arm_name, exchange.final_indices)
            arms[arm_name]["committed_patch_count"] = sum(
                1 for round_record in exchange.rounds if round_record["committed_patch"] is not None
            )

        regimes_payload[regime] = {
            "u": float(u),
            "g_ep": float(g_ep),
            "n_ph_max": int(n_ph_max),
            "pool_size": len(basis),
            "reference_excitations": references,
            "linear_response_class_total_2q": linear_total,
            "arms": arms,
        }
        print(f"\n== {regime} (u={u}, g={g_ep:.4f}, nph{n_ph_max})")
        for arm_name, arm in arms.items():
            errors = " ".join(
                f"{error:.1e}" if error is not None else "--" for error in arm["root_abs_errors"]
            )
            patches = (
                f" ({arm['committed_patch_count']}p)" if "committed_patch_count" in arm else ""
            )
            print(f"  {arm_name:<30} @{arm['total_2q']:.0f}2Q{patches}  roots: {errors}")

    payload = {
        "schema_version": "paper_iii_qse_multiroot_sweep_v1",
        "policy": "diagnostic_only_multiroot_sweep",
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "post_run_diagnostic_only": True,
        },
        "target_roots": target_roots,
        "budget": int(args.budget),
        "cost_weights_preset": "two_qubit_only_v1",
        "objective": "ky_fan_trace_lowest_R_ritz_roots",
        "regimes": regimes_payload,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\noutput_json: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
