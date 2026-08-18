#!/usr/bin/env python3
"""Exchange repair of stuck selection supports in the strong-phonon regimes.

The Paper-I-convention sweep showed low-budget greedy selection stalling at
weak_strong and intermediate_strong (nph7) while the complete linear-response
class reached 1e-4..1e-5 at ~412 2Q. This driver starts from the stalled
geometry alpha=1 support (budget 40) in each regime and applies certified
joint delete--add exchange maintenance in two modes:

- **dominance**: plain Pareto gate — compiled cost may never increase, so
  every accepted patch improves the target root at equal-or-lower 2Q;
- **budgeted(412)**: root-improving patches may spend compiled cost up to
  the complete linear-response class total, giving a cost-parity comparison
  against that fixed class.

References are exact sector-restricted eigenproblems; the target root is the
lowest orthogonal (q0) Ritz root. Diagnostic evidence driver; never feeds
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
    _sector_reference,
)
from pipelines.qse_spectra.compiled_costs import (
    ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
    annotate_basis_with_compiled_costs,
    resolve_cost_weights_preset,
)
from pipelines.qse_spectra.core import QSEBasisVectorPolicy
from pipelines.qse_spectra.exchange_maintenance import (
    QSEExchangeConfig,
    exchange_maintenance_payload,
    run_qse_exchange_maintenance,
)
from pipelines.qse_spectra.record_selection import (
    StaticRecordSelectionConfig,
    select_static_qse_records,
)

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "output/diagnostics/paper_iii_exchange_repair_20260818_v1/exchange_repair_summary.json"
)
_Q0_POLICY = QSEBasisVectorPolicy(
    reference_projection="q0", basis_vector_normalization="raw_projected"
)
_REPAIR_REGIMES = ("weak_strong", "intermediate_strong")
_BUDGET = 40


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budget", type=int, default=_BUDGET)
    parser.add_argument("--max-rounds", type=int, default=30)
    args = parser.parse_args(argv)

    regime_params = {name: (u, g, nph) for name, u, g, nph in PAPER_I_REGIMES}
    regimes_payload: dict[str, Any] = {}
    for regime in _REPAIR_REGIMES:
        u, g_ep, n_ph_max = regime_params[regime]
        nq = _num_qubits(n_ph_max)
        hamiltonian, basis, _pool_meta = _build_regime_pool(u=u, g_ep=g_ep, n_ph_max=n_ph_max)
        dense = _dense_hamiltonian(hamiltonian, 1 << nq)
        ground, _e0_exact, e1_exact = _sector_reference(dense)

        cost_rows = annotate_basis_with_compiled_costs(
            basis,
            num_qubits=nq,
            oracle_kind=ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
            cost_weights=resolve_cost_weights_preset("two_qubit_only_v1"),
        )
        costs = tuple(row.scalarized_canonical_cost for row in cost_rows)
        linear_total = float(
            sum(
                costs[index]
                for index, element in enumerate(basis)
                if _element_family(element.name) in _LINEAR_RESPONSE_FAMILIES
            )
        )

        selection = select_static_qse_records(
            basis,
            config=StaticRecordSelectionConfig(
                mode="geometry_selected",
                max_records=int(args.budget),
                geometry_cost_discount_alpha=1.0,
            ),
            hamiltonian=hamiltonian,
            prepared_state=ground,
            basis_vector_policy=_Q0_POLICY,
            compiled_costs=costs,
        )

        arms: dict[str, Any] = {}
        for arm_name, exchange_config in (
            ("dominance", QSEExchangeConfig(max_rounds=int(args.max_rounds))),
            (
                "budgeted_linear_class_parity",
                QSEExchangeConfig(max_rounds=int(args.max_rounds), cost_budget=linear_total),
            ),
        ):
            result = run_qse_exchange_maintenance(
                basis,
                selection.selected_original_indices,
                costs,
                hamiltonian=hamiltonian,
                prepared_state=ground,
                basis_vector_policy=_Q0_POLICY,
                config=exchange_config,
            )
            payload = exchange_maintenance_payload(result)
            initial, final = payload["initial"], payload["final"]
            arms[arm_name] = {
                "initial_abs_err_E1": abs(float(initial["root0_energy"]) - e1_exact),
                "initial_2q": float(initial["total_compiled_cost"]),
                "final_abs_err_E1": abs(float(final["root0_energy"]) - e1_exact),
                "final_2q": float(final["total_compiled_cost"]),
                "committed_patch_count": int(payload["committed_patch_count"]),
                "telemetry": payload,
            }
            row = arms[arm_name]
            print(
                f"{regime}/{arm_name}: {row['initial_abs_err_E1']:.2e}@{row['initial_2q']:.0f}2Q "
                f"-> {row['final_abs_err_E1']:.2e}@{row['final_2q']:.0f}2Q "
                f"({row['committed_patch_count']} patches)"
            )

        regimes_payload[regime] = {
            "u": float(u),
            "g_ep": float(g_ep),
            "n_ph_max": int(n_ph_max),
            "e1_exact_sector": float(e1_exact),
            "linear_response_class_total_2q": linear_total,
            "arms": arms,
        }

    payload = {
        "schema_version": "paper_iii_qse_exchange_repair_v1",
        "policy": "diagnostic_only_exchange_repair",
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "post_run_diagnostic_only": True,
        },
        "cost_weights_preset": "two_qubit_only_v1",
        "start_support": f"geometry_alpha1_budget{int(args.budget)}",
        "regimes": regimes_payload,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"output_json: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
