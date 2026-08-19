#!/usr/bin/env python3
"""Full-transpilation compiled resources for Paper III QSE supports.

The frontier and multi-root tables cost supports with the analytic
graph-span oracle under the ``two_qubit_only_v1`` scalarization. This
driver recomputes the same supports with the Qiskit backend-transpile
oracle and reports the full Paper-I component ledger --- two-qubit gates,
depth, one-qubit gates, and rotation-angle count --- so the resource
claims rest on realized compilation rather than an analytic proxy.

Supports are reconstructed from the committed multi-root evidence
(``selected_original_indices``). Reporting only; never feeds controller
decisions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.paper_iii_qse_paper_i_convention_sweep import (
    PAPER_I_REGIMES,
    _build_regime_pool,
    _num_qubits,
)
from pipelines.qse_spectra.compiled_costs import (
    ORACLE_KIND_BACKEND_TRANSPILE,
    ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
    annotate_basis_with_compiled_costs,
    resolve_cost_weights_preset,
)

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "output/diagnostics/paper_iii_transpile_costs_20260819_v1/transpile_costs.json"
)
MULTIROOT_JSON = (
    REPO_ROOT
    / "output/diagnostics/paper_iii_multiroot_sweep_20260818_v1/multiroot_sweep_epsstop.json"
)
_ARMS = ("fixed_linear_response_complete", "geometry_alpha1_R6", "exchange_dominance_R6")


def _ledger(rows: Sequence[Any], indices: Sequence[int]) -> dict[str, float]:
    chosen = [rows[int(i)] for i in indices]
    return {
        "c_hat_2q": float(sum(r.estimate.c_hat_2q for r in chosen)),
        "c_hat_d": float(sum(r.estimate.c_hat_d for r in chosen)),
        "c_hat_1q": float(sum(r.estimate.c_hat_1q for r in chosen)),
        "c_hat_theta": float(sum(r.estimate.c_hat_theta for r in chosen)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multiroot-json", type=Path, default=MULTIROOT_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--regimes", default=None)
    args = parser.parse_args(argv)

    multiroot = json.loads(Path(args.multiroot_json).read_text(encoding="utf-8"))
    wanted = None if args.regimes is None else {t.strip() for t in str(args.regimes).split(",")}
    weights = resolve_cost_weights_preset("two_qubit_only_v1")

    regimes_payload: dict[str, Any] = {}
    for regime, u, g_ep, n_ph_max in PAPER_I_REGIMES:
        if wanted is not None and regime not in wanted:
            continue
        record = multiroot["regimes"].get(regime)
        if record is None:
            continue
        missing = [a for a in _ARMS if "selected_original_indices" not in record["arms"].get(a, {})]
        if missing:
            raise SystemExit(
                f"multiroot evidence for {regime} lacks selected_original_indices for {missing}; "
                "rerun pipelines/exact_bench/paper_iii_qse_multiroot_sweep.py"
            )
        nq = _num_qubits(n_ph_max)
        _hamiltonian, basis, _meta = _build_regime_pool(u=u, g_ep=g_ep, n_ph_max=n_ph_max)

        graph_rows = annotate_basis_with_compiled_costs(
            basis, num_qubits=nq, oracle_kind=ORACLE_KIND_MARRAKESH_GRAPH_SPAN, cost_weights=weights
        )
        transpile_rows = annotate_basis_with_compiled_costs(
            basis, num_qubits=nq, oracle_kind=ORACLE_KIND_BACKEND_TRANSPILE, cost_weights=weights
        )

        arms_payload: dict[str, Any] = {}
        for arm_key in _ARMS:
            indices = record["arms"][arm_key]["selected_original_indices"]
            arms_payload[arm_key] = {
                "support_size": len(indices),
                "graph_span": _ledger(graph_rows, indices),
                "backend_transpile": _ledger(transpile_rows, indices),
            }
            g = arms_payload[arm_key]["graph_span"]["c_hat_2q"]
            t = arms_payload[arm_key]["backend_transpile"]["c_hat_2q"]
            arms_payload[arm_key]["transpile_over_graph_span_2q"] = (
                float(t / g) if g > 0 else None
            )

        regimes_payload[regime] = {
            "u": float(u),
            "g_ep": float(g_ep),
            "n_ph_max": int(n_ph_max),
            "num_qubits": nq,
            "arms": arms_payload,
        }
        print(f"\n== {regime} (nq={nq})", flush=True)
        for arm_key, arm in arms_payload.items():
            print(
                f"  {arm_key:<34} k={arm['support_size']:<4} "
                f"2Q graph={arm['graph_span']['c_hat_2q']:.0f} "
                f"transpile={arm['backend_transpile']['c_hat_2q']:.0f} "
                f"depth={arm['backend_transpile']['c_hat_d']:.0f} "
                f"1Q={arm['backend_transpile']['c_hat_1q']:.0f}",
                flush=True,
            )

    payload = {
        "schema_version": "paper_iii_qse_transpile_costs_v1",
        "policy": "diagnostic_only_compiled_resource_ledger",
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "post_run_diagnostic_only": True,
        },
        "oracles": {
            "analytic": ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
            "realized": ORACLE_KIND_BACKEND_TRANSPILE,
        },
        "uses_qiskit": True,
        "regimes": regimes_payload,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\noutput_json: {args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
