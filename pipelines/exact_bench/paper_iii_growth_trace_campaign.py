#!/usr/bin/env python3
"""Growth-trace campaign: the locked Paper III comparison method.

Implements section 1 of `agent_guidance/qse/paper-iii-comparison-protocol.md`.
Every method grows the subspace one record at a time to a fixed cap; at each
step k the tuple (k, N2q, D2q, Dc, max_{nu<=R}|Delta E_nu|) is recorded. The
comparison is read off the trace:

    C*(eps_E) = min{cost over trace : max|Delta E| <= eps_E}

so no method's own stopping rule enters the comparison (the Paper-I
fixed-iteration convention). Our residual stop is reported as a marker on the
trace, never as the comparison mechanism.

Exchange policies, both implemented, evaluated on the cheap tier before one
becomes canonical:

- ``crossing``  (policy A) -- certified exchange applied only to the prefix
  that first crosses each target. Matches what exchange is: post-hoc
  compression of a finished basis. Cheap.
- ``every_k``   (policy B) -- exchange applied at every recorded step,
  producing a fully exchanged trace. Expensive; may compress earlier.

Arms: ``ours`` (geometric acquisition order), ``cheapest_first``,
``input_order``, and ``fixed_class`` (complete, single point -- it is a class,
not a growth process).

Statevector diagnostics; never feeds controller decisions.
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

from pipelines.exact_bench.paper_iii_matched_accuracy_campaign import (
    ERROR_TARGET_LADDER,
    REGIME_SETS,
    STATUS_NOT_IN_POOL,
    STATUS_REACHED,
    STATUS_UNATTAINABLE,
)
from pipelines.exact_bench.paper_iii_qse_paper_i_convention_sweep import (
    _LINEAR_RESPONSE_FAMILIES,
    _element_family,
)
from pipelines.qse_spectra.core import QSEBasisVectorPolicy, compute_qse_spectra
from pipelines.qse_spectra.exchange_maintenance import (
    QSEExchangeConfig,
    run_qse_exchange_maintenance,
)
from pipelines.qse_spectra.paper_iii_problem import load_problem
from pipelines.qse_spectra.record_selection import (
    StaticRecordSelectionConfig,
    select_static_qse_records,
)

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "output/diagnostics/paper_iii_growth_trace_20260826_v1/growth_trace.json"
)
_Q0_POLICY = QSEBasisVectorPolicy(
    reference_projection="q0", basis_vector_normalization="raw_projected"
)
_TARGET_ROOTS = 6
RESIDUAL_MARKER = 1.0e-3


def _window_error(problem: Any, indices: Sequence[int]) -> tuple[float | None, int]:
    try:
        result = compute_qse_spectra(
            problem.hamiltonian,
            problem.ground,
            tuple(problem.basis[int(i)] for i in indices),
            basis_vector_policy=_Q0_POLICY,
        )
    except ValueError:
        return None, 0
    energies = np.asarray(result.eigenvalues, dtype=float).reshape(-1)
    refs = problem.references
    if energies.size < len(refs):
        return None, int(result.retained_rank)
    return (
        max(abs(float(energies[r]) - float(ref)) for r, ref in enumerate(refs)),
        int(result.retained_rank),
    )


def _exchange(problem: Any, indices: Sequence[int]) -> list[int]:
    result = run_qse_exchange_maintenance(
        problem.basis,
        list(indices),
        problem.costs,
        hamiltonian=problem.hamiltonian,
        prepared_state=problem.ground,
        basis_vector_policy=_Q0_POLICY,
        config=QSEExchangeConfig(
            max_rounds=30, target_root_count=_TARGET_ROOTS, insertion_shortlist_size=16
        ),
    )
    return [int(i) for i in result.final_indices]


def _row(problem: Any, indices: Sequence[int], k: int, *, exchanged: bool) -> dict[str, Any]:
    err, rank = _window_error(problem, indices)
    res = problem.resource_triple(indices)
    return {
        "k": int(k),
        "support_size": len(indices),
        "exchanged": bool(exchanged),
        "retained_rank": rank,
        "max_root_abs_error": err,
        **res,
    }


def trace_arm(
    problem: Any,
    order: Sequence[int],
    *,
    stride: int,
    k_max: int,
    exchange_policy: str,
) -> list[dict[str, Any]]:
    """Grow along ``order`` to ``k_max``, recording one row per stride step."""

    rows: list[dict[str, Any]] = []
    cap = min(int(k_max), len(order))
    for k in range(stride, cap + 1, stride):
        prefix = [int(i) for i in order[:k]]
        if exchange_policy == "every_k":
            prefix = _exchange(problem, prefix)
            rows.append(_row(problem, prefix, k, exchanged=True))
        else:
            rows.append(_row(problem, prefix, k, exchanged=False))
    return rows


def resolve_from_trace(
    rows: Sequence[dict[str, Any]], eps_e: float, *, extendable: bool
) -> dict[str, Any]:
    reaching = [
        r for r in rows
        if r.get("max_root_abs_error") is not None
        and float(r["max_root_abs_error"]) <= float(eps_e)
    ]
    if reaching:
        best = min(reaching, key=lambda r: float(r["n2q"]))
        return {"status": STATUS_REACHED, "crossing": dict(best)}
    finite = [r for r in rows if r.get("max_root_abs_error") is not None]
    terminal = min(finite, key=lambda r: float(r["max_root_abs_error"])) if finite else None
    return {
        "status": STATUS_NOT_IN_POOL if extendable else STATUS_UNATTAINABLE,
        "terminal": None if terminal is None else dict(terminal),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--regime-set", choices=sorted(REGIME_SETS), default="hubbard_l2")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=60)
    parser.add_argument(
        "--exchange-policy", choices=("crossing", "every_k", "both"), default="every_k"
    )
    args = parser.parse_args(argv)

    policies = (
        ("crossing", "every_k") if args.exchange_policy == "both"
        else (args.exchange_policy,)
    )
    regimes_payload: dict[str, Any] = {}

    for regime, u, g_ep, n_ph_max in REGIME_SETS[str(args.regime_set)]:
        problem = load_problem(
            regime=regime, u=u, g_ep=g_ep, n_ph_max=n_ph_max, target_roots=_TARGET_ROOTS
        )
        print(f"\n== {regime} (pool={len(problem.basis)})", flush=True)

        selection = select_static_qse_records(
            problem.basis,
            config=StaticRecordSelectionConfig(
                mode="geometry_selected",
                max_records=len(problem.basis),
                geometry_target_roots=_TARGET_ROOTS,
                geometry_cost_discount_alpha=1.0,
            ),
            hamiltonian=problem.hamiltonian,
            prepared_state=problem.ground,
            basis_vector_policy=_Q0_POLICY,
            compiled_costs=problem.costs,
        )
        our_order = [int(i) for i in selection.selected_original_indices]

        marker = select_static_qse_records(
            problem.basis,
            config=StaticRecordSelectionConfig(
                mode="geometry_selected",
                max_records=len(problem.basis),
                geometry_target_roots=_TARGET_ROOTS,
                geometry_cost_discount_alpha=1.0,
                geometry_residual_stop=RESIDUAL_MARKER,
            ),
            hamiltonian=problem.hamiltonian,
            prepared_state=problem.ground,
            basis_vector_policy=_Q0_POLICY,
            compiled_costs=problem.costs,
        )
        marker_k = len(marker.selected_original_indices)

        cheapest = sorted(range(len(problem.basis)),
                          key=lambda i: (float(problem.costs[i]), i))
        fixed_indices = [
            i for i, e in enumerate(problem.basis)
            if _element_family(e.name) in _LINEAR_RESPONSE_FAMILIES
        ]

        arms: dict[str, Any] = {}
        for policy in policies:
            for arm, order in (
                ("ours", our_order),
                ("cheapest_first", cheapest),
                ("input_order", list(range(len(problem.basis)))),
            ):
                # exchange is our maintenance step; benchmarks are not exchanged
                pol = policy if arm == "ours" else "none"
                key = f"{arm}__{policy}" if arm == "ours" else arm
                if key in arms:
                    continue
                rows = trace_arm(
                    problem, order,
                    stride=int(args.stride), k_max=int(args.k_max), exchange_policy=pol,
                )
                arms[key] = {"trace": rows}
                best = min((r for r in rows if r["max_root_abs_error"] is not None),
                           key=lambda r: r["max_root_abs_error"], default=None)
                if best:
                    print(
                        f"   {key:<22} terminal {best['max_root_abs_error']:.1e} "
                        f"@k={best['k']} N2q={best['n2q']:.0f}", flush=True
                    )

        f_err, f_rank = _window_error(problem, fixed_indices)
        arms["fixed_class"] = {
            "trace": [
                {
                    "k": len(fixed_indices), "support_size": len(fixed_indices),
                    "exchanged": False, "retained_rank": f_rank,
                    "max_root_abs_error": f_err,
                    **problem.resource_triple(fixed_indices),
                }
            ]
        }

        cells: dict[str, Any] = {}
        for eps_e in ERROR_TARGET_LADDER:
            key = f"{eps_e:.0e}"
            cells[key] = {}
            for arm_key, arm in arms.items():
                cells[key][arm_key] = resolve_from_trace(
                    arm["trace"], eps_e, extendable=(arm_key != "fixed_class")
                )
        # Policy A: exchange only the prefix that first crosses each target.
        for eps_e in ERROR_TARGET_LADDER:
            key = f"{eps_e:.0e}"
            cell = cells[key].get("ours__crossing")
            if cell and cell.get("status") == STATUS_REACHED:
                k = int(cell["crossing"]["k"])
                compressed = _exchange(problem, our_order[:k])
                cell["crossing_exchanged"] = _row(problem, compressed, k, exchanged=True)

        # Estimator work (axis 4) at every crossing point -- protocol section 3
        # mandates all four axes in a reported cell.
        supports = {
            "ours__crossing": our_order,
            "ours__every_k": our_order,
            "cheapest_first": cheapest,
            "input_order": list(range(len(problem.basis))),
            "fixed_class": fixed_indices,
        }
        for eps_e in ERROR_TARGET_LADDER:
            key = f"{eps_e:.0e}"
            for arm_key, cell in cells[key].items():
                if cell.get("status") != STATUS_REACHED:
                    continue
                order_src = supports.get(arm_key)
                if order_src is None:
                    continue
                k = int(cell["crossing"]["k"])
                idx = (
                    list(order_src) if arm_key == "fixed_class"
                    else list(order_src[:k])
                )
                if arm_key == "ours__every_k":
                    idx = _exchange(problem, idx)
                cell["crossing"]["estimator"] = problem.estimator_cost(idx)

        regimes_payload[regime] = {
            "u": float(u), "g_ep": float(g_ep), "n_ph_max": int(n_ph_max),
            "pool_size": len(problem.basis),
            "shared_problem_receipt": problem.arm_receipt(),
            "reference_ground_energy": problem.ground_energy,
            "reference_excitations": list(problem.references),
            "residual_stop_marker": {"eps_res": RESIDUAL_MARKER, "k": marker_k},
            "fixed_class_size": len(fixed_indices),
            "arms": arms,
            "cells": cells,
        }

    payload = {
        "schema_version": "paper_iii_growth_trace_campaign_v1",
        "protocol": "agent_guidance/qse/paper-iii-comparison-protocol.md#1",
        "policy": "diagnostic_only_growth_trace_campaign",
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "post_run_diagnostic_only": True,
        },
        "convention": (
            "every method grows one record at a time to a fixed cap; C*(eps_E) is read "
            "off the trace, so no method's stopping rule enters the comparison"
        ),
        "regime_set": str(args.regime_set),
        "k_max": int(args.k_max),
        "stride": int(args.stride),
        "exchange_policies": list(policies),
        "error_target_ladder": [float(x) for x in ERROR_TARGET_LADDER],
        "target_roots": _TARGET_ROOTS,
        "regimes": regimes_payload,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\noutput_json: {args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
