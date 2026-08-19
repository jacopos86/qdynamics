#!/usr/bin/env python3
"""Score-term ablations for the Paper III QSE acquisition rule.

Each arm disables one ingredient of the selection score and reruns the
full residual-stop selection, so support size is an output and a
redundant term shows up as "same accuracy, more records" rather than as
a change in a fixed-budget error. Arms:

- ``full``                -- production anchor;
- ``no_novelty_weight``   -- metric-novelty weight zero, hard floor kept;
- ``no_novelty_floor``    -- hard floor zero, weight kept;
- ``no_residual``         -- residual-capture weight zero;
- ``no_ritz``             -- Ritz-gain weight zero;
- ``no_condition``        -- conditioning penalty zero (the QSE analog of a
  trust region);
- ``no_cost_discount``    -- geometric score only, no hardware discount.

Reported per arm: support size at convergence, stop reason, compiled 2Q,
overlap condition estimate, and per-root errors over the lowest six
excitations against exact sector-restricted references.

Statevector diagnostics; never feeds controller decisions.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
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
    / "output/diagnostics/paper_iii_score_ablation_20260819_v1/score_ablation.json"
)
_Q0_POLICY = QSEBasisVectorPolicy(
    reference_projection="q0", basis_vector_normalization="raw_projected"
)
_TARGET_ROOTS = 6

_ARMS: dict[str, dict[str, Any]] = {
    "full": {},
    "no_novelty_weight": {"geometry_metric_novelty_weight": 0.0},
    "no_novelty_floor": {"geometry_min_metric_novelty": 0.0},
    "no_residual": {"geometry_residual_weight": 0.0},
    "no_ritz": {"geometry_ritz_weight": 0.0},
    "no_condition": {"geometry_condition_penalty_weight": 0.0},
    "no_cost_discount": {"geometry_cost_discount_alpha": None},
}

_RITZ_SWEEP: dict[str, dict[str, Any]] = {
    "ritz_w0.00": {"geometry_ritz_weight": 0.0},
    "ritz_w0.25": {},
    "ritz_w0.50": {"geometry_ritz_weight": 0.5},
    "ritz_w1.00": {"geometry_ritz_weight": 1.0},
    "ritz_w2.00": {"geometry_ritz_weight": 2.0},
    "ritz_w4.00": {"geometry_ritz_weight": 4.0},
}

# Minimality frontier: how few terms retain the accuracy/cost advantage.
_MINIMALITY: dict[str, dict[str, Any]] = {
    "production_3term": {},
    "no_transition": {"geometry_transition_weight": 0.0},
    "novelty_plus_transition": {"geometry_residual_weight": 0.0},
    "residual_plus_transition": {"geometry_metric_novelty_weight": 0.0},
    "residual_only": {
        "geometry_metric_novelty_weight": 0.0,
        "geometry_transition_weight": 0.0,
    },
    "novelty_only": {
        "geometry_residual_weight": 0.0,
        "geometry_transition_weight": 0.0,
    },
    "transition_only": {
        "geometry_metric_novelty_weight": 0.0,
        "geometry_residual_weight": 0.0,
    },
}

# Candidate production scores, compared head to head. "merged" replaces the
# residual-capture + Ritz-gain pair by the exact two-level gain alone, which
# contains residual capture as its small-eta limit and removes one weight.
_SCORE_VARIANTS: dict[str, dict[str, Any]] = {
    "current_four_term": {},
    "drop_condition": {"geometry_condition_penalty_weight": 0.0},
    "drop_ritz": {"geometry_ritz_weight": 0.0},
    "three_term": {
        "geometry_condition_penalty_weight": 0.0,
        "geometry_ritz_weight": 0.0,
    },
    "merged_window_gain": {
        "geometry_residual_weight": 0.0,
        "geometry_ritz_weight": 1.0,
    },
    "merged_no_condition": {
        "geometry_residual_weight": 0.0,
        "geometry_ritz_weight": 1.0,
        "geometry_condition_penalty_weight": 0.0,
    },
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--regimes", default=None)
    parser.add_argument("--residual-stop", type=float, default=1.0e-3)
    parser.add_argument("--mode", choices=("ablation", "ritz_sweep", "score_variants", "minimality"), default="ablation")
    args = parser.parse_args(argv)

    wanted = None if args.regimes is None else {t.strip() for t in str(args.regimes).split(",")}
    regimes_payload: dict[str, Any] = {}

    for regime, u, g_ep, n_ph_max in PAPER_I_REGIMES:
        if wanted is not None and regime not in wanted:
            continue
        nq = _num_qubits(n_ph_max)
        hamiltonian, basis, _meta = _build_regime_pool(u=u, g_ep=g_ep, n_ph_max=n_ph_max)
        dense = _dense_hamiltonian(hamiltonian, 1 << nq)
        ground, spectrum = _sector_spectrum(dense, count=_TARGET_ROOTS + 1)
        del dense
        references = spectrum[1 : _TARGET_ROOTS + 1]

        cost_rows = annotate_basis_with_compiled_costs(
            basis,
            num_qubits=nq,
            oracle_kind=ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
            cost_weights=resolve_cost_weights_preset("two_qubit_only_v1"),
        )
        costs = tuple(row.scalarized_canonical_cost for row in cost_rows)

        base = StaticRecordSelectionConfig(
            mode="geometry_selected",
            max_records=len(basis),
            geometry_target_roots=_TARGET_ROOTS,
            geometry_cost_discount_alpha=1.0,
            geometry_residual_stop=float(args.residual_stop),
        )

        print(f"\n== {regime} (u={u}, g={g_ep:.4f}, nph{n_ph_max}) pool={len(basis)}", flush=True)
        arms_payload: dict[str, Any] = {}
        arm_set = {"ablation": _ARMS, "ritz_sweep": _RITZ_SWEEP, "score_variants": _SCORE_VARIANTS, "minimality": _MINIMALITY}[args.mode]
        for arm_name, overrides in arm_set.items():
            config = replace(base, **overrides) if overrides else base
            selection = select_static_qse_records(
                basis,
                config=config,
                hamiltonian=hamiltonian,
                prepared_state=ground,
                basis_vector_policy=_Q0_POLICY,
                compiled_costs=costs,
            )
            indices = list(selection.selected_original_indices)
            result = compute_qse_spectra(
                hamiltonian,
                ground,
                tuple(basis[int(i)] for i in indices),
                basis_vector_policy=_Q0_POLICY,
            )
            energies = np.asarray(result.eigenvalues, dtype=float).reshape(-1)
            errors = [
                abs(float(energies[r]) - float(ref)) if r < energies.size else None
                for r, ref in enumerate(references)
            ]
            finite = [e for e in errors if e is not None]
            condition = result.overlap_condition_estimate
            stop = selection.geometry_stop or {}
            arms_payload[arm_name] = {
                "overrides": {k: v for k, v in overrides.items()},
                "support_size": len(indices),
                "retained_rank": int(result.retained_rank),
                "stop_reason": stop.get("stop_reason"),
                "total_2q": float(sum(costs[int(i)] for i in indices)),
                "overlap_condition_estimate": (
                    float(condition) if condition is not None else None
                ),
                "root_abs_errors": errors,
                "max_root_abs_error": max(finite) if finite else None,
                "selected_original_indices": [int(i) for i in indices],
            }
            print(
                f"  {arm_name:<20} k={len(indices):<4} @{arms_payload[arm_name]['total_2q']:.0f}2Q "
                f"stop={str(stop.get('stop_reason')):<18} "
                f"maxerr={arms_payload[arm_name]['max_root_abs_error']:.1e} "
                f"cond={arms_payload[arm_name]['overlap_condition_estimate']:.1e}",
                flush=True,
            )

        regimes_payload[regime] = {
            "u": float(u),
            "g_ep": float(g_ep),
            "n_ph_max": int(n_ph_max),
            "pool_size": len(basis),
            "reference_ground_energy": float(spectrum[0]),
            "reference_excitations": references,
            "arms": arms_payload,
        }

    payload = {
        "schema_version": "paper_iii_qse_score_ablation_v1",
        "mode": str(args.mode),
        "policy": "diagnostic_only_score_term_ablation",
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "post_run_diagnostic_only": True,
        },
        "target_roots": _TARGET_ROOTS,
        "residual_stop": float(args.residual_stop),
        "convention": "support size is an output; all arms use the same eps-stop",
        "regimes": regimes_payload,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\noutput_json: {args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
