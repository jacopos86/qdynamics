#!/usr/bin/env python3
"""Matched-alphabet accuracy/cost frontiers for Paper III.

Every arm draws from the identical record pool, so the comparison is of
acquisition orderings rather than of alphabets. Each arm produces an
ordering of the pool; the frontier walks that ordering and records, at
every prefix, the support size, the cumulative compiled two-qubit cost,
and the maximum error over the lowest six excitations.

Arms (identical pool, differing only in the order records are bought):

- ``ours``            -- production rule: metric novelty + residual capture,
  cost-discounted;
- ``fixed_class_first`` -- the physically standard linear-response records
  first, then the remainder in pool order. At the prefix equal to the class
  size this arm *is* the complete fixed linear-response class, which makes
  that method comparable on the shared alphabet instead of on its own;
- ``cheapest_first``  -- ascending compiled cost;
- ``input_order``     -- pool order, the naive control.

A single trace supports every comparison protocol: fix an accuracy target
and read off the cost each arm needs, fix a cost budget and read off the
accuracy each arm reaches, or fix the iteration and read off both.

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
from pipelines.qse_spectra.record_selection import (
    StaticRecordSelectionConfig,
    select_static_qse_records,
)

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "output/diagnostics/paper_iii_matched_alphabet_20260819_v1/matched_alphabet_frontier.json"
)
_Q0_POLICY = QSEBasisVectorPolicy(
    reference_projection="q0", basis_vector_normalization="raw_projected"
)
_TARGET_ROOTS = 6


def _trace(
    basis: Sequence[Any],
    order: Sequence[int],
    costs: Sequence[float],
    references: Sequence[float],
    *,
    hamiltonian: Any,
    ground: np.ndarray,
    stride: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cum = 0.0
    for position, index in enumerate(order, start=1):
        cum += float(costs[int(index)])
        if position % stride and position != len(order):
            continue
        prefix = [int(i) for i in order[:position]]
        result = compute_qse_spectra(
            hamiltonian,
            ground,
            tuple(basis[i] for i in prefix),
            basis_vector_policy=_Q0_POLICY,
        )
        energies = np.asarray(result.eigenvalues, dtype=float).reshape(-1)
        errors = [
            abs(float(energies[r]) - float(ref)) if r < energies.size else None
            for r, ref in enumerate(references)
        ]
        finite = [e for e in errors if e is not None]
        rows.append(
            {
                "k": position,
                "cum_2q": float(cum),
                "retained_rank": int(result.retained_rank),
                "max_root_abs_error": max(finite) if finite else None,
                "roots_resolved": len(finite),
            }
        )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--regimes", default=None)
    parser.add_argument("--stride", type=int, default=2)
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

        selection = select_static_qse_records(
            basis,
            config=StaticRecordSelectionConfig(
                mode="geometry_selected",
                max_records=len(basis),
                geometry_target_roots=_TARGET_ROOTS,
                geometry_cost_discount_alpha=1.0,
            ),
            hamiltonian=hamiltonian,
            prepared_state=ground,
            basis_vector_policy=_Q0_POLICY,
            compiled_costs=costs,
        )
        ours = [int(i) for i in selection.selected_original_indices]

        fixed = [
            i for i, e in enumerate(basis)
            if _element_family(e.name) in _LINEAR_RESPONSE_FAMILIES
        ]
        fixed_first = fixed + [i for i in range(len(basis)) if i not in set(fixed)]
        cheapest = sorted(range(len(basis)), key=lambda i: (float(costs[i]), i))
        orders = {
            "ours": ours,
            "fixed_class_first": fixed_first,
            "cheapest_first": cheapest,
            "input_order": list(range(len(basis))),
        }

        print(f"\n== {regime} (pool={len(basis)}, fixed class={len(fixed)})", flush=True)
        arms: dict[str, Any] = {}
        for arm, order in orders.items():
            rows = _trace(
                basis, order, costs, references,
                hamiltonian=hamiltonian, ground=ground, stride=int(args.stride),
            )
            arms[arm] = {"class_size": len(fixed) if arm == "fixed_class_first" else None,
                         "trace": rows}
            best = min((r for r in rows if r["max_root_abs_error"] is not None),
                       key=lambda r: r["max_root_abs_error"], default=None)
            if best:
                print(
                    f"   {arm:<20} best {best['max_root_abs_error']:.1e} @ k={best['k']}, "
                    f"{best['cum_2q']:.0f}2Q   (final k={rows[-1]['k']})",
                    flush=True,
                )

        regimes_payload[regime] = {
            "u": float(u),
            "g_ep": float(g_ep),
            "n_ph_max": int(n_ph_max),
            "pool_size": len(basis),
            "fixed_class_size": len(fixed),
            "reference_excitations": references,
            "arms": arms,
        }

    payload = {
        "schema_version": "paper_iii_matched_alphabet_frontier_v1",
        "policy": "diagnostic_only_matched_alphabet_frontier",
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "post_run_diagnostic_only": True,
        },
        "convention": "identical record pool for every arm; arms differ only in acquisition order",
        "target_roots": _TARGET_ROOTS,
        "regimes": regimes_payload,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\noutput_json: {args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
