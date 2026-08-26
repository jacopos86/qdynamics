#!/usr/bin/env python3
"""Matched-accuracy campaign: the coded comparison method for Paper III.

Implements `agent_guidance/qse/paper-iii-comparison-protocol.md`. Per regime
and per error target ``eps_E`` in the declared ladder, every arm reports

    C*(eps_E) = minimum compiled two-qubit cost at which that arm's
                certified output reaches max_{nu<=R} |Delta E_nu| <= eps_E,

with the protocol's named failure states instead of blanks:

- ``UNATTAINABLE_WITH_MANIFOLD`` -- the arm's class cannot reach eps_E at
  any budget (complete fixed class on a window it does not span);
- ``NOT_REACHED_WITHIN_POOL``   -- an adaptive/ordering arm exhausted the
  shared pool before reaching eps_E.

Arms (identical record pool; the fixed class is the declared physical
subclass of that pool, reported at its complete terminal point):

- ``ours``            -- residual-stop rung ladder + certified exchange at
  each rung; per target the minimum-cost target-reaching rung is selected
  and EVERY rung is reported (selection-bias disclosure, protocol section 5);
- ``fixed_class``     -- the complete linear-response class, one terminal
  point;
- ``cheapest_first``  -- prefix extension in ascending compiled cost;
- ``input_order``     -- prefix extension in pool order.

Exact sector references are read through the content-addressed store
(computed once per regime, verified on read; protocol section 6).

Statevector diagnostics; never feeds controller decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    / "output/diagnostics/paper_iii_matched_accuracy_20260826_v1/matched_accuracy_campaign.json"
)
REFERENCE_STORE = REPO_ROOT / "output/reference_store/paper_iii_exact_sector"
_Q0_POLICY = QSEBasisVectorPolicy(
    reference_projection="q0", basis_vector_normalization="raw_projected"
)
_TARGET_ROOTS = 6
ERROR_TARGET_LADDER = (1.0e-2, 1.0e-4, 1.0e-6)
RESIDUAL_RUNG_LADDER = (3.0e-2, 1.0e-2, 3.0e-3, 1.0e-3)

# Cheap-tier regime sets for method streamlining (user directive 2026-08-26):
# establish the method on cheap physics before any paper-facing matrix.
# hubbard_l2: g_ep=0 at n_ph_max=1 -- electronically a pure Hubbard dimer with
# a decoupled single-phonon register (window then contains E_el + n*omega0
# rows, which is fine for a testbed). nph1: the Paper-I (u, g) points at
# n_ph_max=1. paper_i: the production conventions (nph3/nph7).
REGIME_SETS = {
    "hubbard_l2": tuple(
        (f"hubbard_u{u:g}", u, 0.0, 1) for u in (0.25, 1.25, 8.0)
    ),
    "nph1": tuple(
        (regime, u, g_ep, 1) for regime, u, g_ep, _n in PAPER_I_REGIMES
    ),
    "nph3": tuple(
        (regime, u, g_ep, 3) for regime, u, g_ep, _n in PAPER_I_REGIMES
    ),
    "paper_i": tuple(PAPER_I_REGIMES),
}

STATUS_REACHED = "REACHED"
STATUS_UNATTAINABLE = "UNATTAINABLE_WITH_MANIFOLD"
STATUS_NOT_IN_POOL = "NOT_REACHED_WITHIN_POOL"


# --- content-addressed exact-reference store (protocol section 6) -----------

def _reference_key(identity: Mapping[str, Any]) -> str:
    blob = json.dumps(dict(identity), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def load_exact_reference(
    *, regime: str, u: float, g_ep: float, n_ph_max: int, count: int
) -> tuple[np.ndarray, list[float]]:
    """Ground vector + lowest ``count`` sector energies, computed once."""

    identity = {
        "family": "hh_l2_half_filled_11_sector",
        "regime": str(regime),
        "u": float(u),
        "g_ep": float(g_ep),
        "n_ph_max": int(n_ph_max),
        "count": int(count),
        "schema": "paper_iii_exact_sector_reference_v1",
    }
    key = _reference_key(identity)
    path = REFERENCE_STORE / f"{key}.npz"
    if path.is_file():
        payload = np.load(path, allow_pickle=False)
        stored = json.loads(str(payload["identity_json"]))
        if stored != identity:
            raise RuntimeError(
                f"reference-store key collision or corruption at {path}: "
                f"stored identity {stored} != requested {identity}"
            )
        return np.asarray(payload["ground"], dtype=complex), [
            float(x) for x in payload["energies"]
        ]
    nq = _num_qubits(n_ph_max)
    hamiltonian, _basis, _meta = _build_regime_pool(u=u, g_ep=g_ep, n_ph_max=n_ph_max)
    dense = _dense_hamiltonian(hamiltonian, 1 << nq)
    ground, energies = _sector_spectrum(dense, count=count)
    del dense
    REFERENCE_STORE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        ground=np.asarray(ground, dtype=complex),
        energies=np.asarray(energies, dtype=float),
        identity_json=np.asarray(json.dumps(identity, sort_keys=True)),
    )
    tmp.rename(path)
    return np.asarray(ground, dtype=complex), [float(x) for x in energies]


# --- pure cell-resolution logic (unit-tested) -------------------------------

def resolve_cell(
    rungs: Sequence[Mapping[str, Any]],
    eps_e: float,
    *,
    extendable: bool,
) -> dict[str, Any]:
    """Resolve one (arm, eps_E) cell from evaluated rungs.

    Each rung carries ``max_root_abs_error`` (None = window unresolved) and
    ``total_2q``. ``extendable`` distinguishes an arm that ran out of pool
    (NOT_REACHED_WITHIN_POOL) from a complete class that cannot be extended
    (UNATTAINABLE_WITH_MANIFOLD).
    """

    reaching = [
        r for r in rungs
        if r.get("max_root_abs_error") is not None
        and float(r["max_root_abs_error"]) <= float(eps_e)
    ]
    if reaching:
        best = min(reaching, key=lambda r: float(r["total_2q"]))
        return {
            "status": STATUS_REACHED,
            "cost_at_target": float(best["total_2q"]),
            "selected_rung": {k: v for k, v in best.items()},
        }
    finite = [r for r in rungs if r.get("max_root_abs_error") is not None]
    terminal = min(finite, key=lambda r: float(r["max_root_abs_error"])) if finite else None
    return {
        "status": STATUS_NOT_IN_POOL if extendable else STATUS_UNATTAINABLE,
        "cost_at_target": None,
        "terminal": None if terminal is None else {k: v for k, v in terminal.items()},
    }


# --- arm evaluation ---------------------------------------------------------

def _pencil_errors(
    basis: Sequence[Any],
    indices: Sequence[int],
    references: Sequence[float],
    *,
    hamiltonian: Any,
    ground: np.ndarray,
) -> tuple[float | None, int, list[float] | None]:
    try:
        result = compute_qse_spectra(
            hamiltonian,
            ground,
            tuple(basis[int(i)] for i in indices),
            basis_vector_policy=_Q0_POLICY,
        )
    except ValueError:
        # e.g. every image in the prefix is numerically zero after the q0
        # projection (decoupled records at g=0): rank-0 pencil, window
        # unresolved.
        return None, 0, None
    energies = np.asarray(result.eigenvalues, dtype=float).reshape(-1)
    roots = [float(x) for x in energies[: len(references)]]
    if energies.size < len(references):
        return None, int(result.retained_rank), roots
    return (
        max(abs(float(energies[r]) - float(ref)) for r, ref in enumerate(references)),
        int(result.retained_rank),
        roots,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--regimes", default=None)
    parser.add_argument("--regime-set", choices=sorted(REGIME_SETS), default="paper_i")
    parser.add_argument("--prefix-stride", type=int, default=4)
    args = parser.parse_args(argv)

    wanted = None if args.regimes is None else {t.strip() for t in str(args.regimes).split(",")}
    regimes_payload: dict[str, Any] = {}

    for regime, u, g_ep, n_ph_max in REGIME_SETS[str(args.regime_set)]:
        if wanted is not None and regime not in wanted:
            continue
        nq = _num_qubits(n_ph_max)
        hamiltonian, basis, _meta = _build_regime_pool(u=u, g_ep=g_ep, n_ph_max=n_ph_max)
        ground, spectrum = load_exact_reference(
            regime=regime, u=u, g_ep=g_ep, n_ph_max=n_ph_max, count=_TARGET_ROOTS + 1
        )
        references = spectrum[1 : _TARGET_ROOTS + 1]
        cost_rows = annotate_basis_with_compiled_costs(
            basis,
            num_qubits=nq,
            oracle_kind=ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
            cost_weights=resolve_cost_weights_preset("two_qubit_only_v1"),
        )
        costs = tuple(row.scalarized_canonical_cost for row in cost_rows)
        # Deterministic proxy resource triple per record (Paper-II analog,
        # routing-randomness-free): N2q = c_hat_2q (CX-ladder count),
        # D2q = c_hat_d (routed-chain two-qubit depth), Dc = D2q + c_hat_1q.
        c2q = tuple(float(r.estimate.c_hat_2q) for r in cost_rows)
        cd = tuple(float(r.estimate.c_hat_d) for r in cost_rows)
        c1q = tuple(float(r.estimate.c_hat_1q) for r in cost_rows)

        def _resources(indices: Sequence[int]) -> dict[str, float]:
            n2q = float(sum(c2q[int(i)] for i in indices))
            d2q = float(sum(cd[int(i)] for i in indices))
            return {"n2q": n2q, "d2q": d2q,
                    "dc": d2q + float(sum(c1q[int(i)] for i in indices))}

        print(f"\n== {regime} (pool={len(basis)})", flush=True)

        # ours: residual-rung ladder, exchange at each rung
        our_rungs: list[dict[str, Any]] = []
        for rung in RESIDUAL_RUNG_LADDER:
            selection = select_static_qse_records(
                basis,
                config=StaticRecordSelectionConfig(
                    mode="geometry_selected",
                    max_records=len(basis),
                    geometry_target_roots=_TARGET_ROOTS,
                    geometry_cost_discount_alpha=1.0,
                    geometry_residual_stop=float(rung),
                ),
                hamiltonian=hamiltonian,
                prepared_state=ground,
                basis_vector_policy=_Q0_POLICY,
                compiled_costs=costs,
            )
            exchange = run_qse_exchange_maintenance(
                basis,
                selection.selected_original_indices,
                costs,
                hamiltonian=hamiltonian,
                prepared_state=ground,
                basis_vector_policy=_Q0_POLICY,
                config=QSEExchangeConfig(
                    max_rounds=30,
                    target_root_count=_TARGET_ROOTS,
                    insertion_shortlist_size=16,
                ),
            )
            indices = list(exchange.final_indices)
            err, rank, roots = _pencil_errors(
                basis, indices, references, hamiltonian=hamiltonian, ground=ground
            )
            stop = selection.geometry_stop or {}
            our_rungs.append(
                {
                    "residual_rung": float(rung),
                    "k": len(indices),
                    "root_energies": roots,
                    "retained_rank": rank,
                    "total_2q": float(sum(costs[int(i)] for i in indices)),
                    "resources": _resources(indices),
                    "stop_reason": stop.get("stop_reason"),
                    "max_root_abs_error": err,
                }
            )
            print(
                f"   ours rung {rung:.0e}: k={len(indices)} "
                f"@{our_rungs[-1]['total_2q']:.0f}2Q err={err if err is None else f'{err:.1e}'}",
                flush=True,
            )

        # fixed class: complete, single terminal point
        fixed_indices = [
            i for i, e in enumerate(basis)
            if _element_family(e.name) in _LINEAR_RESPONSE_FAMILIES
        ]
        f_err, f_rank, f_roots = _pencil_errors(
            basis, fixed_indices, references, hamiltonian=hamiltonian, ground=ground
        )
        fixed_rungs = [
            {
                "k": len(fixed_indices),
                "root_energies": f_roots,
                "retained_rank": f_rank,
                "total_2q": float(sum(costs[int(i)] for i in fixed_indices)),
                "resources": _resources(fixed_indices),
                "max_root_abs_error": f_err,
            }
        ]
        print(
            f"   fixed class: k={len(fixed_indices)} @{fixed_rungs[0]['total_2q']:.0f}2Q "
            f"err={f_err if f_err is None else f'{f_err:.1e}'}",
            flush=True,
        )

        # ordering controls: prefix extension
        order_rungs: dict[str, list[dict[str, Any]]] = {}
        orders = {
            "cheapest_first": sorted(range(len(basis)), key=lambda i: (float(costs[i]), i)),
            "input_order": list(range(len(basis))),
        }
        for arm, order in orders.items():
            rows: list[dict[str, Any]] = []
            for position in range(int(args.prefix_stride), len(order) + 1, int(args.prefix_stride)):
                prefix = order[:position]
                err, rank, _roots = _pencil_errors(
                    basis, prefix, references, hamiltonian=hamiltonian, ground=ground
                )
                rows.append(
                    {
                        "k": position,
                        "retained_rank": rank,
                        "total_2q": float(sum(costs[int(i)] for i in prefix)),
                        "resources": _resources(prefix),
                        "max_root_abs_error": err,
                    }
                )
            order_rungs[arm] = rows

        cells: dict[str, Any] = {}
        for eps_e in ERROR_TARGET_LADDER:
            cells[f"{eps_e:.0e}"] = {
                "ours": resolve_cell(our_rungs, eps_e, extendable=True),
                "fixed_class": resolve_cell(fixed_rungs, eps_e, extendable=False),
                "cheapest_first": resolve_cell(order_rungs["cheapest_first"], eps_e, extendable=True),
                "input_order": resolve_cell(order_rungs["input_order"], eps_e, extendable=True),
            }

        regimes_payload[regime] = {
            "u": float(u),
            "g_ep": float(g_ep),
            "n_ph_max": int(n_ph_max),
            "pool_size": len(basis),
            "fixed_class_size": len(fixed_indices),
            "reference_ground_energy": float(spectrum[0]),
            "reference_excitations": references,
            "rungs": {
                "ours": our_rungs,
                "fixed_class": fixed_rungs,
                **order_rungs,
            },
            "cells": cells,
        }

    payload = {
        "schema_version": "paper_iii_matched_accuracy_campaign_v1",
        "protocol": "agent_guidance/qse/paper-iii-comparison-protocol.md",
        "policy": "diagnostic_only_matched_accuracy_campaign",
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "post_run_diagnostic_only": True,
        },
        "locks": {
            "alphabet": "identical record pool for every record-based arm",
            "pencil_policy": "q0_raw_projected_shared_cutoff",
            "cost_model": "two_qubit_only_v1_graph_span",
            "score": "metric_novelty_plus_residual_capture_cost_discounted",
            "error_target_ladder": [float(x) for x in ERROR_TARGET_LADDER],
            "residual_rung_ladder": [float(x) for x in RESIDUAL_RUNG_LADDER],
        },
        "selection_bias_note": (
            "per-cell minimum-cost target-reaching rung is exact-reference-tuned; "
            "all rungs reported"
        ),
        "target_roots": _TARGET_ROOTS,
        "regime_set": str(args.regime_set),
        "regimes": regimes_payload,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\noutput_json: {args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
