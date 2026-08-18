#!/usr/bin/env python3
"""Transition-strength accuracy of cost-selected QSE manifolds (Paper III).

The C1 claim sentence promises Ritz roots *and transition strengths*; this
driver supplies the transition-strength evidence. Per Paper I regime it
builds the site-0 density-fluctuation and phonon-displacement observables
(`hh_response_observables` conventions), computes exact reference strengths
``|<psi_r|O|psi_0>|^2`` for the lowest six sector excitations from dense ED,
and compares the QSE transition strengths of three supports:

- the complete fixed linear-response class,
- the multi-root geometry alpha=1 selection (budget 40),
- that selection after certified exchange (dominance, R=6).

Reported per arm: per-root absolute strength errors and the maximum
relative error over roots whose exact strength exceeds a floor (1e-8);
roots the support fails to resolve are counted separately. Statevector
diagnostics; never feeds controller decisions.
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
)
from pipelines.qse_spectra.compiled_costs import (
    ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
    annotate_basis_with_compiled_costs,
    resolve_cost_weights_preset,
)
from pipelines.qse_spectra.core import (
    QSEBasisVectorPolicy,
    compute_qse_spectra,
    polynomial_observable,
)
from pipelines.qse_spectra.exchange_maintenance import (
    QSEExchangeConfig,
    run_qse_exchange_maintenance,
)
from pipelines.qse_spectra.hh_response_observables import (
    HHResponseLayout,
    phonon_displacement_operator,
    site_density_operator,
)
from pipelines.qse_spectra.record_selection import (
    StaticRecordSelectionConfig,
    select_static_qse_records,
)
from src.quantum.compiled_polynomial import apply_compiled_polynomial, compile_polynomial_action

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "output/diagnostics/paper_iii_transition_strengths_20260818_v1/transition_strengths.json"
)
_Q0_POLICY = QSEBasisVectorPolicy(
    reference_projection="q0", basis_vector_normalization="raw_projected"
)
_FERMION_QUBITS_UP = (0, 1)
_FERMION_QUBITS_DN = (2, 3)
_TARGET_ROOTS = 6
_BUDGET = 40
_STRENGTH_FLOOR = 1.0e-8


def _sector_eigensystem(dense: np.ndarray, *, count: int) -> tuple[np.ndarray, list[np.ndarray]]:
    dim = int(dense.shape[0])
    occ_up = np.array([sum((i >> q) & 1 for q in _FERMION_QUBITS_UP) for i in range(dim)])
    occ_dn = np.array([sum((i >> q) & 1 for q in _FERMION_QUBITS_DN) for i in range(dim)])
    sector = np.where((occ_up == 1) & (occ_dn == 1))[0]
    restricted = dense[np.ix_(sector, sector)]
    _energies, vectors = np.linalg.eigh(0.5 * (restricted + restricted.conj().T))
    states = []
    for column in range(min(int(count), vectors.shape[1])):
        state = np.zeros(dim, dtype=complex)
        state[sector] = vectors[:, column]
        states.append(state)
    return states[0], states


def _dense_apply(poly: Any, state: np.ndarray) -> np.ndarray:
    return apply_compiled_polynomial(state, compile_polynomial_action(poly))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budget", type=int, default=_BUDGET)
    parser.add_argument("--target-roots", type=int, default=_TARGET_ROOTS)
    args = parser.parse_args(argv)
    target_roots = int(args.target_roots)

    regimes_payload: dict[str, Any] = {}
    for regime, u, g_ep, n_ph_max in PAPER_I_REGIMES:
        nq = _num_qubits(n_ph_max)
        hamiltonian, basis, _meta = _build_regime_pool(u=u, g_ep=g_ep, n_ph_max=n_ph_max)
        dense = _dense_hamiltonian(hamiltonian, 1 << nq)
        ground, sector_states = _sector_eigensystem(dense, count=target_roots + 1)

        layout = HHResponseLayout(
            num_sites=2,
            n_ph_max=int(n_ph_max),
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
        )
        observables = [
            polynomial_observable(
                site_density_operator(layout, site=0), name="site0_density"
            ),
            polynomial_observable(
                phonon_displacement_operator(layout, site=0), name="site0_displacement"
            ),
        ]

        exact_strengths: dict[str, list[float]] = {}
        for observable in observables:
            image = _dense_apply(observable.polynomial, ground)
            exact_strengths[observable.name] = [
                float(abs(complex(np.vdot(sector_states[root], image))) ** 2)
                for root in range(1, target_roots + 1)
            ]

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

        selection = select_static_qse_records(
            basis,
            config=StaticRecordSelectionConfig(
                mode="geometry_selected",
                max_records=int(args.budget),
                geometry_target_roots=target_roots,
                geometry_cost_discount_alpha=1.0,
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
                max_rounds=30, target_root_count=target_roots, insertion_shortlist_size=16
            ),
        )

        arms: dict[str, Any] = {}
        for arm_name, indices in (
            ("fixed_linear_response_complete", linear_indices),
            ("geometry_alpha1_R6", list(selection.selected_original_indices)),
            ("exchange_dominance_R6", list(exchange.final_indices)),
        ):
            result = compute_qse_spectra(
                hamiltonian,
                ground,
                tuple(basis[int(index)] for index in indices),
                basis_vector_policy=_Q0_POLICY,
                transition_observables=observables,
            )
            per_observable: dict[str, Any] = {}
            for record in result.transition_observables:
                name = str(record.observable.name)
                strengths = np.asarray(record.transition_strengths, dtype=float).reshape(-1)
                reference = exact_strengths[name]
                abs_errors: list[float | None] = []
                relative: list[float] = []
                unresolved = 0
                for root in range(target_roots):
                    if root >= strengths.size:
                        abs_errors.append(None)
                        unresolved += 1
                        continue
                    error = abs(float(strengths[root]) - reference[root])
                    abs_errors.append(error)
                    if reference[root] > _STRENGTH_FLOOR:
                        relative.append(error / reference[root])
                per_observable[name] = {
                    "abs_errors": abs_errors,
                    "max_relative_error": max(relative) if relative else None,
                    "unresolved_roots": unresolved,
                }
            arms[arm_name] = {
                "support_size": len(indices),
                "total_2q": float(sum(costs[int(index)] for index in indices)),
                "observables": per_observable,
            }

        regimes_payload[regime] = {
            "u": float(u),
            "g_ep": float(g_ep),
            "n_ph_max": int(n_ph_max),
            "exact_strengths": exact_strengths,
            "arms": arms,
        }
        print(f"\n== {regime} (u={u}, g={g_ep:.4f}, nph{n_ph_max})")
        for arm_name, arm in arms.items():
            parts = []
            for name, data in arm["observables"].items():
                rel = data["max_relative_error"]
                parts.append(
                    f"{name}: rel<={rel:.1e}" if rel is not None else f"{name}: --"
                )
                if data["unresolved_roots"]:
                    parts[-1] += f" ({data['unresolved_roots']} unresolved)"
            print(f"  {arm_name:<30} @{arm['total_2q']:.0f}2Q  " + "  ".join(parts))

    payload = {
        "schema_version": "paper_iii_qse_transition_strengths_v1",
        "policy": "diagnostic_only_transition_strengths",
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "post_run_diagnostic_only": True,
        },
        "target_roots": target_roots,
        "strength_floor": _STRENGTH_FLOOR,
        "observables": ["site0_density", "site0_displacement"],
        "regimes": regimes_payload,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\noutput_json: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
