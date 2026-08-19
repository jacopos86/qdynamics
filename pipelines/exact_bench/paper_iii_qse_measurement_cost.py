#!/usr/bin/env python3
"""Matrix-measurement estimator work for Paper III QSE supports.

The compiled-two-qubit cost of a QSE support counts state-preparation
circuits per record.  It is not the dominant resource: building the
pencil requires the ``k(k+1)/2`` independent elements of both ``S`` and
``H``, each of which expands into Pauli expectation values on the shared
reference.  This driver reports that estimator work directly, in the
Paper-I ``S_alg`` spirit:

- ``pair_count``      -- independent matrix elements ``k(k+1)/2`` (per matrix);
- ``naive_terms``     -- total Pauli terms summed over all elements, i.e. the
  work of estimating every element independently with no reuse;
- ``distinct_words``  -- distinct Pauli words over the whole pencil, the work
  after global term reuse across elements;
- ``qwc_groups``      -- qubit-wise-commuting basis-cover groups over those
  words (greedy cover, the Paper-I ``qwc_basis_cover_reuse`` convention),
  i.e. the number of distinct measurement settings.

Words are tracked without phases or coefficients: only which Pauli strings
must be estimated. Reporting only; never feeds controller decisions.
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

import numpy as np

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
from pipelines.qse_spectra.core import QSEBasisVectorPolicy
from pipelines.qse_spectra.exchange_maintenance import (
    QSEExchangeConfig,
    run_qse_exchange_maintenance,
)
from pipelines.qse_spectra.record_selection import (
    StaticRecordSelectionConfig,
    select_static_qse_records,
)

_Q0_POLICY = QSEBasisVectorPolicy(
    reference_projection="q0", basis_vector_normalization="raw_projected"
)
_TARGET_ROOTS = 6

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "output/diagnostics/paper_iii_measurement_cost_20260819_v1/measurement_cost.json"
)
MULTIROOT_JSON = (
    REPO_ROOT
    / "output/diagnostics/paper_iii_multiroot_sweep_20260818_v1/multiroot_sweep_epsstop.json"
)

# Pauli letter product ignoring phase: only the resulting letter matters.
_PROD = {
    ("e", "e"): "e", ("e", "x"): "x", ("e", "y"): "y", ("e", "z"): "z",
    ("x", "e"): "x", ("x", "x"): "e", ("x", "y"): "z", ("x", "z"): "y",
    ("y", "e"): "y", ("y", "x"): "z", ("y", "y"): "e", ("y", "z"): "x",
    ("z", "e"): "z", ("z", "x"): "y", ("z", "y"): "x", ("z", "z"): "e",
}


def _word_product(left: str, right: str) -> str:
    return "".join(_PROD[(a, b)] for a, b in zip(left, right))


def _element_words(element: Any, *, nq: int) -> tuple[str, ...]:
    if str(element.kind) == "pauli_string":
        return (str(element.pauli_label_exyz),)
    terms = element.polynomial.return_polynomial()
    return tuple(dict.fromkeys(str(term.pw2strng()) for term in terms))


def _polynomial_words(polynomial: Any) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(str(term.pw2strng()) for term in polynomial.return_polynomial())
    )


def _qwc_cover(words: Sequence[str], *, nq: int) -> int:
    """Greedy qubit-wise-commuting basis cover; returns the group count."""

    groups: list[list[str]] = []
    for word in sorted(words, key=lambda w: (-sum(c != "e" for c in w), w)):
        placed = False
        for basis in groups:
            if all(b == "e" or c == "e" or b == c for b, c in zip(basis[0], word)):
                merged = [
                    c if b == "e" else b for b, c in zip(basis[0], word)
                ]
                basis[0] = "".join(merged)
                placed = True
                break
        if not placed:
            groups.append([word])
    return len(groups)


def _support_cost(
    basis: Sequence[Any],
    indices: Sequence[int],
    hamiltonian_words: Sequence[str],
    *,
    nq: int,
) -> dict[str, Any]:
    words_by_index = {int(i): _element_words(basis[int(i)], nq=nq) for i in indices}
    ordered = [int(i) for i in indices]

    overlap_words: set[str] = set()
    hamiltonian_matrix_words: set[str] = set()
    naive_overlap = 0
    naive_hamiltonian = 0

    for pos_a, a in enumerate(ordered):
        wa = words_by_index[a]
        for b in ordered[pos_a:]:
            wb = words_by_index[b]
            products = {_word_product(x, y) for x in wa for y in wb}
            overlap_words |= products
            naive_overlap += len(products)
            ham_products = {
                _word_product(p, h) for p in products for h in hamiltonian_words
            }
            hamiltonian_matrix_words |= ham_products
            naive_hamiltonian += len(ham_products)

    all_words = overlap_words | hamiltonian_matrix_words
    k = len(ordered)
    return {
        "support_size": k,
        "pair_count": k * (k + 1) // 2,
        "naive_terms_overlap": int(naive_overlap),
        "naive_terms_hamiltonian": int(naive_hamiltonian),
        "naive_terms_total": int(naive_overlap + naive_hamiltonian),
        "distinct_words_overlap": len(overlap_words),
        "distinct_words_hamiltonian": len(hamiltonian_matrix_words),
        "distinct_words_total": len(all_words),
        "qwc_groups_total": _qwc_cover(sorted(all_words), nq=nq),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--regimes", default=None)
    parser.add_argument("--residual-stop", type=float, default=1.0e-3)
    parser.add_argument("--max-rounds", type=int, default=30)
    args = parser.parse_args(argv)

    wanted = None if args.regimes is None else {t.strip() for t in str(args.regimes).split(",")}

    regimes_payload: dict[str, Any] = {}
    for regime, u, g_ep, n_ph_max in PAPER_I_REGIMES:
        if wanted is not None and regime not in wanted:
            continue
        nq = _num_qubits(n_ph_max)
        hamiltonian, basis, _meta = _build_regime_pool(u=u, g_ep=g_ep, n_ph_max=n_ph_max)
        hamiltonian_words = _polynomial_words(hamiltonian)
        dense = _dense_hamiltonian(hamiltonian, 1 << nq)
        ground, _spectrum = _sector_spectrum(dense, count=_TARGET_ROOTS + 1)
        del dense
        cost_rows = annotate_basis_with_compiled_costs(
            basis,
            num_qubits=nq,
            oracle_kind=ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
            cost_weights=resolve_cost_weights_preset("two_qubit_only_v1"),
        )
        costs = tuple(row.scalarized_canonical_cost for row in cost_rows)

        fixed_indices = [
            index
            for index, element in enumerate(basis)
            if _element_family(element.name) in _LINEAR_RESPONSE_FAMILIES
        ]
        selection = select_static_qse_records(
            basis,
            config=StaticRecordSelectionConfig(
                mode="geometry_selected",
                max_records=len(basis),
                geometry_target_roots=_TARGET_ROOTS,
                geometry_cost_discount_alpha=1.0,
                geometry_residual_stop=float(args.residual_stop),
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
                max_rounds=int(args.max_rounds),
                target_root_count=_TARGET_ROOTS,
                insertion_shortlist_size=16,
            ),
        )
        arms_payload: dict[str, Any] = {}
        for arm_key, indices in (
            ("fixed_linear_response_complete", fixed_indices),
            ("geometry_alpha1_R6", list(selection.selected_original_indices)),
            ("exchange_dominance_R6", list(exchange.final_indices)),
        ):
            arms_payload[arm_key] = _support_cost(basis, indices, hamiltonian_words, nq=nq)
            arms_payload[arm_key]["total_2q"] = float(sum(costs[int(i)] for i in indices))
            arms_payload[arm_key]["selected_original_indices"] = [int(i) for i in indices]

        regimes_payload[regime] = {
            "u": float(u),
            "g_ep": float(g_ep),
            "n_ph_max": int(n_ph_max),
            "num_qubits": nq,
            "hamiltonian_pauli_words": len(hamiltonian_words),
            "arms": arms_payload,
        }
        print(f"\n== {regime} (nq={nq}, H words={len(hamiltonian_words)})", flush=True)
        for arm_key, cost in arms_payload.items():
            print(
                f"  {arm_key:<34} k={cost['support_size']:<4} elems={cost['pair_count']:<6} "
                f"naive={cost['naive_terms_total']:<9} distinct={cost['distinct_words_total']:<7} "
                f"qwc={cost['qwc_groups_total']}",
                flush=True,
            )

    payload = {
        "schema_version": "paper_iii_qse_measurement_cost_v1",
        "policy": "diagnostic_only_measurement_estimator_work",
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "post_run_diagnostic_only": True,
        },
        "grouping_mode": "qwc_basis_cover_greedy",
        "convention": "words tracked without phase or coefficient; S and H pencils counted together",
        "regimes": regimes_payload,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\noutput_json: {args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
