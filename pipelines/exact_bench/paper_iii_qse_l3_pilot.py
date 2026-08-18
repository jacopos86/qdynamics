#!/usr/bin/env python3
"""Paper III L=3 pilot: multi-root selection and exchange beyond the dimer.

Runs the multi-root (R=6) selection + certified-exchange pipeline on the
L=3 Hubbard--Holstein chain at the canonical Paper I L=3 weak-Holstein
diagnostic points (nph1, binary encoding, blocked ordering, open boundary;
9 qubits, dim 512), with the half-filled ``(n_up, n_dn) = (2, 1)`` sector
supplying exact references. The purpose is scaling evidence: at L=3 the
full_meta pool grows well beyond the fixed linear-response class, so the
selection problem stops being a toy. Arms mirror the L=2 multi-root sweep.
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
    _LINEAR_RESPONSE_FAMILIES,
    _element_family,
)
from pipelines.qse_spectra.compiled_costs import (
    ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
    annotate_basis_with_compiled_costs,
    resolve_cost_weights_preset,
)
from pipelines.qse_spectra.core import (
    QSEBasisVectorPolicy,
    compute_qse_spectra,
    pauli_string_basis_element,
    polynomial_basis_element,
)
from pipelines.qse_spectra.exchange_maintenance import (
    QSEExchangeConfig,
    run_qse_exchange_maintenance,
)
from pipelines.qse_spectra.record_selection import (
    StaticRecordSelectionConfig,
    select_static_qse_records,
)
from pipelines.qse_spectra.static_adapt_adapter import (
    build_artifact_problem_hamiltonian,
    build_hh_full_meta_pool_for_qse,
)
from src.quantum.compiled_polynomial import apply_compiled_polynomial, compile_polynomial_action

DEFAULT_OUTPUT = (
    REPO_ROOT / "output/diagnostics/paper_iii_l3_pilot_20260819_v1/l3_pilot_summary.json"
)
_Q0_POLICY = QSEBasisVectorPolicy(
    reference_projection="q0", basis_vector_normalization="raw_projected"
)
_L = 3
_N_PH_MAX = 1
_NUM_PARTICLES = (2, 1)
_G_WEAK = 0.353553390593
L3_REGIMES: tuple[tuple[str, float, float], ...] = (
    ("l3_weak", 0.25, _G_WEAK),
    ("l3_u8", 8.0, _G_WEAK),
)
_TARGET_ROOTS = 6
_BUDGET = 60


def _num_qubits() -> int:
    qubits_per_boson = 1  # nph1 binary
    return 2 * _L + _L * qubits_per_boson


def _build_l3_pool(*, u: float, g_ep: float) -> tuple[Any, list[Any]]:
    physics = dict(
        num_sites=_L,
        t=1.0,
        u=float(u),
        omega0=1.0,
        g_ep=float(g_ep),
        dv=0.0,
        n_ph_max=_N_PH_MAX,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
    )
    hamiltonian = build_artifact_problem_hamiltonian(
        problem_key="hh", include_zero_point=True, **physics
    )
    terms, _meta = build_hh_full_meta_pool_for_qse(
        h_poly=hamiltonian,
        paop_r=1,
        paop_split_paulis=False,
        paop_prune_eps=0.0,
        paop_normalization="none",
        num_particles=_NUM_PARTICLES,
        **physics,
    )
    nq = _num_qubits()
    basis = [pauli_string_basis_element("e" * nq, nq=nq, name="identity")]
    for term in terms:
        basis.append(polynomial_basis_element(term.polynomial, name=str(term.label)))
    return hamiltonian, basis


def _dense_hamiltonian(hamiltonian: Any, dim: int) -> np.ndarray:
    compiled = compile_polynomial_action(hamiltonian)
    matrix = np.zeros((dim, dim), dtype=complex)
    for column in range(dim):
        unit = np.zeros(dim, dtype=complex)
        unit[column] = 1.0
        matrix[:, column] = apply_compiled_polynomial(unit, compiled)
    return 0.5 * (matrix + matrix.conj().T)


def _sector_spectrum(dense: np.ndarray, *, count: int) -> tuple[np.ndarray, list[float]]:
    dim = int(dense.shape[0])
    up_qubits = tuple(range(_L))
    dn_qubits = tuple(range(_L, 2 * _L))
    occ_up = np.array([sum((i >> q) & 1 for q in up_qubits) for i in range(dim)])
    occ_dn = np.array([sum((i >> q) & 1 for q in dn_qubits) for i in range(dim)])
    sector = np.where(
        (occ_up == _NUM_PARTICLES[0]) & (occ_dn == _NUM_PARTICLES[1])
    )[0]
    restricted = dense[np.ix_(sector, sector)]
    energies, vectors = np.linalg.eigh(0.5 * (restricted + restricted.conj().T))
    ground = np.zeros(dim, dtype=complex)
    ground[sector] = vectors[:, 0]
    return ground, [float(value) for value in energies[: int(count)]]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budget", type=int, default=_BUDGET)
    parser.add_argument("--target-roots", type=int, default=_TARGET_ROOTS)
    parser.add_argument("--max-rounds", type=int, default=30)
    args = parser.parse_args(argv)
    target_roots = int(args.target_roots)
    nq = _num_qubits()

    regimes_payload: dict[str, Any] = {}
    for regime, u, g_ep in L3_REGIMES:
        hamiltonian, basis = _build_l3_pool(u=u, g_ep=g_ep)
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

        def _root_errors(indices: Sequence[int]) -> tuple[list[float | None], float]:
            result = compute_qse_spectra(
                hamiltonian,
                ground,
                tuple(basis[int(index)] for index in indices),
                basis_vector_policy=_Q0_POLICY,
            )
            energies = np.asarray(result.eigenvalues, dtype=float).reshape(-1)
            errors: list[float | None] = []
            for root, reference in enumerate(references):
                errors.append(
                    abs(float(energies[root]) - reference) if root < energies.size else None
                )
            return errors, float(sum(costs[int(index)] for index in indices))

        arms: dict[str, Any] = {}

        def _record(arm_name: str, indices: Sequence[int], **extra: Any) -> None:
            errors, total = _root_errors(indices)
            arms[arm_name] = {
                "support_size": len(indices),
                "total_2q": total,
                "root_abs_errors": errors,
                **extra,
            }

        _record("input_order", list(range(min(int(args.budget), len(basis)))))
        _record("fixed_linear_response_complete", linear_indices)
        _record("full_pool_manifold_limit", list(range(len(basis))))

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
        _record("geometry_alpha1_R6", selection.selected_original_indices)

        exchange = run_qse_exchange_maintenance(
            basis,
            selection.selected_original_indices,
            costs,
            hamiltonian=hamiltonian,
            prepared_state=ground,
            basis_vector_policy=_Q0_POLICY,
            config=QSEExchangeConfig(
                max_rounds=int(args.max_rounds),
                target_root_count=target_roots,
                insertion_shortlist_size=16,
            ),
        )
        _record(
            "exchange_dominance_R6",
            exchange.final_indices,
            committed_patch_count=sum(
                1 for round_record in exchange.rounds if round_record["committed_patch"] is not None
            ),
        )

        regimes_payload[regime] = {
            "u": float(u),
            "g_ep": float(g_ep),
            "L": _L,
            "n_ph_max": _N_PH_MAX,
            "num_qubits": nq,
            "num_particles": list(_NUM_PARTICLES),
            "pool_size": len(basis),
            "fixed_class_size": len(linear_indices),
            "reference_excitations": references,
            "arms": arms,
        }
        print(f"\n== {regime} (u={u}, g={g_ep:.4f}, L=3 nph1, pool {len(basis)})")
        for arm_name, arm in arms.items():
            roots = " ".join(
                f"{e:.1e}" if e is not None else "--" for e in arm["root_abs_errors"]
            )
            patches = (
                f" ({arm['committed_patch_count']}p)" if "committed_patch_count" in arm else ""
            )
            print(f"  {arm_name:<30} @{arm['total_2q']:.0f}2Q{patches}  roots: {roots}")

    payload = {
        "schema_version": "paper_iii_qse_l3_pilot_v1",
        "policy": "diagnostic_only_l3_pilot",
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "post_run_diagnostic_only": True,
        },
        "target_roots": target_roots,
        "budget": int(args.budget),
        "cost_weights_preset": "two_qubit_only_v1",
        "regimes": regimes_payload,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\noutput_json: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
