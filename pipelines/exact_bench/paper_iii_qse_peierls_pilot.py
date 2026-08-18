#!/usr/bin/env python3
"""Paper III Peierls--Hubbard pilot: bond-coupled phonons, multi-root arms.

Second electron--phonon family for the generality arm. The L=2
Peierls--Hubbard (SSH-type) dimer couples one bond phonon to the *hopping*
rather than the density:

    H = -t K + g X_b K + U sum_i n_{i,up} n_{i,dn} + omega (n_b + 1/2)

with K = sum_sigma (c^dag_0 c_1 + h.c.), X_b = b + b^dag on the single
bond boson (nph3, binary, 2 qubits), fermions blocked (up q0,q1; dn q2,q3),
phonon on q4,q5 — 6 qubits, dim 64. Built from repo primitives
(JW hopping pairs, ``boson_operator``/``boson_displacement_operator``,
``PauliPolynomial`` products); this is a pilot-grade in-driver family, with
promotion into the problem registry left as the architecture step for the
user to review.

The pool (``peierls_v1``, documented here) adapts the HH full_meta classes
to bond coupling: identity, lifted fermionic UCCSD singles/doubles, hop and
current layers, bond-phonon ladder/displacement/momentum powers, and mixed
hop/current/density-difference times displacement/momentum products. The
fixed "linear response" comparator class is identity + fermionic singles +
bond-phonon ladder — the class a practitioner would write down first,
which for bond coupling is expected to miss the hop-dressed directions.

Arms mirror the L=3 pilot (input order, fixed class, full pool,
multi-root geometry alpha=1, certified exchange R=6) with exact
(1,1)-sector references. Statevector diagnostics; never feeds controller
decisions.
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
from src.quantum.compiled_polynomial import apply_compiled_polynomial, compile_polynomial_action
from src.quantum.hubbard_latex_python_pairs import (
    boson_displacement_operator,
    boson_operator,
    jw_number_operator,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm

DEFAULT_OUTPUT = (
    REPO_ROOT / "output/diagnostics/paper_iii_peierls_pilot_20260819_v1/peierls_pilot_summary.json"
)
_Q0_POLICY = QSEBasisVectorPolicy(
    reference_projection="q0", basis_vector_normalization="raw_projected"
)
_NQ = 6
_N_PH_MAX = 3
_PHONON_QUBITS = (4, 5)
_UP_MODES = (0, 1)
_DN_MODES = (2, 3)
_TARGET_ROOTS = 6
_BUDGET = 20
_G_WEAK = 0.353553390593
_G_STRONG = 0.790569415042
PEIERLS_REGIMES: tuple[tuple[str, float, float], ...] = (
    ("peierls_weak_weak", 0.25, _G_WEAK),
    ("peierls_weak_strong", 0.25, _G_STRONG),
    ("peierls_u8_weak", 8.0, _G_WEAK),
)


def _label(placements: dict[int, str]) -> str:
    chars = ["e"] * _NQ
    for qubit, pauli in placements.items():
        chars[_NQ - 1 - int(qubit)] = pauli
    return "".join(chars)


def _poly(terms: list[tuple[str, complex]]) -> PauliPolynomial:
    out = PauliPolynomial("JW")
    for label, coeff in terms:
        out.add_term(PauliTerm(_NQ, ps=label, pc=complex(coeff)))
    return out


def _pair_hop(low: int, high: int) -> PauliPolynomial:
    """(c^dag_low c_high + h.c.) on adjacent JW modes: (X X + Y Y)/2."""

    return _poly(
        [
            (_label({high: "x", low: "x"}), 0.5),
            (_label({high: "y", low: "y"}), 0.5),
        ]
    )


def _pair_current(low: int, high: int) -> PauliPolynomial:
    """i(c^dag_high c_low - h.c.): (X Y - Y X)/2."""

    return _poly(
        [
            (_label({high: "x", low: "y"}), 0.5),
            (_label({high: "y", low: "x"}), -0.5),
        ]
    )


def _bond_number() -> PauliPolynomial:
    out = PauliPolynomial("JW")
    for offset, qubit in enumerate(_PHONON_QUBITS):
        weight = float(1 << offset)
        out.add_term(PauliTerm(_NQ, ps=_label({}), pc=0.5 * weight))
        out.add_term(PauliTerm(_NQ, ps=_label({qubit: "z"}), pc=-0.5 * weight))
    return out


def build_peierls_hamiltonian(*, u: float, g_ep: float, t: float = 1.0, omega0: float = 1.0) -> PauliPolynomial:
    hop_total = _pair_hop(*_UP_MODES) + _pair_hop(*_DN_MODES)
    displacement = boson_displacement_operator(
        "JW", _NQ, _PHONON_QUBITS, n_ph_max=_N_PH_MAX, encoding="binary"
    )
    hamiltonian = (-float(t)) * hop_total
    hamiltonian = hamiltonian + float(g_ep) * (displacement * hop_total)
    for site in range(2):
        n_up = jw_number_operator("JW", _NQ, _UP_MODES[site])
        n_dn = jw_number_operator("JW", _NQ, _DN_MODES[site])
        hamiltonian = hamiltonian + float(u) * (n_up * n_dn)
    hamiltonian = hamiltonian + float(omega0) * _bond_number()
    hamiltonian = hamiltonian + (0.5 * float(omega0))
    return hamiltonian


def build_peierls_pool() -> list[Any]:
    """peierls_v1: HH full_meta operator classes adapted to bond coupling."""

    displacement = boson_displacement_operator(
        "JW", _NQ, _PHONON_QUBITS, n_ph_max=_N_PH_MAX, encoding="binary"
    )
    b_minus = boson_operator("JW", _NQ, _PHONON_QUBITS, which="b", n_ph_max=_N_PH_MAX, encoding="binary")
    b_plus = boson_operator("JW", _NQ, _PHONON_QUBITS, which="bdag", n_ph_max=_N_PH_MAX, encoding="binary")
    momentum = (1j * b_plus) + ((-1j) * b_minus)
    hop_up, hop_dn = _pair_hop(*_UP_MODES), _pair_hop(*_DN_MODES)
    cur_up, cur_dn = _pair_current(*_UP_MODES), _pair_current(*_DN_MODES)
    hop_total, cur_total = hop_up + hop_dn, cur_up + cur_dn
    density_diff = (
        jw_number_operator("JW", _NQ, _UP_MODES[1])
        + jw_number_operator("JW", _NQ, _DN_MODES[1])
        + (-1.0) * jw_number_operator("JW", _NQ, _UP_MODES[0])
        + (-1.0) * jw_number_operator("JW", _NQ, _DN_MODES[0])
    )
    # Lifted fermionic UCCSD anti-Hermitian generators i(T - T^dag) appear
    # through their action directions; for QSE basis vectors the Hermitian
    # hop/current pair spans the same single-excitation directions, and the
    # double excitation is the paired-hop product.
    double_hop = hop_up * hop_dn
    spin_z_site0 = jw_number_operator("JW", _NQ, _UP_MODES[0]) + (-1.0) * jw_number_operator("JW", _NQ, _DN_MODES[0])
    spin_z_site1 = jw_number_operator("JW", _NQ, _UP_MODES[1]) + (-1.0) * jw_number_operator("JW", _NQ, _DN_MODES[1])
    spin_z_diff = spin_z_site0 + (-1.0) * spin_z_site1

    named: list[tuple[str, PauliPolynomial]] = [
        ("uccsd_sing_up_hop", hop_up),
        ("uccsd_sing_dn_hop", hop_dn),
        ("uccsd_sing_up_cur", cur_up),
        ("uccsd_sing_dn_cur", cur_dn),
        ("uccsd_dbl", double_hop),
        ("hop_layer", hop_total),
        ("current_layer", cur_total),
        ("bond_b", b_minus),
        ("bond_bdag", b_plus),
        ("bond_X", displacement),
        ("bond_P", momentum),
        ("bond_XX", displacement * displacement),
        ("bond_XP", displacement * momentum),
        ("bond_n", _bond_number()),
        ("density_diff", density_diff),
        ("peierls_hopX", hop_total * displacement),
        ("peierls_hopP", hop_total * momentum),
        ("peierls_curX", cur_total * displacement),
        ("peierls_curP", cur_total * momentum),
        ("peierls_hopXX", hop_total * (displacement * displacement)),
        ("peierls_dnX", density_diff * displacement),
        ("peierls_dnP", density_diff * momentum),
        ("peierls_dblX", double_hop * displacement),
        ("peierls_up_hopX", hop_up * displacement),
        ("peierls_dn_hopX", hop_dn * displacement),
        ("peierls_up_curX", cur_up * displacement),
        ("peierls_dn_curX", cur_dn * displacement),
        ("peierls_up_curP", cur_up * momentum),
        ("peierls_dn_curP", cur_dn * momentum),
        ("site0_density", jw_number_operator("JW", _NQ, _UP_MODES[0]) + jw_number_operator("JW", _NQ, _DN_MODES[0])),
        ("site0_densityX", (jw_number_operator("JW", _NQ, _UP_MODES[0]) + jw_number_operator("JW", _NQ, _DN_MODES[0])) * displacement),
        # v2 enrichment: multi-quanta phonon powers and their dressed products
        # so the manifold spans the 2- and 3-quantum sector states at nph3.
        ("bond_XXX", displacement * (displacement * displacement)),
        ("bond_PP", momentum * momentum),
        ("bond_XPP", displacement * (momentum * momentum)),
        ("peierls_hopXXX", hop_total * (displacement * (displacement * displacement))),
        ("peierls_hopPP", hop_total * (momentum * momentum)),
        ("peierls_curXX", cur_total * (displacement * displacement)),
        ("peierls_curPP", cur_total * (momentum * momentum)),
        ("peierls_dnXX", density_diff * (displacement * displacement)),
        ("peierls_dblXX", double_hop * (displacement * displacement)),
        ("peierls_dblP", double_hop * momentum),
        ("site0_densityXX", (jw_number_operator("JW", _NQ, _UP_MODES[0]) + jw_number_operator("JW", _NQ, _DN_MODES[0])) * (displacement * displacement)),
        ("site0_densityP", (jw_number_operator("JW", _NQ, _UP_MODES[0]) + jw_number_operator("JW", _NQ, _DN_MODES[0])) * momentum),
        ("bond_nX", _bond_number() * displacement),
        ("bond_nHop", _bond_number() * hop_total),
        # v3: S^2-breaking (S_z- and number-conserving) spin-structure
        # operators — the parity-odd triplet-like sector states are invisible
        # to every spin-symmetric operator above.
        ("spin_z_diff", spin_z_diff),
        ("spin_z_total_site0", spin_z_site0),
        ("spin_z_diffX", spin_z_diff * displacement),
        ("spin_z_diffP", spin_z_diff * momentum),
        ("spin_z_diffXX", spin_z_diff * (displacement * displacement)),
        ("spin_z_diff_hop", spin_z_diff * hop_total),
        ("spin_z_diff_cur", spin_z_diff * cur_total),
    ]
    basis = [pauli_string_basis_element("e" * _NQ, nq=_NQ, name="identity")]
    for name, poly in named:
        basis.append(polynomial_basis_element(poly, name=name, metadata={"pool": "peierls_v1"}))
    return basis


_FIXED_CLASS = (
    "identity",
    "uccsd_sing_up_hop",
    "uccsd_sing_dn_hop",
    "uccsd_sing_up_cur",
    "uccsd_sing_dn_cur",
    "uccsd_dbl",
    "bond_b",
    "bond_bdag",
    "bond_X",
    "bond_P",
)


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
    occ_up = np.array([sum((i >> q) & 1 for q in _UP_MODES) for i in range(dim)])
    occ_dn = np.array([sum((i >> q) & 1 for q in _DN_MODES) for i in range(dim)])
    sector = np.where((occ_up == 1) & (occ_dn == 1))[0]
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
    args = parser.parse_args(argv)
    target_roots = int(args.target_roots)

    basis = build_peierls_pool()
    cost_rows = annotate_basis_with_compiled_costs(
        basis,
        num_qubits=_NQ,
        oracle_kind=ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
        cost_weights=resolve_cost_weights_preset("two_qubit_only_v1"),
    )
    costs = tuple(row.scalarized_canonical_cost for row in cost_rows)
    fixed_indices = [i for i, e in enumerate(basis) if e.name in _FIXED_CLASS]

    regimes_payload: dict[str, Any] = {}
    for regime, u, g_ep in PEIERLS_REGIMES:
        hamiltonian = build_peierls_hamiltonian(u=u, g_ep=g_ep)
        dense = _dense_hamiltonian(hamiltonian, 1 << _NQ)
        ground, spectrum = _sector_spectrum(dense, count=target_roots + 1)
        references = spectrum[1 : target_roots + 1]

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
        _record("fixed_linear_response_complete", fixed_indices)
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
                max_rounds=20, target_root_count=target_roots, insertion_shortlist_size=12
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
            "num_qubits": _NQ,
            "pool_size": len(basis),
            "reference_excitations": references,
            "arms": arms,
        }
        print(f"\n== {regime} (u={u}, g={g_ep:.4f}, bond phonon nph3)")
        for arm_name, arm in arms.items():
            roots = " ".join(
                f"{e:.1e}" if e is not None else "--" for e in arm["root_abs_errors"]
            )
            patches = (
                f" ({arm['committed_patch_count']}p)" if "committed_patch_count" in arm else ""
            )
            print(f"  {arm_name:<30} @{arm['total_2q']:.0f}2Q{patches}  roots: {roots}")

    payload = {
        "schema_version": "paper_iii_qse_peierls_pilot_v1",
        "policy": "diagnostic_only_peierls_pilot",
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "post_run_diagnostic_only": True,
        },
        "model": "peierls_hubbard_dimer_bond_phonon_nph3",
        "pool": "peierls_v1_in_driver_pilot",
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
