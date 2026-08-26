#!/usr/bin/env python3
"""Cross-family comparator matrix on the full Paper III Hamiltonian set.

Extends the dimer-point comparators to every benchmark Hamiltonian: the six
Paper-I Hubbard--Holstein regimes (nph3/nph7 conventions) and the three
Peierls--Hubbard coupling points. Per case, on the identical Hamiltonian,
sector, and reference set used by the selection arms:

- **fixed linear-response class** (complete, deterministic): per-root
  errors over the lowest six excitations and its compiled-2Q total;
- **real-time Krylov**: states ``exp(-i H k dt)`` from the normalized
  Hamiltonian-residual kick of the sector ground state, statevector-exact
  propagation, pencil solved with the standard cutoff. Per target root the
  error is the best-matching pencil root (Krylov-favoring convention);
  state preparation is costed as one first-order Trotter step of H per
  grid interval via the same graph-span oracle, so a dimension-K basis
  costs ``c_step * K(K-1)/2``. The best-per-K envelope over the dt grid is
  reported.

Memory profile: one dense Hamiltonian at a time (max 16 MB at nph7),
processed sequentially. Statevector diagnostics; never feeds controller
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

from pipelines.exact_bench.paper_iii_qse_paper_i_convention_sweep import (
    PAPER_I_REGIMES,
    _LINEAR_RESPONSE_FAMILIES,
    _build_regime_pool,
    _element_family,
    _num_qubits,
)
from pipelines.exact_bench.paper_iii_qse_peierls_pilot import (
    _FIXED_CLASS as PEIERLS_FIXED_CLASS,
)
from pipelines.exact_bench.paper_iii_qse_peierls_pilot import (
    _NQ as PEIERLS_NQ,
)
from pipelines.exact_bench.paper_iii_qse_peierls_pilot import (
    PEIERLS_REGIMES,
    build_peierls_hamiltonian,
    build_peierls_pool,
)
from pipelines.exact_bench.paper_iii_qse_peierls_pilot import (
    _sector_spectrum as peierls_sector_spectrum,
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
from pipelines.qse_spectra.core import (
    QSEBasisVectorPolicy,
    compute_qse_spectra,
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

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "output/diagnostics/paper_iii_comparator_matrix_20260819_v1/comparator_matrix.json"
)
_Q0_POLICY = QSEBasisVectorPolicy(
    reference_projection="q0", basis_vector_normalization="raw_projected"
)
_TARGET_ROOTS = 6
_DT_GRID = (0.25, 0.5)
_MAX_KRYLOV = 12
_PENCIL_CUTOFF = 1.0e-12


def _annotate_2q(elements: Sequence[Any], *, nq: int) -> list[float]:
    rows = annotate_basis_with_compiled_costs(
        elements,
        num_qubits=nq,
        oracle_kind=ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
        cost_weights=resolve_cost_weights_preset("two_qubit_only_v1"),
    )
    return [float(row.estimate.c_hat_2q) for row in rows]


def _krylov_arm(
    dense: np.ndarray, ground: np.ndarray, references: Sequence[float], *, step_2q: float
) -> dict[str, Any]:
    energies, vectors = np.linalg.eigh(dense)
    # The exact sector ground state has an identically zero Hamiltonian
    # residual, so the kick is a seeded random vector in the ground state's
    # support sector, orthogonalized against it: it overlaps every target
    # eigenstate with probability one, which is maximally generous to the
    # Krylov construction (no symmetry sector is invisible to the kick).
    rng = np.random.default_rng(20260819)
    support = np.abs(ground) > 0.0
    source = np.zeros_like(ground)
    source[support] = rng.normal(size=int(support.sum())) + 1j * rng.normal(
        size=int(support.sum())
    )
    source = source - complex(np.vdot(ground, source)) * ground
    norm = float(np.linalg.norm(source))
    if norm <= 1.0e-14:
        return {"status": "zero_kick"}
    source = source / norm
    amplitudes = vectors.conj().T @ source

    rows: list[dict[str, Any]] = []
    for dt in _DT_GRID:
        states = [
            vectors @ (np.exp(-1j * energies * float(dt) * k) * amplitudes)
            for k in range(_MAX_KRYLOV)
        ]
        for dimension in range(2, _MAX_KRYLOV + 1):
            block = states[:dimension]
            overlap = np.array([[np.vdot(a, b) for b in block] for a in block])
            ham = np.array([[np.vdot(a, dense @ b) for b in block] for a in block])
            overlap = 0.5 * (overlap + overlap.conj().T)
            ham = 0.5 * (ham + ham.conj().T)
            eigvals, eigvecs = np.linalg.eigh(overlap)
            retained = eigvals > _PENCIL_CUTOFF * float(max(eigvals.max(), 0.0))
            if int(retained.sum()) < 1:
                continue
            transform = eigvecs[:, retained] / np.sqrt(eigvals[retained])
            reduced = transform.conj().T @ ham @ transform
            roots = np.sort(np.linalg.eigvalsh(0.5 * (reduced + reduced.conj().T)))
            per_root = [
                float(np.min(np.abs(roots - float(reference)))) for reference in references
            ]
            steps = dimension * (dimension - 1) // 2
            rows.append(
                {
                    "dt": float(dt),
                    "K": int(dimension),
                    "cum_2q": float(step_2q * steps),
                    "root_abs_errors": per_root,
                    "max_root_abs_error": max(per_root),
                }
            )
    envelope: list[dict[str, Any]] = []
    best = float("inf")
    for row in sorted(rows, key=lambda item: (item["cum_2q"], item["max_root_abs_error"])):
        if row["max_root_abs_error"] < best:
            best = row["max_root_abs_error"]
            envelope.append(row)
    return {
        "trotter_step_2q": float(step_2q),
        "convention": "seeded_random_sector_kick_best_matching_root_krylov_favoring",
        "rows": rows,
        "best_per_cost_envelope": envelope,
    }


def _selection_arm(
    basis: Sequence[Any],
    hamiltonian: Any,
    ground: np.ndarray,
    costs: Sequence[float],
    references: Sequence[float],
    *,
    overrides: dict[str, Any],
    with_exchange: bool,
) -> dict[str, Any]:
    """Run a selection arm and score it on the shared reference window."""

    kwargs: dict[str, Any] = {
        "mode": "geometry_selected",
        "max_records": len(basis),
        "geometry_target_roots": _TARGET_ROOTS,
        "geometry_cost_discount_alpha": 1.0,
        "geometry_residual_stop": 1.0e-3,
    }
    kwargs.update(overrides)
    config = StaticRecordSelectionConfig(**kwargs)
    selection = select_static_qse_records(
        basis,
        config=config,
        hamiltonian=hamiltonian,
        prepared_state=ground,
        basis_vector_policy=_Q0_POLICY,
        compiled_costs=tuple(costs),
    )
    indices = list(selection.selected_original_indices)
    if with_exchange:
        exchange = run_qse_exchange_maintenance(
            basis,
            indices,
            tuple(costs),
            hamiltonian=hamiltonian,
            prepared_state=ground,
            basis_vector_policy=_Q0_POLICY,
            config=QSEExchangeConfig(
                max_rounds=30, target_root_count=_TARGET_ROOTS, insertion_shortlist_size=16
            ),
        )
        indices = list(exchange.final_indices)
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
    stop = selection.geometry_stop or {}
    return {
        "support_size": len(indices),
        "total_2q": float(sum(float(costs[int(i)]) for i in indices)),
        "stop_reason": stop.get("stop_reason"),
        "root_abs_errors": errors,
        "max_root_abs_error": max(finite) if finite else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    cases_payload: dict[str, Any] = {}
    cases: list[tuple[str, str, dict[str, Any]]] = []
    for regime, u, g_ep, n_ph_max in PAPER_I_REGIMES:
        cases.append((regime, "hh", {"u": u, "g_ep": g_ep, "n_ph_max": n_ph_max}))
    for regime, u, g_ep in PEIERLS_REGIMES:
        cases.append((regime, "peierls", {"u": u, "g_ep": g_ep}))

    for case_name, family, params in cases:
        if family == "hh":
            nq = _num_qubits(params["n_ph_max"])
            hamiltonian, basis, _meta = _build_regime_pool(**params)
            dense = _dense_hamiltonian(hamiltonian, 1 << nq)
            ground, spectrum = _sector_spectrum(dense, count=_TARGET_ROOTS + 1)
            fixed_indices = [
                index
                for index, element in enumerate(basis)
                if _element_family(element.name) in _LINEAR_RESPONSE_FAMILIES
            ]
        else:
            nq = PEIERLS_NQ
            hamiltonian = build_peierls_hamiltonian(u=params["u"], g_ep=params["g_ep"])
            basis = build_peierls_pool()
            dense = _dense_hamiltonian(hamiltonian, 1 << nq)
            ground, spectrum = peierls_sector_spectrum(dense, count=_TARGET_ROOTS + 1)
            fixed_indices = [
                index for index, element in enumerate(basis) if element.name in PEIERLS_FIXED_CLASS
            ]
        references = spectrum[1 : _TARGET_ROOTS + 1]
        costs = _annotate_2q(basis, nq=nq)

        fixed_result = compute_qse_spectra(
            hamiltonian,
            ground,
            tuple(basis[index] for index in fixed_indices),
            basis_vector_policy=_Q0_POLICY,
        )
        fixed_energies = np.asarray(fixed_result.eigenvalues, dtype=float).reshape(-1)
        fixed_errors = [
            abs(float(fixed_energies[root]) - reference) if root < fixed_energies.size else None
            for root, reference in enumerate(references)
        ]
        step_2q = _annotate_2q(
            [polynomial_basis_element(hamiltonian, name="first_order_trotter_step")], nq=nq
        )[0]

        cases_payload[case_name] = {
            "family": family,
            **params,
            "reference_excitations": references,
            "fixed_linear_response": {
                "class_size": len(fixed_indices),
                "total_2q": float(sum(costs[index] for index in fixed_indices)),
                "root_abs_errors": fixed_errors,
                "max_root_abs_error": max(
                    (error for error in fixed_errors if error is not None), default=None
                ),
            },
            "krylov": _krylov_arm(dense, ground, references, step_2q=step_2q),
            "selected_plus_exchange": _selection_arm(
                basis, hamiltonian, ground, costs, references,
                overrides={}, with_exchange=True,
            ),
        }
        record = cases_payload[case_name]
        krylov_best = (
            record["krylov"]["best_per_cost_envelope"][-1]
            if record["krylov"].get("best_per_cost_envelope")
            else None
        )
        print(
            f"{case_name:<22} fixed: {record['fixed_linear_response']['max_root_abs_error']:.1e}"
            f"@{record['fixed_linear_response']['total_2q']:.0f}2Q   "
            + (
                f"krylov best: {krylov_best['max_root_abs_error']:.1e}@{krylov_best['cum_2q']:.0f}2Q (K={krylov_best['K']})"
                if krylov_best
                else "krylov: n/a"
            )
        )
        del dense

    payload = {
        "schema_version": "paper_iii_qse_comparator_matrix_v1",
        "policy": "diagnostic_only_comparator_matrix",
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "post_run_diagnostic_only": True,
        },
        "target_roots": _TARGET_ROOTS,
        "cost_weights_preset": "two_qubit_only_v1",
        "cases": cases_payload,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\noutput_json: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
