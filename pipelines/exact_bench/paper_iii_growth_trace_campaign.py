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
from pipelines.qse_spectra.adaptive_qse_benchmark import run_adaptive_qse_benchmark
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
    # The terminal prefix must always be evaluated: an order whose length is
    # not a multiple of the stride would otherwise drop its final record, and
    # that record is often the one completing the retained rank.
    steps = sorted({*range(stride, cap + 1, stride), cap})
    for k in steps:
        if k <= 0:
            continue
        prefix = [int(i) for i in order[:k]]
        if exchange_policy == "every_k":
            prefix = _exchange(problem, prefix)
            rows.append(_row(problem, prefix, k, exchanged=True))
        else:
            rows.append(_row(problem, prefix, k, exchanged=False))
    return rows


def trace_adaptive_qse(problem: Any, *, k_max: int) -> list[dict[str, Any]]:
    """Growth trace for the external adaptive-QSE (quantum Davidson) arm.

    Grown to the same dimension cap as every other arm and scored on the same
    exact references. It synthesizes directions rather than consuming the
    record alphabet, so its resources come from the arm's own costing
    convention, not from `problem.resource_triple`.
    """

    seed = [
        problem.basis[i] for i, e in enumerate(problem.basis)
        if _element_family(e.name) in _LINEAR_RESPONSE_FAMILIES
    ]
    # Charge each synthesized direction as one first-order Trotter step of H,
    # the same convention the Krylov arm uses (handoff spec). Charging a pool
    # record instead is wrong: record 0 is the identity and costs nothing.
    from pipelines.qse_spectra.compiled_costs import (
        ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
        annotate_basis_with_compiled_costs,
        resolve_cost_weights_preset,
    )
    from pipelines.qse_spectra.core import polynomial_basis_element

    step_rows = annotate_basis_with_compiled_costs(
        [polynomial_basis_element(problem.hamiltonian, name="first_order_trotter_step")],
        num_qubits=problem.num_qubits,
        oracle_kind=ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
        cost_weights=resolve_cost_weights_preset("two_qubit_only_v1"),
    )
    est = step_rows[0].estimate
    per_direction = {
        "n2q": float(est.c_hat_2q),
        "d2q": float(est.c_hat_d),
        "dc": float(est.c_hat_d) + float(est.c_hat_1q),
    }
    audit = run_adaptive_qse_benchmark(
        problem.hamiltonian,
        problem.ground,
        # The benchmark returns the lowest N Ritz values INCLUDING the ground
        # root, so request R+1 and compare the excitations root_energies[1:].
        target_roots=_TARGET_ROOTS + 1,
        eps_residual=1.0e-14,      # effectively run to the cap; the trace supplies C*
        max_dimension=int(k_max),
        seed_elements=seed,
        direction_resources=per_direction,
    )
    from pipelines.exact_bench.paper_iii_qse_measurement_cost import _polynomial_words

    # Estimator queries for a synthesized-basis pencil. This arm's basis
    # vectors are states, not O_i|psi0>, so its S/H elements are not Pauli-word
    # products and cannot be QWC-grouped: each of the d(d+1)/2 independent
    # elements needs an overlap estimation plus one per Hamiltonian Pauli term
    # (Hadamard-test style, no grouping reuse). The convention is deliberately
    # less favourable than the record arms' QWC cover and is declared as such.
    ham_term_count = len(_polynomial_words(problem.hamiltonian))

    def _adaptive_estimator(dimension: int) -> dict[str, Any]:
        pairs = int(dimension) * (int(dimension) + 1) // 2
        return {
            "pair_count": pairs,
            "queries": int(pairs * (1 + ham_term_count)),
            "hamiltonian_pauli_terms": int(ham_term_count),
            "convention": "synthesized_basis_pencil_no_qwc_reuse",
        }

    refs = problem.references
    rows: list[dict[str, Any]] = []
    for it in audit["iterations"]:
        all_roots = it.get("root_energies") or []
        roots = list(all_roots[1:])          # drop the ground root
        err = (
            max(abs(float(roots[r]) - float(ref)) for r, ref in enumerate(refs))
            if len(roots) >= len(refs) else None
        )
        res = it.get("resources") or {}
        rows.append(
            {
                "k": int(it["dimension"]),
                "support_size": int(it["dimension"]),
                "exchanged": False,
                "retained_rank": int(it["retained_rank"]),
                "max_root_abs_error": err,
                "n2q": float(res.get("n2q", 0.0)),
                "d2q": float(res.get("d2q", 0.0)),
                "dc": float(res.get("dc", 0.0)),
                "root_energies": roots,
                "estimator": _adaptive_estimator(int(it["dimension"])),
            }
        )
    return rows


def trace_krylov(problem: Any, *, k_max: int) -> list[dict[str, Any]]:
    """Growth trace for real-time Krylov, scored on the same references.

    States exp(-i H k dt) from a seeded random sector kick orthogonal to the
    reference (the exact reference has an identically zero Hamiltonian
    residual, so a residual kick is unusable). Per dimension the pencil is
    solved with the shared cutoff and each target root is scored by its
    best-matching pencil root -- the Krylov-favouring convention. State
    preparation is charged one first-order Trotter step of H per interval,
    the same costing the adaptive method uses.
    """

    from pipelines.qse_spectra.compiled_costs import (
        ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
        annotate_basis_with_compiled_costs,
        resolve_cost_weights_preset,
    )
    from pipelines.qse_spectra.core import polynomial_basis_element
    from pipelines.exact_bench.paper_iii_qse_measurement_cost import _polynomial_words

    step = annotate_basis_with_compiled_costs(
        [polynomial_basis_element(problem.hamiltonian, name="first_order_trotter_step")],
        num_qubits=problem.num_qubits,
        oracle_kind=ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
        cost_weights=resolve_cost_weights_preset("two_qubit_only_v1"),
    )[0].estimate
    ham_terms = len(_polynomial_words(problem.hamiltonian))

    dim = 1 << int(problem.num_qubits)
    dense = np.zeros((dim, dim), dtype=complex)
    from pipelines.exact_bench.paper_iii_qse_regime_frontier_sweep import _dense_hamiltonian

    dense = _dense_hamiltonian(problem.hamiltonian, dim)
    energies, vectors = np.linalg.eigh(dense)
    ground = np.asarray(problem.ground, dtype=complex)
    rng = np.random.default_rng(20260826)
    support = np.abs(ground) > 0.0
    source = np.zeros_like(ground)
    n = int(support.sum())
    source[support] = rng.normal(size=n) + 1j * rng.normal(size=n)
    source = source - complex(np.vdot(ground, source)) * ground
    norm = float(np.linalg.norm(source))
    if norm <= 1.0e-14:
        return []
    amplitudes = vectors.conj().T @ (source / norm)
    refs = problem.references

    best: dict[int, dict[str, Any]] = {}
    for dt in (0.25, 0.5):
        states = [
            vectors @ (np.exp(-1j * energies * float(dt) * k) * amplitudes)
            for k in range(int(k_max) + 1)
        ]
        for d in range(2, int(k_max) + 1):
            block = states[:d]
            S = np.array([[np.vdot(a, b) for b in block] for a in block])
            M = np.array([[np.vdot(a, dense @ b) for b in block] for a in block])
            S = 0.5 * (S + S.conj().T)
            M = 0.5 * (M + M.conj().T)
            w, U = np.linalg.eigh(S)
            keep = w > 1.0e-12 * float(max(w.max(), 0.0))
            if int(keep.sum()) < 1:
                continue
            X = U[:, keep] / np.sqrt(w[keep])
            red = X.conj().T @ M @ X
            roots = np.sort(np.linalg.eigvalsh(0.5 * (red + red.conj().T)))
            err = max(float(np.min(np.abs(roots - float(r)))) for r in refs)
            steps_charged = d * (d - 1) // 2
            pairs = d * (d + 1) // 2
            row = {
                "k": int(d),
                "support_size": int(d),
                "exchanged": False,
                "retained_rank": int(keep.sum()),
                "max_root_abs_error": err,
                "n2q": float(step.c_hat_2q) * steps_charged,
                "d2q": float(step.c_hat_d) * steps_charged,
                "dc": (float(step.c_hat_d) + float(step.c_hat_1q)) * steps_charged,
                "dt": float(dt),
                "estimator": {
                    "pair_count": pairs,
                    "queries": int(pairs * (1 + ham_terms)),
                    "hamiltonian_pauli_terms": int(ham_terms),
                    "convention": "synthesized_basis_pencil_no_qwc_reuse",
                },
            }
            prev = best.get(d)
            if prev is None or err < prev["max_root_abs_error"]:
                best[d] = row
    del dense
    return [best[d] for d in sorted(best)]


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
        return {"status": STATUS_REACHED, "crossing": dict(best)}  # carries row estimator if present
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

        arms["adaptive_qse"] = {
            "trace": trace_adaptive_qse(problem, k_max=int(args.k_max)),
            "note": "external benchmark; synthesizes directions, does not consume the alphabet",
        }
        best_ad = min(
            (r for r in arms["adaptive_qse"]["trace"] if r["max_root_abs_error"] is not None),
            key=lambda r: r["max_root_abs_error"], default=None,
        )
        if best_ad:
            print(f"   {'adaptive_qse':<22} terminal {best_ad['max_root_abs_error']:.1e} "
                  f"@k={best_ad['k']} N2q={best_ad['n2q']:.0f}", flush=True)

        arms["krylov"] = {
            "trace": trace_krylov(problem, k_max=min(int(args.k_max), 14)),
            "note": "external benchmark; different construction family, does not consume the alphabet",
        }
        best_kr = min(
            (r for r in arms["krylov"]["trace"] if r["max_root_abs_error"] is not None),
            key=lambda r: r["max_root_abs_error"], default=None,
        )
        if best_kr:
            print(f"   {'krylov':<22} terminal {best_kr['max_root_abs_error']:.1e} "
                  f"@k={best_kr['k']} N2q={best_kr['n2q']:.0f}", flush=True)

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
        }  # adaptive_qse absent here: its estimator rides on its own trace rows
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
                est = problem.estimator_cost(idx)
                est["convention"] = "record_pencil_qwc_basis_cover"
                est["queries"] = int(est["qwc_groups"])
                cell["crossing"]["estimator"] = est

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
