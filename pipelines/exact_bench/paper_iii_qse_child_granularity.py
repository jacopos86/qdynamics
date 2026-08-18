#!/usr/bin/env python3
"""Paper III child-singleton granularity study for QSE response bases.

Paper II fixes the support atom at the Pauli/poly-child coordinate; the
stored Paper III pool is macro records (whole multi-term polynomials). This
driver decomposes the 158 macro records into deduplicated single-Pauli
children and answers three questions per regime:

1. **Span**: does the child manifold (span superset of the macro manifold)
   improve the manifold-limit error, especially where the macro pool is
   truncation-limited at nph3 (weak_strong)?
2. **Sector purity**: individual children break fermion-number conservation,
   so child images leave the (1,1) sector; the unprojected child pencil is
   shown to produce contaminated roots (below the true sector excitation),
   and the exact sector projector restores validity. Macro records conserve
   number by construction, so this requirement is specific to child
   granularity.
3. **Frontier granularity**: a cost-discounted greedy selection over
   children (score mirroring the geometry alpha=1 rule, compiled-2Q costs
   per child) versus the macro-level frontier at matched budgets.

All solves are statevector pencils with exact (1,1)-sector references
(sector-restricted eigenproblem — never expectation-filtered). Compiled
costs use the Marrakesh graph-span oracle under ``two_qubit_only_v1``.
Diagnostic evidence driver; never feeds controller decisions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.paper_iii_qse_regime_frontier_sweep import (
    REGIMES,
    _dense_hamiltonian,
    _sector_reference,
    _settings_payload,
)
from pipelines.qse_spectra.compiled_costs import (
    ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
    annotate_basis_with_compiled_costs,
    resolve_cost_weights_preset,
)
from pipelines.qse_spectra.core import pauli_string_basis_element
from pipelines.qse_spectra.io import load_operator_basis_json, load_polynomial_json

GOLDEN_QSE_RESULT = (
    REPO_ROOT / "output/diagnostics/paper_iii_hh_advisor_demo_20260802_a005/qse_result.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "output/diagnostics/paper_iii_child_granularity_20260818_v1/child_granularity_summary.json"
)
_NQ = 8
_FERMION_QUBITS_UP = (0, 1)
_FERMION_QUBITS_DN = (2, 3)
_STUDY_REGIMES = ("demo_weak", "weak_strong", "strong_strong")
_PENCIL_CUTOFF = 1.0e-12


def build_child_pool(macro_basis: Sequence[Any]) -> tuple[list[Any], dict[str, Any]]:
    """Decompose macro records into deduplicated single-Pauli child elements."""

    children: list[Any] = []
    seen: dict[str, int] = {}
    parent_counts: dict[str, int] = {}
    for element in macro_basis:
        if element.kind == "pauli_string":
            labels = [str(element.pauli_label_exyz)]
        else:
            labels = [str(term.pw2strng()) for term in element.polynomial.return_polynomial()]
        for label in labels:
            if label in seen:
                parent_counts[label] += 1
                continue
            seen[label] = len(children)
            parent_counts[label] = 1
            children.append(
                pauli_string_basis_element(
                    label,
                    nq=_NQ,
                    name=f"child[{label}]",
                    metadata={"source": "macro_decomposition_v1"},
                )
            )
    provenance = {
        "macro_count": len(macro_basis),
        "child_count": len(children),
        "max_parent_multiplicity": max(parent_counts.values()) if parent_counts else 0,
    }
    return children, provenance


def _pauli_label_matrix_free_apply(label: str, psi: np.ndarray) -> np.ndarray:
    from pipelines.qse_spectra.core import _apply_pauli_label

    return _apply_pauli_label(label, psi, nq=_NQ, pauli_action_cache={})


def _sector_mask(dim: int) -> np.ndarray:
    occ_up = np.array([sum((i >> q) & 1 for q in _FERMION_QUBITS_UP) for i in range(dim)])
    occ_dn = np.array([sum((i >> q) & 1 for q in _FERMION_QUBITS_DN) for i in range(dim)])
    return (occ_up == 1) & (occ_dn == 1)


def _pencil_root0(
    vectors: Sequence[np.ndarray], dense: np.ndarray
) -> tuple[float | None, int]:
    block = [np.asarray(v, dtype=complex).reshape(-1) for v in vectors]
    block = [v for v in block if float(np.linalg.norm(v)) > 1.0e-12]
    if not block:
        return None, 0
    count = len(block)
    overlap = np.empty((count, count), dtype=complex)
    ham = np.empty((count, count), dtype=complex)
    h_block = [dense @ v for v in block]
    for i in range(count):
        for j in range(count):
            overlap[i, j] = complex(np.vdot(block[i], block[j]))
            ham[i, j] = complex(np.vdot(block[i], h_block[j]))
    overlap = 0.5 * (overlap + overlap.conj().T)
    ham = 0.5 * (ham + ham.conj().T)
    eigvals, eigvecs = np.linalg.eigh(overlap)
    retained = eigvals > _PENCIL_CUTOFF * float(max(eigvals.max(), 0.0))
    if not bool(retained.any()):
        return None, 0
    transform = eigvecs[:, retained] / np.sqrt(eigvals[retained])
    reduced = transform.conj().T @ ham @ transform
    roots = np.linalg.eigvalsh(0.5 * (reduced + reduced.conj().T))
    return float(np.min(roots)), int(retained.sum())


def _child_images(
    children: Sequence[Any],
    psi: np.ndarray,
    *,
    sector_project: bool,
    sector: np.ndarray,
) -> list[np.ndarray]:
    images = []
    for element in children:
        image = _pauli_label_matrix_free_apply(str(element.pauli_label_exyz), psi)
        image = image - complex(np.vdot(psi, image)) * psi  # q0 projection
        if sector_project:
            image = np.where(sector, image, 0.0)
            image = image - complex(np.vdot(psi, image)) * psi
        images.append(np.asarray(image, dtype=complex).reshape(-1))
    return images


def _greedy_child_selection(
    images: Sequence[np.ndarray],
    costs: Sequence[float],
    dense: np.ndarray,
    psi: np.ndarray,
    *,
    budget: int,
    alpha: float = 1.0,
) -> list[int]:
    """Cost-discounted greedy mirroring the geometry alpha rule at child level."""

    h_psi = dense @ psi
    residual = h_psi - complex(np.vdot(psi, h_psi)) * psi
    residual_norm = float(np.linalg.norm(residual))
    residual_hat = residual / residual_norm if residual_norm > 0 else None
    max_cost = max(max(float(c) for c in costs), 1.0)

    accepted_units: list[np.ndarray] = []
    selected: list[int] = []
    remaining = set(range(len(images)))
    while remaining and len(selected) < int(budget):
        best_score, best_index, best_unit = None, None, None
        for index in list(remaining):
            image = images[index]
            norm_sq = float(np.vdot(image, image).real)
            if norm_sq <= 1.0e-24:
                remaining.discard(index)
                continue
            projected = image.copy()
            for unit in accepted_units:
                projected -= complex(np.vdot(unit, projected)) * unit
            p_norm_sq = float(np.vdot(projected, projected).real)
            novelty = max(0.0, p_norm_sq / norm_sq)
            if novelty < 1.0e-12:
                remaining.discard(index)
                continue
            unit_vec = projected / math.sqrt(p_norm_sq)
            capture = (
                float(abs(complex(np.vdot(unit_vec, residual_hat))))
                if residual_hat is not None
                else 0.0
            )
            utility = 0.25 * novelty + capture
            score = utility / (1.0e-12 + float(costs[index]) / max_cost) ** float(alpha)
            if best_score is None or score > best_score:
                best_score, best_index, best_unit = score, index, unit_vec
        if best_index is None:
            break
        selected.append(best_index)
        remaining.discard(best_index)
        accepted_units.append(best_unit)
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budgets", type=int, nargs="+", default=[20, 40, 80])
    args = parser.parse_args(argv)

    macro_basis, _ = load_operator_basis_json(GOLDEN_QSE_RESULT, nq=_NQ)
    children, child_provenance = build_child_pool(macro_basis)
    child_rows = annotate_basis_with_compiled_costs(
        children,
        num_qubits=_NQ,
        oracle_kind=ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
        cost_weights=resolve_cost_weights_preset("two_qubit_only_v1"),
    )
    child_costs = [float(row.scalarized_canonical_cost) for row in child_rows]
    macro_rows = annotate_basis_with_compiled_costs(
        macro_basis,
        num_qubits=_NQ,
        oracle_kind=ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
        cost_weights=resolve_cost_weights_preset("two_qubit_only_v1"),
    )

    dim = 1 << _NQ
    sector = _sector_mask(dim)
    regimes_payload: dict[str, Any] = {}
    scratch = args.output_json.parent / "settings"
    scratch.mkdir(parents=True, exist_ok=True)
    regime_params = {name: (u, g) for name, u, g in REGIMES}
    for regime in _STUDY_REGIMES:
        u, g_ep = regime_params[regime]
        settings_path = scratch / f"hh_{regime}_settings.json"
        settings_path.write_text(
            json.dumps(_settings_payload(u, g_ep), indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        hamiltonian, _prov = load_polynomial_json(settings_path)
        dense = _dense_hamiltonian(hamiltonian, dim)
        ground, e0_exact, e1_exact = _sector_reference(dense)

        projected_images = _child_images(children, ground, sector_project=True, sector=sector)
        raw_images = _child_images(children, ground, sector_project=False, sector=sector)

        child_limit_root, child_rank = _pencil_root0(projected_images, dense)
        raw_limit_root, _raw_rank = _pencil_root0(raw_images, dense)

        macro_images = []
        from pipelines.qse_spectra.core import QSEBasisVectorPolicy, compute_qse_spectra

        macro_result = compute_qse_spectra(
            hamiltonian,
            ground,
            tuple(macro_basis),
            basis_vector_policy=QSEBasisVectorPolicy(
                reference_projection="q0", basis_vector_normalization="raw_projected"
            ),
        )
        macro_limit_root = float(np.asarray(macro_result.eigenvalues, dtype=float).reshape(-1)[0])

        frontier = []
        for budget in args.budgets:
            selected = _greedy_child_selection(
                projected_images, child_costs, dense, ground, budget=int(budget)
            )
            root, rank = _pencil_root0([projected_images[i] for i in selected], dense)
            frontier.append(
                {
                    "budget": int(budget),
                    "selected_count": len(selected),
                    "retained_rank": rank,
                    "cum_2q": float(sum(child_costs[i] for i in selected)),
                    "abs_err_E1": abs(root - e1_exact) if root is not None else None,
                }
            )

        regimes_payload[regime] = {
            "u": float(u),
            "g_ep": float(g_ep),
            "e1_exact_sector": float(e1_exact),
            "macro_manifold_abs_err_E1": abs(macro_limit_root - e1_exact),
            "child_manifold_abs_err_E1": (
                abs(child_limit_root - e1_exact) if child_limit_root is not None else None
            ),
            "child_manifold_retained_rank": child_rank,
            "unprojected_child_root0": raw_limit_root,
            "unprojected_contamination": (
                bool(raw_limit_root < e1_exact - 1.0e-6) if raw_limit_root is not None else None
            ),
            "child_frontier_alpha1": frontier,
        }
        print(
            f"{regime}: macro-limit {regimes_payload[regime]['macro_manifold_abs_err_E1']:.2e}  "
            f"child-limit {regimes_payload[regime]['child_manifold_abs_err_E1']:.2e}  "
            f"unprojected root0 {raw_limit_root:+.4f} vs E1 {e1_exact:+.4f} "
            f"(contaminated: {regimes_payload[regime]['unprojected_contamination']})"
        )
        for row in frontier:
            print(
                f"  child alpha1 k={row['budget']}: {row['abs_err_E1']:.2e}@{row['cum_2q']:.0f}2Q"
            )

    payload = {
        "schema_version": "paper_iii_qse_child_granularity_v1",
        "policy": "diagnostic_only_child_granularity_study",
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "post_run_diagnostic_only": True,
        },
        "child_pool": child_provenance,
        "cost_weights_preset": "two_qubit_only_v1",
        "sector_projection": "exact_(1,1)_diagonal_projector_plus_q0",
        "regimes": regimes_payload,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"output_json: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
