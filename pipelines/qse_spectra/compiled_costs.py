#!/usr/bin/env python3
"""Qiskit compiled-cost annotation for QSE basis elements (Paper III Phase 1).

Implements the Phase 1 contract of
``prompt-exports/paper_iii_qse_qiskit_cost_integration_and_route_repair_spec_20260817.md``
(Documents clone): every QSE basis element is priced by the Paper I backend
compile machinery, with no behavior change to selection — costs are
annotation and manifest columns only.

Each :class:`~pipelines.qse_spectra.core.QSEBasisElement` maps to one
:class:`~src.quantum.vqe_latex_python_pairs.AnsatzTerm` (one excitation
direction = one generator). Costs come from the Paper I oracles in
``pipelines/static_adapt/hh_backend_compile_oracle.py``:

- ``marrakesh_graph_span_v1``: the analytic FakeMarrakesh coupling-graph
  span estimator (no transpilation; fast, deterministic);
- ``backend_transpile_single_v1``: full Qiskit transpilation against the
  resolved backend (``transpile_single_v1`` oracle mode).

Every row carries the oracle's :class:`CompileCostEstimate` with its
``hardware_cost_source``/``source_mode`` provenance, plus the scalarized
canonical Paper I cost ``sum_k lambda_k * c_hat_k`` under
``PAPER_I_CANONICAL_COST_WEIGHTS``. The shot component (``lambda_shot``) has
no compile-side coordinate and is recorded as not annotated rather than
silently folded in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.qse_spectra.core import QSEBasisElement
from pipelines.scaffold.hh_continuation_types import CompileCostEstimate
from pipelines.static_adapt.hh_backend_compile_oracle import (
    BackendCompileConfig,
    BackendCompileOracle,
    MarrakeshGraphSpanCostOracle,
)
from pipelines.static_adapt.paper_i_config import (
    PAPER_I_CANONICAL_COST_WEIGHTS,
    PaperICostWeights,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm

QSE_COMPILED_COSTS_SCHEMA_VERSION = "qse_compiled_costs_v1"
ORACLE_KIND_MARRAKESH_GRAPH_SPAN = "marrakesh_graph_span_v1"
ORACLE_KIND_BACKEND_TRANSPILE = "backend_transpile_single_v1"
_ORACLE_KINDS = (ORACLE_KIND_MARRAKESH_GRAPH_SPAN, ORACLE_KIND_BACKEND_TRANSPILE)


def qse_basis_element_to_ansatz_term(element: QSEBasisElement) -> AnsatzTerm:
    """Map one QSE basis element to the Paper I ``AnsatzTerm`` contract."""

    if element.kind == "pauli_string":
        label = str(element.pauli_label_exyz)
        if not label:
            raise ValueError(f"QSE basis element {element.name!r} has no Pauli label.")
        poly = PauliPolynomial("JW")
        poly.add_term(PauliTerm(len(label), ps=label, pc=1.0))
        return AnsatzTerm(label=str(element.name), polynomial=poly)
    if element.kind == "pauli_polynomial":
        if element.polynomial is None:
            raise ValueError(f"QSE basis element {element.name!r} is missing its polynomial.")
        return AnsatzTerm(label=str(element.name), polynomial=element.polynomial)
    raise ValueError(f"Unsupported QSE basis kind {element.kind!r} for cost annotation.")


def qse_basis_elements_to_ansatz_terms(
    elements: Sequence[QSEBasisElement],
) -> tuple[AnsatzTerm, ...]:
    return tuple(qse_basis_element_to_ansatz_term(element) for element in elements)


@dataclass(frozen=True)
class QSECompiledCostRow:
    """Compiled-cost annotation for one QSE basis element."""

    basis_index: int
    name: str
    kind: str
    estimate: CompileCostEstimate
    cost_components: dict[str, float]
    scalarized_canonical_cost: float
    cumulative_scalarized_cost: float
    hardware_cost_source: str
    source_mode: str
    metadata: dict[str, Any] = field(default_factory=dict)


def scalarize_compile_cost(
    estimate: CompileCostEstimate,
    *,
    cost_weights: PaperICostWeights | None = None,
) -> tuple[float, dict[str, float]]:
    """Return ``sum_k lambda_k * c_hat_k`` and the per-component terms.

    The shot weight has no compile-side coordinate; it is excluded from the
    scalarization and callers must not treat this scalar as shot-inclusive.
    """

    weights = PAPER_I_CANONICAL_COST_WEIGHTS if cost_weights is None else cost_weights
    lambdas = weights.as_lambda_dict()
    components = {
        "2q": float(lambdas["2q"]) * float(estimate.c_hat_2q),
        "d": float(lambdas["d"]) * float(estimate.c_hat_d),
        "1q": float(lambdas["1q"]) * float(estimate.c_hat_1q),
        "theta": float(lambdas["theta"]) * float(estimate.c_hat_theta),
    }
    total = float(sum(components.values()))
    if not math.isfinite(total):
        raise ValueError("Scalarized compiled cost must be finite.")
    return total, components


def _build_oracle(
    *,
    oracle_kind: str,
    oracle_config: BackendCompileConfig | None,
    num_qubits: int,
    ref_state: np.ndarray | None,
) -> MarrakeshGraphSpanCostOracle | BackendCompileOracle:
    if oracle_kind == ORACLE_KIND_MARRAKESH_GRAPH_SPAN:
        config = oracle_config if oracle_config is not None else BackendCompileConfig()
        return MarrakeshGraphSpanCostOracle(
            config=config, num_qubits=int(num_qubits), ref_state=ref_state
        )
    if oracle_kind == ORACLE_KIND_BACKEND_TRANSPILE:
        config = (
            oracle_config
            if oracle_config is not None
            else BackendCompileConfig(
                mode="transpile_single_v1", requested_backend_name="FakeMarrakesh"
            )
        )
        return BackendCompileOracle(config=config, num_qubits=int(num_qubits), ref_state=ref_state)
    raise ValueError(f"oracle_kind must be one of {list(_ORACLE_KINDS)!r}; got {oracle_kind!r}.")


def annotate_basis_with_compiled_costs(
    elements: Sequence[QSEBasisElement],
    *,
    num_qubits: int,
    oracle_kind: str = ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
    oracle_config: BackendCompileConfig | None = None,
    ref_state: np.ndarray | None = None,
    base_ops: Sequence[AnsatzTerm] = (),
    cost_weights: PaperICostWeights | None = None,
) -> tuple[QSECompiledCostRow, ...]:
    """Annotate every basis element with its Paper I compiled cost.

    The oracle snapshots the (possibly empty) prepared-state ansatz once and
    prices each element as a tail insertion at the final circuit cut. Rows
    are returned in input order with a running cumulative scalarized cost.
    """

    oracle = _build_oracle(
        oracle_kind=str(oracle_kind),
        oracle_config=oracle_config,
        num_qubits=int(num_qubits),
        ref_state=ref_state,
    )
    base_terms = tuple(base_ops)
    snapshot = oracle.snapshot_base(base_terms)
    tail_position = len(base_terms)

    rows: list[QSECompiledCostRow] = []
    cumulative = 0.0
    for index, element in enumerate(elements):
        term = qse_basis_element_to_ansatz_term(element)
        estimate = oracle.estimate_insertion(
            snapshot, candidate_term=term, position_id=int(tail_position)
        )
        scalarized, components = scalarize_compile_cost(estimate, cost_weights=cost_weights)
        cumulative += scalarized
        rows.append(
            QSECompiledCostRow(
                basis_index=int(index),
                name=str(element.name),
                kind=str(element.kind),
                estimate=estimate,
                cost_components=components,
                scalarized_canonical_cost=float(scalarized),
                cumulative_scalarized_cost=float(cumulative),
                hardware_cost_source=str(estimate.hardware_cost_source),
                source_mode=str(estimate.source_mode),
                metadata=dict(element.metadata or {}),
            )
        )
    return tuple(rows)


QSE_ACCURACY_COST_FRONTIER_SCHEMA_VERSION = "qse_accuracy_cost_frontier_v1"


def build_accuracy_cost_frontier(
    selected_basis: Sequence[QSEBasisElement],
    selected_cost_rows: Sequence[QSECompiledCostRow],
    *,
    hamiltonian: Any,
    prepared_state: np.ndarray,
    qse_config: Any = None,
    basis_vector_policy: Any = None,
    transition_observables: Sequence[Any] = (),
    reference_energies: Sequence[float] | None = None,
    max_reported_roots: int = 8,
) -> dict[str, Any]:
    """Solve every admitted prefix of the selected basis and pair accuracy with cost.

    Row ``k`` reports the QSE solve over the first ``k`` selected elements
    together with the cumulative compiled cost of those elements. When
    ``reference_energies`` (exact reference, index-matched) is supplied, the
    per-root absolute errors are included. Prefixes whose solve fails (e.g.
    zero retained rank under q0 projection) are recorded explicitly rather
    than dropped. Diagnostic reporting only.
    """

    from pipelines.qse_spectra.core import compute_qse_spectra

    if len(selected_cost_rows) != len(selected_basis):
        raise ValueError(
            f"frontier requires one cost row per selected element; got "
            f"{len(selected_cost_rows)} rows for {len(selected_basis)} elements."
        )
    reference = (
        [float(value) for value in reference_energies] if reference_energies is not None else None
    )
    rows: list[dict[str, Any]] = []
    cumulative_scalarized = 0.0
    cumulative_c_hat_2q = 0.0
    for prefix_size in range(1, len(selected_basis) + 1):
        cost_row = selected_cost_rows[prefix_size - 1]
        cumulative_scalarized += float(cost_row.scalarized_canonical_cost)
        cumulative_c_hat_2q += float(cost_row.estimate.c_hat_2q)
        row: dict[str, Any] = {
            "prefix_size": int(prefix_size),
            "cumulative_scalarized_cost": float(cumulative_scalarized),
            "cumulative_c_hat_2q": float(cumulative_c_hat_2q),
        }
        try:
            result = compute_qse_spectra(
                hamiltonian,
                prepared_state,
                tuple(selected_basis[:prefix_size]),
                config=qse_config,
                basis_vector_policy=basis_vector_policy,
                transition_observables=tuple(transition_observables),
            )
        except ValueError as exc:
            row.update({"solve_status": "failed", "solve_error": str(exc)})
            rows.append(row)
            continue
        energies = [float(value) for value in np.asarray(result.eigenvalues, dtype=float).reshape(-1)]
        reported = energies[: int(max_reported_roots)]
        row.update(
            {
                "solve_status": "solved",
                "retained_rank": int(result.retained_rank),
                "eigenvalue_count": len(energies),
                "lowest_energy": float(energies[0]) if energies else None,
                "root_energies": reported,
            }
        )
        if reference is not None:
            row["root_abs_errors_vs_reference"] = [
                abs(energy - reference[index])
                for index, energy in enumerate(reported)
                if index < len(reference)
            ]
        rows.append(row)
    return {
        "schema_version": QSE_ACCURACY_COST_FRONTIER_SCHEMA_VERSION,
        "policy": "diagnostic_only_accuracy_cost_frontier",
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "post_run_diagnostic_only": True,
        },
        "reference_supplied": reference is not None,
        "rows": rows,
    }


def compiled_costs_manifest_payload(
    rows: Sequence[QSECompiledCostRow],
    *,
    oracle_kind: str,
    num_qubits: int,
    cost_weights: PaperICostWeights | None = None,
) -> dict[str, Any]:
    """Render annotation rows as an additive, diagnostic-only manifest payload."""

    weights = PAPER_I_CANONICAL_COST_WEIGHTS if cost_weights is None else cost_weights
    row_payloads: list[dict[str, Any]] = []
    for row in rows:
        row_payloads.append(
            {
                "basis_index": int(row.basis_index),
                "name": str(row.name),
                "kind": str(row.kind),
                "c_hat_2q": float(row.estimate.c_hat_2q),
                "c_hat_d": float(row.estimate.c_hat_d),
                "c_hat_1q": float(row.estimate.c_hat_1q),
                "c_hat_theta": float(row.estimate.c_hat_theta),
                "cost_components": dict(row.cost_components),
                "scalarized_canonical_cost": float(row.scalarized_canonical_cost),
                "cumulative_scalarized_cost": float(row.cumulative_scalarized_cost),
                "hardware_cost_source": str(row.hardware_cost_source),
                "source_mode": str(row.source_mode),
                "compile_gate_open": bool(row.estimate.compile_gate_open),
                "failure_reason": row.estimate.failure_reason,
            }
        )
    return {
        "schema_version": QSE_COMPILED_COSTS_SCHEMA_VERSION,
        "policy": "diagnostic_only_compiled_cost_annotation",
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "post_run_diagnostic_only": True,
        },
        "oracle_kind": str(oracle_kind),
        "num_qubits": int(num_qubits),
        "cost_weights": {str(key): float(value) for key, value in weights.as_lambda_dict().items()},
        "shot_component_annotated": False,
        "rows": row_payloads,
        "summary": {
            "row_count": len(row_payloads),
            "total_scalarized_cost": float(
                rows[-1].cumulative_scalarized_cost if rows else 0.0
            ),
            "hardware_cost_sources": sorted({str(row.hardware_cost_source) for row in rows}),
            "source_modes": sorted({str(row.source_mode) for row in rows}),
        },
    }
