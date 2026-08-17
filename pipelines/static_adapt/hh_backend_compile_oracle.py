#!/usr/bin/env python3
"""Backend-conditioned transpilation oracle for HH Phase 3 candidate scoring."""

from __future__ import annotations

from collections import deque
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from threading import Lock
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.hardcoded.adapt_circuit_execution import build_structural_ansatz_circuit
from pipelines.scaffold.hh_continuation_types import CompileCostEstimate
from pipelines.qiskit_backend_tools import (
    ResolvedBackendTarget,
    backend_coupling_graph_snapshot,
    compile_circuit_for_backend,
    compiled_gate_stats,
    load_static_fake_graph_backend,
    rank_compile_rows,
    resolve_backend_targets,
    safe_circuit_depth,
    snapshot_backend_target,
    undirected_coupling_edges,
)
from src.quantum.ansatz_parameterization import AnsatzParameterLayout
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


_DEFAULT_PREFERRED_FAKES = ("FakeMarrakesh", "FakeNighthawk", "FakeFez")
_TRANSPILE_SINGLE_MODES = ("transpile_single_v1", "incremental_prefix_suffix_v1")
_INCREMENTAL_PREFIX_SUFFIX_MODE = "incremental_prefix_suffix_v1"
_FULL_TRANSPILE_SOURCE = "backend_transpile_v1"
_INCREMENTAL_SOURCE = "backend_incremental_prefix_suffix_v1"
_REF_STATE_SENTINEL = object()

BACKEND_COMPILE_SCOPE_SHARED_ALL_PHASES_V1 = (
    "shared_backend_compile_cost_all_phases_v1"
)
BACKEND_COMPILE_SCOPE_PHASE3_QISKIT_ONLY_V1 = (
    "phase_i_phase_ii_marrakesh_graph_span_"
    "phase_iii_qiskit_transpile_v1"
)
BACKEND_COMPILE_SCOPE_PHASE2_PHASE3_QISKIT_ONLY_V1 = (
    "phase_i_proxy_phase_ii_phase_iii_qiskit_transpile_v1"
)
BACKEND_COMPILE_SCOPE_PHASE123_QISKIT_V1 = (
    "phase0_proxy_or_off_phase_i_phase_ii_phase_iii_qiskit_transpile_v1"
)
ONE_QUBIT_COORDINATE_PROXY_BASELINE_V1 = "proxy_baseline_v1"
ONE_QUBIT_COORDINATE_COMPILED_POSITIVE_DELTA_V1 = (
    "compiled_positive_delta_v1"
)


def backend_compile_scope_uses_qiskit_for_stage(
    scope: str,
    stage: str,
) -> bool:
    """Return whether a staged secondary oracle owns this scoring stage.

    ``full`` is the legacy combined Phase-II/III evaluator stage.  It belongs
    to the new Phase-I--III scope, while the older staged scopes retain their
    exact historical routing.
    """

    scope_key = str(scope).strip()
    stage_key = str(stage).strip().lower()
    if scope_key == BACKEND_COMPILE_SCOPE_PHASE123_QISKIT_V1:
        return stage_key in {"phase1", "phase2", "phase3", "full"}
    if scope_key == BACKEND_COMPILE_SCOPE_PHASE2_PHASE3_QISKIT_ONLY_V1:
        return stage_key in {"phase2", "phase3"}
    if scope_key == BACKEND_COMPILE_SCOPE_PHASE3_QISKIT_ONLY_V1:
        return stage_key == "phase3"
    return False


@dataclass(frozen=True)
class BackendCompileConfig:
    mode: str = "proxy"
    requested_backend_name: str | None = None
    requested_backend_shortlist: tuple[str, ...] = ()
    seed_transpiler: int = 7
    optimization_level: int = 1
    structure_theta_value: float = 1.0
    preferred_fake_backends: tuple[str, ...] = _DEFAULT_PREFERRED_FAKES
    shortlist_reduction_mode: str = "best_backend_in_shortlist_v1"
    penalty_version: str = "transpile_signed_burden_scalar_v2"
    reward_negative_deltas: bool = True
    allow_preferred_fallback: bool = True
    one_qubit_coordinate_policy: str = (
        ONE_QUBIT_COORDINATE_PROXY_BASELINE_V1
    )
    weight_2q: float = 1.0
    weight_depth: float = 0.1
    weight_size: float = 0.01

    def __post_init__(self) -> None:
        if str(self.one_qubit_coordinate_policy) not in {
            ONE_QUBIT_COORDINATE_PROXY_BASELINE_V1,
            ONE_QUBIT_COORDINATE_COMPILED_POSITIVE_DELTA_V1,
        }:
            raise ValueError("Unknown one-qubit compile coordinate policy.")
        for field_name in ("weight_2q", "weight_depth", "weight_size"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("backend compile cost weights must be finite and nonnegative.")


@dataclass(frozen=True)
class BackendCompileBaseSnapshot:
    base_ops: tuple[AnsatzTerm, ...]
    base_structure_key: str
    base_layout: AnsatzParameterLayout
    base_backend_rows: tuple[dict[str, Any], ...]
    logical_depth: int


MARRAKESH_GRAPH_SPAN_MODE = "marrakesh_graph_span_v1"
_MARRAKESH_GRAPH_SPAN_EMBEDDINGS_V1: dict[int, tuple[int, ...]] = {
    4: (0, 1, 2, 3),
    6: (0, 1, 2, 3, 4, 16),
    8: (0, 1, 2, 3, 4, 16, 5, 23),
    9: (0, 1, 2, 3, 4, 16, 5, 23, 6),
    10: (0, 1, 2, 3, 4, 16, 5, 23, 6, 22),
    12: (0, 1, 2, 3, 4, 16, 5, 23, 6, 22, 7, 21),
}
_PAULI_CHARS = {"e", "x", "y", "z"}


@dataclass(frozen=True)
class MarrakeshGraphSpanBaseSnapshot:
    base_ops: tuple[AnsatzTerm, ...]
    logical_depth: int
    num_qubits: int
    embedding: tuple[int, ...]
    backend_name: str
    graph_snapshot: dict[str, Any]
    base_absolute_cost: dict[str, float]


def marrakesh_logical_embedding_v1(num_qubits: int) -> tuple[int, ...]:
    nq = int(num_qubits)
    if nq not in _MARRAKESH_GRAPH_SPAN_EMBEDDINGS_V1:
        raise ValueError(f"unsupported_marrakesh_graph_span_embedding_v1_size:{nq}")
    return tuple(int(q) for q in _MARRAKESH_GRAPH_SPAN_EMBEDDINGS_V1[nq])


def _normalize_pauli_label_exyz(raw: Any, *, num_qubits: int) -> str:
    label = str(raw).strip().lower().replace("i", "e")
    if len(label) != int(num_qubits):
        raise ValueError(f"pauli_label_length_mismatch:{len(label)}!={int(num_qubits)}:{label}")
    invalid = sorted({ch for ch in label if ch not in _PAULI_CHARS})
    if invalid:
        raise ValueError(f"invalid_pauli_label:{label}:{''.join(invalid)}")
    return label


def marrakesh_pauli_support_from_label(label: str, *, num_qubits: int) -> tuple[int, ...]:
    normalized = _normalize_pauli_label_exyz(label, num_qubits=int(num_qubits))
    nq = int(num_qubits)
    return tuple(int(nq - 1 - idx) for idx, ch in enumerate(normalized) if ch != "e")


def _extract_marrakesh_pauli_terms(
    candidate_term: AnsatzTerm,
    *,
    num_qubits: int,
    coefficient_tolerance: float = 1.0e-12,
) -> tuple[dict[str, Any], ...]:
    polynomial = getattr(candidate_term, "polynomial", None)
    if polynomial is None or not hasattr(polynomial, "return_polynomial"):
        raise ValueError("candidate_missing_pauli_polynomial")
    terms = list(polynomial.return_polynomial())
    if not terms:
        raise ValueError("candidate_missing_pauli_polynomial")
    out: list[dict[str, Any]] = []
    for term in terms:
        coeff = complex(getattr(term, "p_coeff"))
        if abs(coeff) <= float(coefficient_tolerance):
            continue
        label = _normalize_pauli_label_exyz(term.pw2strng(), num_qubits=int(num_qubits))
        support = marrakesh_pauli_support_from_label(label, num_qubits=int(num_qubits))
        out.append(
            {
                "label": str(label),
                "coefficient_abs": float(abs(coeff)),
                "logical_support": [int(q) for q in support],
                "support_size": int(len(support)),
                "is_identity": bool(len(support) == 0),
            }
        )
    return tuple(out)


def _adjacency_from_edges(edges: Sequence[Sequence[int]]) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {}
    for raw_u, raw_v in edges:
        u = int(raw_u)
        v = int(raw_v)
        if u == v:
            continue
        adjacency.setdefault(u, set()).add(v)
        adjacency.setdefault(v, set()).add(u)
    return adjacency


def _bfs_distances(adjacency: Mapping[int, set[int]], source: int) -> dict[int, int]:
    src = int(source)
    distances = {src: 0}
    queue: deque[int] = deque([src])
    while queue:
        node = int(queue.popleft())
        for nbr in sorted(adjacency.get(node, set())):
            if int(nbr) in distances:
                continue
            distances[int(nbr)] = int(distances[node] + 1)
            queue.append(int(nbr))
    return distances


def marrakesh_graph_span_edges_for_support(
    physical_support: Sequence[int],
    *,
    coupling_edges: Sequence[Sequence[int]],
) -> int:
    terminals = tuple(dict.fromkeys(int(q) for q in physical_support))
    if len(terminals) == 0:
        return 0
    adjacency = _adjacency_from_edges(coupling_edges)
    for terminal in terminals:
        if int(terminal) not in adjacency:
            raise ValueError(f"marrakesh_graph_span_terminal_not_in_graph:{terminal}")
    if len(terminals) == 1:
        return 0
    if len(terminals) == 2:
        distances = _bfs_distances(adjacency, int(terminals[0]))
        if int(terminals[1]) not in distances:
            raise RuntimeError("marrakesh_graph_span_disconnected_support")
        return int(distances[int(terminals[1])])

    nodes = sorted(adjacency)
    n_terms = len(terminals)
    all_mask = (1 << n_terms) - 1
    inf = 10**12
    terminal_distances = {
        int(terminal): _bfs_distances(adjacency, int(terminal))
        for terminal in terminals
    }
    dp: dict[int, dict[int, int]] = {mask: {node: inf for node in nodes} for mask in range(1, all_mask + 1)}
    for term_idx, terminal in enumerate(terminals):
        mask = 1 << term_idx
        distances = terminal_distances[int(terminal)]
        for node in nodes:
            if node in distances:
                dp[mask][node] = int(distances[node])
    for mask in range(1, all_mask + 1):
        sub = (mask - 1) & mask
        while sub:
            other = mask ^ sub
            if other:
                for node in nodes:
                    merged = int(dp[sub][node] + dp[other][node])
                    if merged < dp[mask][node]:
                        dp[mask][node] = merged
            sub = (sub - 1) & mask
        # Metric closure relaxation. Repeating is cheap at Paper-I terminal counts and
        # avoids depending on a mutable priority queue implementation here.
        improved = True
        while improved:
            improved = False
            for u, nbrs in adjacency.items():
                base = dp[mask][int(u)]
                if base >= inf:
                    continue
                for v in nbrs:
                    cand = int(base + 1)
                    if cand < dp[mask][int(v)]:
                        dp[mask][int(v)] = cand
                        improved = True
    answer = min(dp[all_mask].values())
    if answer >= inf:
        raise RuntimeError("marrakesh_graph_span_disconnected_support")
    return int(answer)


def _fallback_one_qubit_template_count(pauli_terms: Sequence[Mapping[str, Any]]) -> float:
    total = 0.0
    for row in pauli_terms:
        label = str(row.get("label", ""))
        if int(row.get("support_size", 0)) <= 0:
            continue
        total += float(2 * label.count("x") + 4 * label.count("y") + 1)
    return float(total)


def _proxy_baseline_payload(proxy_baseline: CompileCostEstimate | None) -> dict[str, float] | None:
    if proxy_baseline is None:
        return None
    return {
        "new_pauli_actions": float(proxy_baseline.new_pauli_actions),
        "new_rotation_steps": float(proxy_baseline.new_rotation_steps),
        "position_shift_span": float(proxy_baseline.position_shift_span),
        "refit_active_count": float(proxy_baseline.refit_active_count),
        "proxy_total": float(proxy_baseline.proxy_total),
        "cx_proxy_total": float(proxy_baseline.cx_proxy_total),
        "sq_proxy_total": float(proxy_baseline.sq_proxy_total),
        "gate_proxy_total": float(proxy_baseline.gate_proxy_total),
        "max_pauli_weight": float(proxy_baseline.max_pauli_weight),
        "c_hat_2q": float(proxy_baseline.c_hat_2q),
        "c_hat_d": float(proxy_baseline.c_hat_d),
        "c_hat_1q": float(proxy_baseline.c_hat_1q),
        "c_hat_theta": float(proxy_baseline.c_hat_theta),
    }


class MarrakeshGraphSpanCostOracle:
    """Paper-I analytic graph-span estimator over the FakeMarrakesh coupling graph."""

    coefficient_tolerance: float = 1.0e-12

    def __init__(
        self,
        *,
        config: BackendCompileConfig,
        num_qubits: int,
        ref_state: np.ndarray | None = None,
    ) -> None:
        self.config = config
        self.num_qubits = int(num_qubits)
        self.ref_state = None if ref_state is None else np.asarray(ref_state, dtype=complex).reshape(-1)
        requested_backend_name = "FakeMarrakesh" if config.requested_backend_name in {None, ""} else str(config.requested_backend_name)
        if requested_backend_name != "FakeMarrakesh":
            raise ValueError("marrakesh_graph_span_v1 requires phase3_backend_name='FakeMarrakesh'.")
        if tuple(config.requested_backend_shortlist):
            raise ValueError("marrakesh_graph_span_v1 does not accept --phase3-backend-shortlist.")
        self.embedding = marrakesh_logical_embedding_v1(int(self.num_qubits))
        self.targets, self.resolution_audit = resolve_backend_targets(
            requested_names=("FakeMarrakesh",),
            preferred_fake_backends=("FakeMarrakesh",),
            allow_preferred_fallback=False,
            fallback_mode="single",
            allow_runtime_lookup=False,
        )
        if len(self.targets) != 1:
            try:
                backend_obj, resolved_name, fallback_meta = load_static_fake_graph_backend("FakeMarrakesh")
            except Exception:
                pass
            else:
                snapshot = snapshot_backend_target(backend_obj)
                snapshot["static_graph_backend"] = fallback_meta
                self.targets = (
                    ResolvedBackendTarget(
                        requested_name="FakeMarrakesh",
                        resolved_name=str(resolved_name),
                        resolution_kind="fake_static_graph_conf",
                        using_fake_backend=True,
                        backend_obj=backend_obj,
                        target_snapshot=dict(snapshot),
                    ),
                )
                self.resolution_audit = [
                    *list(self.resolution_audit),
                    {
                        "requested_name": "FakeMarrakesh",
                        "resolved_name": str(resolved_name),
                        "success": True,
                        "resolution_kind": "fake_static_graph_conf",
                        "using_fake_backend": True,
                        "runtime_lookup_attempted": False,
                        "runtime_error": None,
                        "fake_exact_attempted": "FakeMarrakesh",
                        "fallback_reason": "installed_fake_backend_class_unavailable_or_too_heavy_for_graph_span",
                        "error": None,
                        "target_snapshot": dict(snapshot),
                    },
                ]
        if len(self.targets) != 1:
            raise RuntimeError("marrakesh_graph_span_v1 could not resolve FakeMarrakesh.")
        target = self.targets[0]
        if str(target.resolved_name) != "FakeMarrakesh":
            raise RuntimeError(f"marrakesh_graph_span_v1 resolved unexpected backend {target.resolved_name!r}.")
        self.backend = target.backend_obj
        self.graph_snapshot = backend_coupling_graph_snapshot(self.backend)
        self.coupling_edges = undirected_coupling_edges(self.backend)
        physical_qubits = {int(q) for edge in self.coupling_edges for q in edge}
        invalid = [int(q) for q in self.embedding if int(q) not in physical_qubits]
        if invalid:
            raise ValueError(f"marrakesh_graph_span_embedding_invalid_physical_qubits:{invalid}")
        self.estimate_count = 0
        self._span_cache: dict[tuple[int, ...], int] = {}
        self._span_cache_lock = Lock()

    def _span_edges_for_physical_support(self, physical_support: Sequence[int]) -> int:
        key = tuple(sorted(dict.fromkeys(int(q) for q in physical_support)))
        with self._span_cache_lock:
            cached = self._span_cache.get(key)
        if cached is not None:
            return int(cached)
        span_edges = marrakesh_graph_span_edges_for_support(
            key,
            coupling_edges=self.coupling_edges,
        )
        with self._span_cache_lock:
            existing = self._span_cache.setdefault(key, int(span_edges))
        return int(existing)

    def _aggregation_mode(self) -> str:
        return "single_backend_graph_span"

    def _absolute_cost_for_ops(self, ops: Sequence[AnsatzTerm]) -> dict[str, float]:
        total_2q = 0.0
        total_d = 0.0
        total_1q = 0.0
        total_theta = 0.0
        for op in ops:
            rows = self._candidate_cost_rows(op)
            total_2q += float(sum(float(row["c_hat_2q_term"]) for row in rows))
            total_d += float(sum(float(row["c_hat_d_term"]) for row in rows))
            total_1q += _fallback_one_qubit_template_count(rows)
            if any(int(row.get("support_size", 0)) > 0 for row in rows):
                total_theta += 1.0
        return {
            "absolute_c_hat_2q": float(total_2q),
            "absolute_c_hat_d": float(total_d),
            "absolute_c_hat_1q": float(total_1q),
            "absolute_theta_count": float(total_theta),
        }

    def snapshot_base(self, ops: Sequence[AnsatzTerm]) -> MarrakeshGraphSpanBaseSnapshot:
        return MarrakeshGraphSpanBaseSnapshot(
            base_ops=tuple(ops),
            logical_depth=int(len(ops)),
            num_qubits=int(self.num_qubits),
            embedding=tuple(int(q) for q in self.embedding),
            backend_name="FakeMarrakesh",
            graph_snapshot=dict(self.graph_snapshot),
            base_absolute_cost=self._absolute_cost_for_ops(ops),
        )

    def _candidate_cost_rows(self, candidate_term: AnsatzTerm) -> tuple[dict[str, Any], ...]:
        extracted = _extract_marrakesh_pauli_terms(
            candidate_term,
            num_qubits=int(self.num_qubits),
            coefficient_tolerance=float(self.coefficient_tolerance),
        )
        rows: list[dict[str, Any]] = []
        for row in extracted:
            logical_support = [int(q) for q in row.get("logical_support", [])]
            physical_support = [int(self.embedding[int(q)]) for q in logical_support]
            support_size = int(len(logical_support))
            span_edges = self._span_edges_for_physical_support(physical_support)
            rows.append(
                {
                    **dict(row),
                    "physical_support": [int(q) for q in physical_support],
                    "span_edges": int(span_edges),
                    "c_hat_2q_term": float(2 * max(0, support_size - 1)),
                    "c_hat_d_term": float(2 * int(span_edges)),
                }
            )
        return tuple(rows)

    def estimate_insertion(
        self,
        snapshot: MarrakeshGraphSpanBaseSnapshot,
        *,
        candidate_term: AnsatzTerm,
        position_id: int,
        proxy_baseline: CompileCostEstimate | None = None,
    ) -> CompileCostEstimate:
        if int(snapshot.num_qubits) != int(self.num_qubits):
            raise ValueError("marrakesh_graph_span_snapshot_num_qubits_mismatch")
        rows = self._candidate_cost_rows(candidate_term)
        c_hat_2q = float(sum(float(row["c_hat_2q_term"]) for row in rows))
        c_hat_d = float(sum(float(row["c_hat_d_term"]) for row in rows))
        any_nonidentity = any(int(row.get("support_size", 0)) > 0 for row in rows)
        c_hat_1q = (
            float(proxy_baseline.c_hat_1q)
            if proxy_baseline is not None
            else _fallback_one_qubit_template_count(rows)
        )
        c_hat_theta = (
            float(proxy_baseline.c_hat_theta)
            if proxy_baseline is not None
            else (1.0 if bool(any_nonidentity) else 0.0)
        )
        penalty_total = float(float(self.config.weight_2q) * c_hat_2q + float(self.config.weight_depth) * c_hat_d)
        selected_row = {
            "schema": "marrakesh_graph_span_cost_v1",
            "source_mode": MARRAKESH_GRAPH_SPAN_MODE,
            "hardware_cost_source": MARRAKESH_GRAPH_SPAN_MODE,
            "no_transpile": True,
            "transpile_status": "not_run",
            "selected_backend_name": "FakeMarrakesh",
            "transpile_backend": "FakeMarrakesh",
            "resolution_kind": str(self.targets[0].resolution_kind),
            "using_fake_backend": bool(self.targets[0].using_fake_backend),
            "coupling_edge_count": int(self.graph_snapshot.get("coupling_edge_count", len(self.coupling_edges))),
            "embedding_logical_to_physical": [int(q) for q in self.embedding],
            "position_id": int(position_id),
            "num_qubits": int(self.num_qubits),
            "coefficient_tolerance": float(self.coefficient_tolerance),
            "c_hat_2q": float(c_hat_2q),
            "c_hat_d": float(c_hat_d),
            "c_hat_1q": float(c_hat_1q),
            "c_hat_theta": float(c_hat_theta),
            "penalty_total": float(penalty_total),
            "penalty_weight_2q": float(self.config.weight_2q),
            "penalty_weight_depth": float(self.config.weight_depth),
            "penalty_weight_size": float(self.config.weight_size),
            "pauli_terms": [dict(row) for row in rows],
            "telemetry_reason": (
                "all_terms_below_tolerance_or_identity"
                if not bool(any_nonidentity)
                else "ok"
            ),
            "aggregation_mode": self._aggregation_mode(),
            "target_backend_names": ["FakeMarrakesh"],
        }
        self.estimate_count += 1
        return CompileCostEstimate(
            new_pauli_actions=(0.0 if proxy_baseline is None else float(proxy_baseline.new_pauli_actions)),
            new_rotation_steps=(0.0 if proxy_baseline is None else float(proxy_baseline.new_rotation_steps)),
            position_shift_span=(0.0 if proxy_baseline is None else float(proxy_baseline.position_shift_span)),
            refit_active_count=(0.0 if proxy_baseline is None else float(proxy_baseline.refit_active_count)),
            proxy_total=(0.0 if proxy_baseline is None else float(proxy_baseline.proxy_total)),
            cx_proxy_total=(0.0 if proxy_baseline is None else float(proxy_baseline.cx_proxy_total)),
            sq_proxy_total=(0.0 if proxy_baseline is None else float(proxy_baseline.sq_proxy_total)),
            gate_proxy_total=(0.0 if proxy_baseline is None else float(proxy_baseline.gate_proxy_total)),
            max_pauli_weight=(0.0 if proxy_baseline is None else float(proxy_baseline.max_pauli_weight)),
            c_hat_2q=float(c_hat_2q),
            c_hat_d=float(c_hat_d),
            c_hat_1q=float(c_hat_1q),
            c_hat_theta=float(c_hat_theta),
            hardware_cost_source=MARRAKESH_GRAPH_SPAN_MODE,
            source_mode=MARRAKESH_GRAPH_SPAN_MODE,
            penalty_total=float(penalty_total),
            depth_surrogate=float(c_hat_d),
            compile_gate_open=True,
            failure_reason=None,
            selected_backend_name="FakeMarrakesh",
            selected_resolution_kind=str(self.targets[0].resolution_kind),
            aggregation_mode=self._aggregation_mode(),
            target_backend_names=["FakeMarrakesh"],
            successful_target_count=1,
            failed_target_count=0,
            raw_delta_compiled_count_2q=None,
            delta_compiled_count_2q=None,
            raw_delta_compiled_depth=None,
            delta_compiled_depth=None,
            raw_delta_compiled_depth_2q=None,
            delta_compiled_depth_2q=None,
            raw_delta_compiled_size=None,
            delta_compiled_size=None,
            delta_compiled_cx_count=None,
            delta_compiled_ecr_count=None,
            proxy_baseline=_proxy_baseline_payload(proxy_baseline),
            selected_backend_row=dict(selected_row),
        )

    def final_scaffold_summary(self, ops: Sequence[AnsatzTerm]) -> dict[str, Any]:
        absolute = self._absolute_cost_for_ops(ops)
        return {
            "schema": "marrakesh_graph_span_final_summary_v1",
            "no_transpile": True,
            **dict(absolute),
            "embedding_logical_to_physical": [int(q) for q in self.embedding],
            "coupling_edge_count": int(self.graph_snapshot.get("coupling_edge_count", len(self.coupling_edges))),
            "selected_backend": "FakeMarrakesh",
            "graph_snapshot": dict(self.graph_snapshot),
        }

    def cache_summary(self) -> dict[str, Any]:
        with self._span_cache_lock:
            span_cache_entries = int(len(self._span_cache))
        return {
            "estimate_count": int(self.estimate_count),
            "cache_entries": int(span_cache_entries),
            "span_cache_entries": int(span_cache_entries),
            "mode": MARRAKESH_GRAPH_SPAN_MODE,
        }


class BackendCompileOracle:
    def __init__(
        self,
        *,
        config: BackendCompileConfig,
        num_qubits: int,
        ref_state: np.ndarray | None,
    ) -> None:
        self.config = config
        self.num_qubits = int(num_qubits)
        self.ref_state = None if ref_state is None else np.asarray(ref_state, dtype=complex).reshape(-1)
        requested_names = [str(config.requested_backend_name)] if str(config.mode) in _TRANSPILE_SINGLE_MODES else list(config.requested_backend_shortlist)
        fallback_mode = "single" if str(config.mode) in _TRANSPILE_SINGLE_MODES else "shortlist"
        self.targets, self.resolution_audit = resolve_backend_targets(
            requested_names=requested_names,
            preferred_fake_backends=tuple(str(x) for x in config.preferred_fake_backends),
            allow_preferred_fallback=bool(config.allow_preferred_fallback),
            fallback_mode=str(fallback_mode),
        )
        self.stats_cache: dict[tuple[str, ...], dict[str, Any]] = {}
        self._cache_lock = Lock()
        self._cache_key_locks: dict[tuple[str, ...], Lock] = {}
        self.row_hits = 0
        self.row_misses = 0
        self.compile_failures = 0
        self.estimate_count = 0

    def _aggregation_mode(self) -> str:
        if str(self.config.mode) in _TRANSPILE_SINGLE_MODES:
            return "single_backend"
        return str(self.config.shortlist_reduction_mode)

    @staticmethod
    def _normalize_initial_layout(initial_layout: Sequence[int] | None) -> tuple[int, ...] | None:
        if initial_layout is None:
            return None
        return tuple(int(q) for q in initial_layout)

    @staticmethod
    def _state_hash(ref_state: np.ndarray | None) -> str:
        if ref_state is None:
            return "none"
        arr = np.asarray(ref_state, dtype=np.complex128).reshape(-1)
        return hashlib.sha256(arr.tobytes()).hexdigest()

    def _ref_state_hash(self) -> str:
        return self._state_hash(self.ref_state)

    def _structure_key(
        self,
        layout: AnsatzParameterLayout,
        *,
        ref_state_hash: str | None = None,
        initial_layout: Sequence[int] | None = None,
        segment_kind: str = "full",
    ) -> str:
        initial_layout_tuple = self._normalize_initial_layout(initial_layout)
        structural_layout = {
            "mode": str(layout.mode),
            "term_order": str(layout.term_order),
            "ignore_identity": bool(layout.ignore_identity),
            "coefficient_tolerance": float(layout.coefficient_tolerance),
            "blocks": [
                {
                    "runtime_count": int(block.runtime_count),
                    "runtime_terms_exyz": [
                        {
                            "pauli_exyz": str(spec.pauli_exyz),
                            "coeff_re": float(spec.coeff_real),
                            "nq": int(spec.nq),
                        }
                        for spec in block.terms
                    ],
                }
                for block in layout.blocks
            ],
        }
        payload = {
            "num_qubits": int(self.num_qubits),
            "ref_state_hash": self._ref_state_hash() if ref_state_hash is None else str(ref_state_hash),
            "structure_theta_value": float(self.config.structure_theta_value),
            "initial_layout": None if initial_layout_tuple is None else [int(q) for q in initial_layout_tuple],
            "segment_kind": str(segment_kind),
            "layout": structural_layout,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _compile_structure(
        self,
        *,
        structure_key: str | None,
        layout: AnsatzParameterLayout | None,
        ops: Sequence[AnsatzTerm],
        ref_state: Any = _REF_STATE_SENTINEL,
        initial_layout: Sequence[int] | None = None,
        initial_layout_by_backend: Mapping[str, Sequence[int] | None] | None = None,
        segment_kind: str = "full",
        cache_namespace: str | None = None,
    ) -> tuple[dict[str, Any], ...]:
        ref_state_use = self.ref_state if ref_state is _REF_STATE_SENTINEL else ref_state
        if ref_state_use is not None and not isinstance(ref_state_use, np.ndarray):
            ref_state_use = np.asarray(ref_state_use, dtype=complex).reshape(-1)
        layout_use, qc = build_structural_ansatz_circuit(
            ops,
            nq=int(self.num_qubits),
            ref_state=ref_state_use,
            structure_theta_value=float(self.config.structure_theta_value),
        )
        if layout is not None and self._structure_key(layout) != self._structure_key(layout_use):
            # Layout mismatch indicates a caller passed stale structural metadata.
            raise ValueError("compile_structure_layout_mismatch")
        rows: list[dict[str, Any]] = []
        ref_hash = self._state_hash(ref_state_use if isinstance(ref_state_use, np.ndarray) else None)
        for target in self.targets:
            target_initial_layout = (
                initial_layout_by_backend.get(str(target.resolved_name), initial_layout)
                if initial_layout_by_backend is not None
                else initial_layout
            )
            target_initial_layout_tuple = self._normalize_initial_layout(target_initial_layout)
            row_structure_key = (
                str(structure_key)
                if structure_key is not None and target_initial_layout_tuple is None and segment_kind == "full"
                else self._structure_key(
                    layout_use,
                    ref_state_hash=str(ref_hash),
                    initial_layout=target_initial_layout_tuple,
                    segment_kind=str(segment_kind),
                )
            )
            cache_key = (
                (str(row_structure_key), str(target.resolved_name))
                if cache_namespace is None
                else (
                    str(row_structure_key),
                    str(target.resolved_name),
                    str(cache_namespace),
                )
            )
            with self._cache_lock:
                cached = self.stats_cache.get(cache_key, None)
                if cached is not None:
                    self.row_hits += 1
                    rows.append(dict(cached))
                    continue
                cache_key_lock = self._cache_key_locks.setdefault(cache_key, Lock())
            with cache_key_lock:
                with self._cache_lock:
                    cached = self.stats_cache.get(cache_key, None)
                    if cached is not None:
                        self.row_hits += 1
                        rows.append(dict(cached))
                        continue
                    self.row_misses += 1
                row: dict[str, Any] = {
                    "structure_key": str(row_structure_key),
                    "compile_cache_namespace": (
                        None
                        if cache_namespace is None
                        else str(cache_namespace)
                    ),
                    "segment_kind": str(segment_kind),
                    "initial_layout": None if target_initial_layout_tuple is None else [int(q) for q in target_initial_layout_tuple],
                    "transpile_backend": str(target.resolved_name),
                    "requested_backend": str(target.requested_name),
                    "resolution_kind": str(target.resolution_kind),
                    "using_fake_backend": bool(target.using_fake_backend),
                    "target_snapshot": dict(getattr(target, "target_snapshot", {}) or {}),
                }
                try:
                    compile_kwargs: dict[str, Any] = {
                        "seed_transpiler": int(self.config.seed_transpiler),
                        "optimization_level": int(self.config.optimization_level),
                    }
                    if target_initial_layout_tuple is not None:
                        compile_kwargs["initial_layout"] = target_initial_layout_tuple
                    compiled_info = compile_circuit_for_backend(qc, target.backend_obj, **compile_kwargs)
                    compiled = compiled_info["compiled"]
                    row.update(
                        {
                            "transpile_status": "ok",
                            "compiled_depth": int(safe_circuit_depth(compiled)),
                            "compiled_size": int(compiled.size()),
                            "logical_to_physical": [int(x) for x in compiled_info.get("logical_to_physical", ())],
                            "compiled_num_qubits": int(compiled_info.get("compiled_num_qubits", compiled.num_qubits)),
                        }
                    )
                    row.update(dict(compiled_gate_stats(compiled)))
                    row["error"] = None
                except Exception as exc:
                    with self._cache_lock:
                        self.compile_failures += 1
                    row.update(
                        {
                            "transpile_status": "error",
                            "compiled_depth": None,
                            "compiled_size": None,
                            "compiled_count_2q": None,
                            "compiled_cx_count": None,
                            "compiled_ecr_count": None,
                            "compiled_op_counts": {},
                            "compiled_num_qubits": None,
                            "logical_to_physical": [],
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                with self._cache_lock:
                    self.stats_cache[cache_key] = dict(row)
                rows.append(dict(row))
        return tuple(rows)

    def snapshot_base(self, ops: Sequence[AnsatzTerm]) -> BackendCompileBaseSnapshot:
        layout, _qc = build_structural_ansatz_circuit(
            ops,
            nq=int(self.num_qubits),
            ref_state=self.ref_state,
            structure_theta_value=float(self.config.structure_theta_value),
        )
        structure_key = self._structure_key(layout)
        rows = self._compile_structure(structure_key=str(structure_key), layout=layout, ops=ops)
        return BackendCompileBaseSnapshot(
            base_ops=tuple(ops),
            base_structure_key=str(structure_key),
            base_layout=layout,
            base_backend_rows=tuple(rows),
            logical_depth=int(len(ops)),
        )

    @staticmethod
    def _proxy_baseline_dict(proxy_baseline: CompileCostEstimate | None) -> dict[str, float] | None:
        if proxy_baseline is None:
            return None
        return {
            "new_pauli_actions": float(proxy_baseline.new_pauli_actions),
            "new_rotation_steps": float(proxy_baseline.new_rotation_steps),
            "position_shift_span": float(proxy_baseline.position_shift_span),
            "refit_active_count": float(proxy_baseline.refit_active_count),
            "proxy_total": float(proxy_baseline.proxy_total),
            "cx_proxy_total": float(proxy_baseline.cx_proxy_total),
            "sq_proxy_total": float(proxy_baseline.sq_proxy_total),
            "gate_proxy_total": float(proxy_baseline.gate_proxy_total),
            "max_pauli_weight": float(proxy_baseline.max_pauli_weight),
            "c_hat_2q": float(proxy_baseline.c_hat_2q),
            "c_hat_d": float(proxy_baseline.c_hat_d),
            "c_hat_1q": float(proxy_baseline.c_hat_1q),
            "c_hat_theta": float(proxy_baseline.c_hat_theta),
        }

    def _estimate_from_rows(
        self,
        *,
        base_rows: Sequence[Mapping[str, Any]],
        trial_rows: Sequence[Mapping[str, Any]],
        proxy_baseline: CompileCostEstimate | None,
        source_mode: str = _FULL_TRANSPILE_SOURCE,
        hardware_cost_source: str = _FULL_TRANSPILE_SOURCE,
    ) -> CompileCostEstimate:
        rows: list[dict[str, Any]] = []
        base_map = {str(row.get("transpile_backend", "")): dict(row) for row in base_rows}
        for trial in trial_rows:
            trial_row = dict(trial)
            backend_name = str(trial_row.get("transpile_backend", ""))
            base_row = dict(base_map.get(backend_name, {}))
            if str(trial_row.get("transpile_status", "")) != "ok" or str(base_row.get("transpile_status", "")) != "ok":
                rows.append(
                    {
                        **trial_row,
                        "selected_backend_name": backend_name,
                        "transpile_status": "error",
                        "raw_delta_compiled_count_2q": None,
                        "delta_compiled_count_2q": None,
                        "raw_delta_compiled_count_1q": None,
                        "delta_compiled_count_1q": None,
                        "raw_delta_compiled_depth": None,
                        "delta_compiled_depth": None,
                        "raw_delta_compiled_depth_2q": None,
                        "delta_compiled_depth_2q": None,
                        "raw_delta_compiled_size": None,
                        "delta_compiled_size": None,
                        "delta_compiled_cx_count": None,
                        "delta_compiled_ecr_count": None,
                        "penalty_total": float("inf"),
                        "error": str(trial_row.get("error") or base_row.get("error") or "transpile_failed"),
                    }
                )
                continue
            raw_1q = int(trial_row.get("compiled_count_1q", 0)) - int(
                base_row.get("compiled_count_1q", 0)
            )
            raw_2q = int(trial_row.get("compiled_count_2q", 0)) - int(base_row.get("compiled_count_2q", 0))
            raw_depth = int(trial_row.get("compiled_depth", 0)) - int(base_row.get("compiled_depth", 0))
            base_depth_2q_raw = base_row.get("compiled_depth_2q", None)
            trial_depth_2q_raw = trial_row.get("compiled_depth_2q", None)
            raw_depth_2q = (
                int(trial_depth_2q_raw) - int(base_depth_2q_raw)
                if trial_depth_2q_raw is not None and base_depth_2q_raw is not None
                else int(raw_depth)
            )
            raw_size = int(trial_row.get("compiled_size", 0)) - int(base_row.get("compiled_size", 0))
            delta_2q = max(raw_2q, 0)
            delta_1q = max(raw_1q, 0)
            delta_depth = max(raw_depth, 0)
            delta_depth_2q = max(raw_depth_2q, 0)
            delta_size = max(raw_size, 0)
            proxy_c_hat_1q = (
                0.0
                if proxy_baseline is None
                else float(
                    proxy_baseline.c_hat_1q
                    or proxy_baseline.sq_proxy_total
                )
            )
            c_hat_1q = (
                float(delta_1q)
                if str(self.config.one_qubit_coordinate_policy)
                == ONE_QUBIT_COORDINATE_COMPILED_POSITIVE_DELTA_V1
                else float(max(0.0, proxy_c_hat_1q))
            )
            proxy_c_hat_theta = 0.0 if proxy_baseline is None else float(proxy_baseline.c_hat_theta)
            w_2q = float(self.config.weight_2q)
            w_depth = float(self.config.weight_depth)
            w_size = float(self.config.weight_size)
            clipped_penalty_total = float(w_2q * delta_2q + w_depth * delta_depth + w_size * delta_size)
            signed_penalty_total = float(w_2q * raw_2q + w_depth * raw_depth + w_size * raw_size)
            penalty_total = (
                float(signed_penalty_total)
                if bool(self.config.reward_negative_deltas)
                else float(clipped_penalty_total)
            )
            rows.append(
                {
                    **trial_row,
                    "selected_backend_name": backend_name,
                    "base_structure_key": str(
                        base_row.get("structure_key", "")
                    ),
                    "trial_structure_key": str(
                        trial_row.get("structure_key", "")
                    ),
                    "base_initial_layout": base_row.get("initial_layout"),
                    "trial_initial_layout": trial_row.get("initial_layout"),
                    "base_logical_to_physical": [
                        int(value)
                        for value in base_row.get(
                            "logical_to_physical", ()
                        )
                    ],
                    "trial_logical_to_physical": [
                        int(value)
                        for value in trial_row.get(
                            "logical_to_physical", ()
                        )
                    ],
                    "base_trial_layout_coupling_policy": (
                        "independent_unconstrained_full_transpiles_v1"
                    ),
                    "base_compiled_count_1q": int(
                        base_row.get("compiled_count_1q", 0)
                    ),
                    "base_compiled_count_2q": int(base_row.get("compiled_count_2q", 0)),
                    "base_compiled_depth": int(base_row.get("compiled_depth", 0)),
                    "base_compiled_depth_2q": (None if base_row.get("compiled_depth_2q") is None else int(base_row.get("compiled_depth_2q", 0))),
                    "base_compiled_size": int(base_row.get("compiled_size", 0)),
                    "base_compiled_cx_count": int(base_row.get("compiled_cx_count", 0)),
                    "base_compiled_ecr_count": int(base_row.get("compiled_ecr_count", 0)),
                    "raw_delta_compiled_count_1q": int(raw_1q),
                    "delta_compiled_count_1q": int(delta_1q),
                    "raw_delta_compiled_count_2q": int(raw_2q),
                    "delta_compiled_count_2q": int(delta_2q),
                    "raw_delta_compiled_depth": int(raw_depth),
                    "delta_compiled_depth": int(delta_depth),
                    "raw_delta_compiled_depth_2q": int(raw_depth_2q),
                    "delta_compiled_depth_2q": int(delta_depth_2q),
                    "raw_delta_compiled_size": int(raw_size),
                    "delta_compiled_size": int(delta_size),
                    "delta_compiled_cx_count": int(max(int(trial_row.get("compiled_cx_count", 0)) - int(base_row.get("compiled_cx_count", 0)), 0)),
                    "delta_compiled_ecr_count": int(max(int(trial_row.get("compiled_ecr_count", 0)) - int(base_row.get("compiled_ecr_count", 0)), 0)),
                    "penalty_total": float(penalty_total),
                    "clipped_penalty_total": float(clipped_penalty_total),
                    "signed_penalty_total": float(signed_penalty_total),
                    "c_hat_2q": float(delta_2q),
                    "c_hat_d": float(delta_depth_2q),
                    "c_hat_1q": float(c_hat_1q),
                    "c_hat_theta": float(max(0.0, proxy_c_hat_theta)),
                    "hardware_cost_source": str(hardware_cost_source),
                    "source_mode": str(source_mode),
                    "negative_delta_reward_enabled": bool(self.config.reward_negative_deltas),
                    "one_qubit_coordinate_policy": str(
                        self.config.one_qubit_coordinate_policy
                    ),
                    "penalty_weight_2q": float(self.config.weight_2q),
                    "penalty_weight_depth": float(self.config.weight_depth),
                    "penalty_weight_size": float(self.config.weight_size),
                }
            )

        selected = rank_compile_rows(
            rows,
            status_key="transpile_status",
            field_order=(
                "delta_compiled_count_2q",
                "delta_compiled_depth_2q",
                "delta_compiled_depth",
                "delta_compiled_size",
                "signed_penalty_total",
                "selected_backend_name",
            ),
        )
        proxy_dict = self._proxy_baseline_dict(proxy_baseline)
        if selected is None:
            return CompileCostEstimate(
                new_pauli_actions=(0.0 if proxy_baseline is None else float(proxy_baseline.new_pauli_actions)),
                new_rotation_steps=(0.0 if proxy_baseline is None else float(proxy_baseline.new_rotation_steps)),
                position_shift_span=(0.0 if proxy_baseline is None else float(proxy_baseline.position_shift_span)),
                refit_active_count=(0.0 if proxy_baseline is None else float(proxy_baseline.refit_active_count)),
                proxy_total=(0.0 if proxy_baseline is None else float(proxy_baseline.proxy_total)),
                cx_proxy_total=(0.0 if proxy_baseline is None else float(proxy_baseline.cx_proxy_total)),
                sq_proxy_total=(0.0 if proxy_baseline is None else float(proxy_baseline.sq_proxy_total)),
                gate_proxy_total=(0.0 if proxy_baseline is None else float(proxy_baseline.gate_proxy_total)),
                max_pauli_weight=(0.0 if proxy_baseline is None else float(proxy_baseline.max_pauli_weight)),
                c_hat_2q=(0.0 if proxy_baseline is None else float(proxy_baseline.c_hat_2q)),
                c_hat_d=(0.0 if proxy_baseline is None else float(proxy_baseline.c_hat_d)),
                c_hat_1q=(0.0 if proxy_baseline is None else float(proxy_baseline.c_hat_1q)),
                c_hat_theta=(0.0 if proxy_baseline is None else float(proxy_baseline.c_hat_theta)),
                hardware_cost_source=str(hardware_cost_source),
                source_mode=str(source_mode),
                penalty_total=float("inf"),
                depth_surrogate=float("inf"),
                compile_gate_open=False,
                failure_reason="all_targets_failed",
                aggregation_mode=self._aggregation_mode(),
                target_backend_names=[str(target.resolved_name) for target in self.targets],
                successful_target_count=0,
                failed_target_count=int(len(rows)),
                proxy_baseline=proxy_dict,
            )
        successful_target_count = sum(1 for row in rows if str(row.get("transpile_status", "")) == "ok")
        failed_target_count = int(len(rows) - successful_target_count)
        return CompileCostEstimate(
            new_pauli_actions=(0.0 if proxy_baseline is None else float(proxy_baseline.new_pauli_actions)),
            new_rotation_steps=(0.0 if proxy_baseline is None else float(proxy_baseline.new_rotation_steps)),
            position_shift_span=(0.0 if proxy_baseline is None else float(proxy_baseline.position_shift_span)),
            refit_active_count=(0.0 if proxy_baseline is None else float(proxy_baseline.refit_active_count)),
            proxy_total=(0.0 if proxy_baseline is None else float(proxy_baseline.proxy_total)),
            cx_proxy_total=(0.0 if proxy_baseline is None else float(proxy_baseline.cx_proxy_total)),
            sq_proxy_total=(0.0 if proxy_baseline is None else float(proxy_baseline.sq_proxy_total)),
            gate_proxy_total=(0.0 if proxy_baseline is None else float(proxy_baseline.gate_proxy_total)),
            max_pauli_weight=(0.0 if proxy_baseline is None else float(proxy_baseline.max_pauli_weight)),
            c_hat_2q=float(selected.get("c_hat_2q", 0.0)),
            c_hat_d=float(selected.get("c_hat_d", 0.0)),
            c_hat_1q=float(selected.get("c_hat_1q", 0.0)),
            c_hat_theta=float(selected.get("c_hat_theta", 0.0)),
            hardware_cost_source=str(selected.get("hardware_cost_source", str(hardware_cost_source))),
            source_mode=str(source_mode),
            penalty_total=float(selected.get("penalty_total", float("inf"))),
            depth_surrogate=float(selected.get("penalty_total", float("inf"))),
            compile_gate_open=True,
            failure_reason=None,
            selected_backend_name=str(selected.get("selected_backend_name", "")) or None,
            selected_resolution_kind=(None if selected.get("resolution_kind") is None else str(selected.get("resolution_kind"))),
            aggregation_mode=self._aggregation_mode(),
            target_backend_names=[str(target.resolved_name) for target in self.targets],
            successful_target_count=int(successful_target_count),
            failed_target_count=int(failed_target_count),
            raw_delta_compiled_count_2q=(None if selected.get("raw_delta_compiled_count_2q") is None else float(selected.get("raw_delta_compiled_count_2q", 0.0))),
            delta_compiled_count_2q=(None if selected.get("delta_compiled_count_2q") is None else float(selected.get("delta_compiled_count_2q", 0.0))),
            raw_delta_compiled_depth=(None if selected.get("raw_delta_compiled_depth") is None else float(selected.get("raw_delta_compiled_depth", 0.0))),
            delta_compiled_depth=(None if selected.get("delta_compiled_depth") is None else float(selected.get("delta_compiled_depth", 0.0))),
            raw_delta_compiled_depth_2q=(None if selected.get("raw_delta_compiled_depth_2q") is None else float(selected.get("raw_delta_compiled_depth_2q", 0.0))),
            delta_compiled_depth_2q=(None if selected.get("delta_compiled_depth_2q") is None else float(selected.get("delta_compiled_depth_2q", 0.0))),
            raw_delta_compiled_size=(None if selected.get("raw_delta_compiled_size") is None else float(selected.get("raw_delta_compiled_size", 0.0))),
            delta_compiled_size=(None if selected.get("delta_compiled_size") is None else float(selected.get("delta_compiled_size", 0.0))),
            delta_compiled_cx_count=(None if selected.get("delta_compiled_cx_count") is None else float(selected.get("delta_compiled_cx_count", 0.0))),
            delta_compiled_ecr_count=(None if selected.get("delta_compiled_ecr_count") is None else float(selected.get("delta_compiled_ecr_count", 0.0))),
            base_compiled_count_2q=(None if selected.get("base_compiled_count_2q") is None else float(selected.get("base_compiled_count_2q", 0.0))),
            base_compiled_depth=(None if selected.get("base_compiled_depth") is None else float(selected.get("base_compiled_depth", 0.0))),
            base_compiled_size=(None if selected.get("base_compiled_size") is None else float(selected.get("base_compiled_size", 0.0))),
            trial_compiled_count_2q=(None if selected.get("compiled_count_2q") is None else float(selected.get("compiled_count_2q", 0.0))),
            trial_compiled_depth=(None if selected.get("compiled_depth") is None else float(selected.get("compiled_depth", 0.0))),
            trial_compiled_size=(None if selected.get("compiled_size") is None else float(selected.get("compiled_size", 0.0))),
            proxy_baseline=proxy_dict,
            selected_backend_row={k: v for k, v in selected.items() if k not in {"compiled_op_counts"}},
        )

    def _initial_layouts_from_prefix_rows(
        self,
        prefix_rows: Sequence[Mapping[str, Any]],
        *,
        position_id: int,
    ) -> dict[str, tuple[int, ...]]:
        layouts: dict[str, tuple[int, ...]] = {}
        failures: list[str] = []
        for row in prefix_rows:
            backend_name = str(row.get("transpile_backend", ""))
            if str(row.get("transpile_status", "")) != "ok":
                failures.append(f"{backend_name}:prefix_transpile_status={row.get('transpile_status')}")
                continue
            raw_layout = row.get("logical_to_physical", ())
            layout = self._normalize_initial_layout(raw_layout if raw_layout is not None else None)
            if layout is None or len(layout) != int(self.num_qubits):
                failures.append(f"{backend_name}:missing_or_invalid_prefix_final_layout")
                continue
            layouts[backend_name] = tuple(int(q) for q in layout)
        if failures or not layouts:
            raise RuntimeError(
                "incremental_prefix_suffix_v1 requires successful prefix layouts for every backend; "
                f"position={int(position_id)}; failures={';'.join(failures) if failures else 'none'}"
            )
        return layouts

    def _estimate_incremental_prefix_suffix(
        self,
        snapshot: BackendCompileBaseSnapshot,
        *,
        candidate_term: AnsatzTerm,
        position_id: int,
        proxy_baseline: CompileCostEstimate | None = None,
    ) -> CompileCostEstimate:
        base_ops = list(snapshot.base_ops)
        pos = max(0, min(int(position_id), len(base_ops)))
        if pos == len(base_ops):
            prefix_rows = tuple(snapshot.base_backend_rows)
        else:
            prefix_rows = self._compile_structure(
                structure_key=None,
                layout=None,
                ops=base_ops[:pos],
                segment_kind=f"incremental_prefix_pos_{pos}",
            )
        initial_layouts = self._initial_layouts_from_prefix_rows(prefix_rows, position_id=pos)
        base_tail_ops = base_ops[pos:]
        trial_tail_ops = [candidate_term] + base_tail_ops
        base_tail_rows = self._compile_structure(
            structure_key=None,
            layout=None,
            ops=base_tail_ops,
            ref_state=None,
            initial_layout_by_backend=initial_layouts,
            segment_kind=f"incremental_base_tail_pos_{pos}",
        )
        trial_tail_rows = self._compile_structure(
            structure_key=None,
            layout=None,
            ops=trial_tail_ops,
            ref_state=None,
            initial_layout_by_backend=initial_layouts,
            segment_kind=f"incremental_trial_tail_pos_{pos}",
        )
        estimate = self._estimate_from_rows(
            base_rows=base_tail_rows,
            trial_rows=trial_tail_rows,
            proxy_baseline=proxy_baseline,
            source_mode=_INCREMENTAL_SOURCE,
            hardware_cost_source=_INCREMENTAL_SOURCE,
        )
        if estimate.selected_backend_row is not None:
            row = dict(estimate.selected_backend_row)
            row["incremental_prefix_suffix"] = {
                "schema": "incremental_prefix_suffix_compile_cost_v1",
                "position_id": int(pos),
                "base_depth": int(len(base_ops)),
                "prefix_depth": int(pos),
                "base_tail_depth": int(len(base_tail_ops)),
                "trial_tail_depth": int(len(trial_tail_ops)),
                "initial_layout_by_backend": {
                    str(k): [int(q) for q in v] for k, v in sorted(initial_layouts.items())
                },
                "strict_no_proxy_fallback": True,
            }
            estimate = CompileCostEstimate(
                **{
                    **estimate.__dict__,
                    "selected_backend_row": row,
                }
            )
        return estimate

    def estimate_insertion(
        self,
        snapshot: BackendCompileBaseSnapshot,
        *,
        candidate_term: AnsatzTerm,
        position_id: int,
        proxy_baseline: CompileCostEstimate | None = None,
        cache_identity: Mapping[str, Any] | None = None,
    ) -> CompileCostEstimate:
        self.estimate_count += 1
        if cache_identity is not None and str(self.config.mode) == (
            _INCREMENTAL_PREFIX_SUFFIX_MODE
        ):
            raise ValueError(
                "Phase-I--III candidate-position cache identity requires "
                "full base/trial transpilation."
            )
        if str(self.config.mode) == _INCREMENTAL_PREFIX_SUFFIX_MODE:
            return self._estimate_incremental_prefix_suffix(
                snapshot,
                candidate_term=candidate_term,
                position_id=int(position_id),
                proxy_baseline=proxy_baseline,
            )
        trial_ops = list(snapshot.base_ops)
        pos = max(0, min(int(position_id), len(trial_ops)))
        trial_ops.insert(pos, candidate_term)
        trial_layout, _qc = build_structural_ansatz_circuit(
            trial_ops,
            nq=int(self.num_qubits),
            ref_state=self.ref_state,
            structure_theta_value=float(self.config.structure_theta_value),
        )
        trial_key = self._structure_key(trial_layout)
        base_rows: Sequence[Mapping[str, Any]] = snapshot.base_backend_rows
        cache_payload: dict[str, Any] | None = None
        cache_digest: str | None = None
        if cache_identity is not None:
            raw_identity = dict(cache_identity)
            expected_keys = {
                "scope",
                "candidate_label",
                "generator_id",
                "position_id",
            }
            if (
                set(raw_identity) != expected_keys
                or raw_identity.get("scope")
                != BACKEND_COMPILE_SCOPE_PHASE123_QISKIT_V1
                or str(raw_identity.get("candidate_label", ""))
                != str(candidate_term.label)
                or not str(raw_identity.get("generator_id", ""))
                or isinstance(raw_identity.get("position_id"), bool)
                or int(raw_identity.get("position_id", -1)) != int(pos)
                or int(position_id) != int(pos)
            ):
                raise ValueError(
                    "Phase-I--III compile cache identity is incomplete or "
                    "does not match the candidate insertion."
                )
            cache_payload = {
                "schema": (
                    "phase123_qiskit_candidate_position_compile_cache_v1"
                ),
                "scope": BACKEND_COMPILE_SCOPE_PHASE123_QISKIT_V1,
                "candidate_label": str(candidate_term.label),
                "generator_id": str(raw_identity["generator_id"]),
                "position_id": int(pos),
                "base_structure_key": str(snapshot.base_structure_key),
                "trial_structure_key": str(trial_key),
            }
            cache_digest = hashlib.sha256(
                json.dumps(
                    cache_payload,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            base_rows = self._compile_structure(
                structure_key=str(snapshot.base_structure_key),
                layout=snapshot.base_layout,
                ops=snapshot.base_ops,
                cache_namespace=f"{cache_digest}:base",
            )
        trial_rows = self._compile_structure(
            structure_key=str(trial_key),
            layout=trial_layout,
            ops=trial_ops,
            cache_namespace=(
                None if cache_digest is None else f"{cache_digest}:trial"
            ),
        )
        estimate = self._estimate_from_rows(
            base_rows=base_rows,
            trial_rows=trial_rows,
            proxy_baseline=proxy_baseline,
        )
        if cache_payload is not None:
            selected = estimate.selected_backend_row
            if not isinstance(selected, Mapping) or (
                str(selected.get("base_structure_key", ""))
                != str(cache_payload["base_structure_key"])
                or str(selected.get("trial_structure_key", ""))
                != str(cache_payload["trial_structure_key"])
            ):
                raise RuntimeError(
                    "Phase-I--III compile result drifted from its exact "
                    "candidate-position cache identity."
                )
            estimate = CompileCostEstimate(
                **{
                    **estimate.__dict__,
                    "selected_backend_row": {
                        **dict(selected),
                        "compile_cache_identity": cache_payload,
                        "compile_cache_identity_sha256": str(cache_digest),
                    },
                }
            )
        return estimate

    def final_scaffold_summary(self, ops: Sequence[AnsatzTerm]) -> dict[str, Any]:
        snapshot = self.snapshot_base(ops)
        rows: list[dict[str, Any]] = []
        for row in snapshot.base_backend_rows:
            row_dict = dict(row)
            if str(row_dict.get("transpile_status", "")) == "ok":
                row_dict["absolute_burden_score_v1"] = float(
                    float(row_dict.get("compiled_count_2q", 0.0))
                    + 0.1 * float(row_dict.get("compiled_depth", 0.0))
                    + 0.01 * float(row_dict.get("compiled_size", 0.0))
                )
            else:
                row_dict["absolute_burden_score_v1"] = float("inf")
            rows.append(row_dict)
        best = rank_compile_rows(rows)
        return {
            "rows": rows,
            "selected_backend": (None if best is None else dict(best)),
        }

    def cache_summary(self) -> dict[str, Any]:
        return {
            "estimate_count": int(self.estimate_count),
            "row_hits": int(self.row_hits),
            "row_misses": int(self.row_misses),
            "compile_failures": int(self.compile_failures),
            "cache_entries": int(len(self.stats_cache)),
        }
