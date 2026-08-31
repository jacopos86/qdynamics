#!/usr/bin/env python3
"""Backend-conditioned transpilation oracle for HH Phase 3 candidate scoring."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.hardcoded.adapt_circuit_execution import build_structural_ansatz_circuit
from pipelines.hardcoded.hh_continuation_types import CompileCostEstimate
from pipelines.qiskit_backend_tools import (
    compile_circuit_for_backend,
    compiled_gate_stats,
    rank_compile_rows,
    resolve_backend_targets,
    safe_circuit_depth,
)
from src.quantum.ansatz_parameterization import AnsatzParameterLayout
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


_DEFAULT_PREFERRED_FAKES = ("FakeNighthawk", "FakeFez", "FakeMarrakesh")


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
    penalty_version: str = "transpile_burden_scalar_v1"


@dataclass(frozen=True)
class BackendCompileBaseSnapshot:
    base_ops: tuple[AnsatzTerm, ...]
    base_structure_key: str
    base_layout: AnsatzParameterLayout
    base_backend_rows: tuple[dict[str, Any], ...]
    logical_depth: int


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
        requested_names = [str(config.requested_backend_name)] if str(config.mode) == "transpile_single_v1" else list(config.requested_backend_shortlist)
        fallback_mode = "single" if str(config.mode) == "transpile_single_v1" else "shortlist"
        self.targets, self.resolution_audit = resolve_backend_targets(
            requested_names=requested_names,
            preferred_fake_backends=tuple(str(x) for x in config.preferred_fake_backends),
            allow_preferred_fallback=True,
            fallback_mode=str(fallback_mode),
        )
        self.stats_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self.row_hits = 0
        self.row_misses = 0
        self.compile_failures = 0

    def _ref_state_hash(self) -> str:
        if self.ref_state is None:
            return "none"
        arr = np.asarray(self.ref_state, dtype=np.complex128).reshape(-1)
        return hashlib.sha256(arr.tobytes()).hexdigest()

    def _structure_key(self, layout: AnsatzParameterLayout) -> str:
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
            "ref_state_hash": self._ref_state_hash(),
            "structure_theta_value": float(self.config.structure_theta_value),
            "layout": structural_layout,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _compile_structure(self, *, structure_key: str, layout: AnsatzParameterLayout, ops: Sequence[AnsatzTerm]) -> tuple[dict[str, Any], ...]:
        _layout_unused, qc = build_structural_ansatz_circuit(
            ops,
            nq=int(self.num_qubits),
            ref_state=self.ref_state,
            structure_theta_value=float(self.config.structure_theta_value),
        )
        rows: list[dict[str, Any]] = []
        for target in self.targets:
            cache_key = (str(structure_key), str(target.resolved_name))
            cached = self.stats_cache.get(cache_key, None)
            if cached is not None:
                self.row_hits += 1
                rows.append(dict(cached))
                continue
            self.row_misses += 1
            row: dict[str, Any] = {
                "structure_key": str(structure_key),
                "transpile_backend": str(target.resolved_name),
                "requested_backend": str(target.requested_name),
                "resolution_kind": str(target.resolution_kind),
                "using_fake_backend": bool(target.using_fake_backend),
                "target_snapshot": dict(getattr(target, "target_snapshot", {}) or {}),
            }
            try:
                compiled_info = compile_circuit_for_backend(
                    qc,
                    target.backend_obj,
                    seed_transpiler=int(self.config.seed_transpiler),
                    optimization_level=int(self.config.optimization_level),
                )
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
        }

    def _estimate_from_rows(
        self,
        *,
        base_rows: Sequence[Mapping[str, Any]],
        trial_rows: Sequence[Mapping[str, Any]],
        proxy_baseline: CompileCostEstimate | None,
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
                        "raw_delta_compiled_depth": None,
                        "delta_compiled_depth": None,
                        "raw_delta_compiled_size": None,
                        "delta_compiled_size": None,
                        "delta_compiled_cx_count": None,
                        "delta_compiled_ecr_count": None,
                        "penalty_total": float("inf"),
                        "error": str(trial_row.get("error") or base_row.get("error") or "transpile_failed"),
                    }
                )
                continue
            raw_2q = int(trial_row.get("compiled_count_2q", 0)) - int(base_row.get("compiled_count_2q", 0))
            raw_depth = int(trial_row.get("compiled_depth", 0)) - int(base_row.get("compiled_depth", 0))
            raw_size = int(trial_row.get("compiled_size", 0)) - int(base_row.get("compiled_size", 0))
            delta_2q = max(raw_2q, 0)
            delta_depth = max(raw_depth, 0)
            delta_size = max(raw_size, 0)
            penalty_total = float(delta_2q + 0.1 * delta_depth + 0.01 * delta_size)
            rows.append(
                {
                    **trial_row,
                    "selected_backend_name": backend_name,
                    "base_compiled_count_2q": int(base_row.get("compiled_count_2q", 0)),
                    "base_compiled_depth": int(base_row.get("compiled_depth", 0)),
                    "base_compiled_size": int(base_row.get("compiled_size", 0)),
                    "base_compiled_cx_count": int(base_row.get("compiled_cx_count", 0)),
                    "base_compiled_ecr_count": int(base_row.get("compiled_ecr_count", 0)),
                    "raw_delta_compiled_count_2q": int(raw_2q),
                    "delta_compiled_count_2q": int(delta_2q),
                    "raw_delta_compiled_depth": int(raw_depth),
                    "delta_compiled_depth": int(delta_depth),
                    "raw_delta_compiled_size": int(raw_size),
                    "delta_compiled_size": int(delta_size),
                    "delta_compiled_cx_count": int(max(int(trial_row.get("compiled_cx_count", 0)) - int(base_row.get("compiled_cx_count", 0)), 0)),
                    "delta_compiled_ecr_count": int(max(int(trial_row.get("compiled_ecr_count", 0)) - int(base_row.get("compiled_ecr_count", 0)), 0)),
                    "penalty_total": float(penalty_total),
                }
            )

        selected = rank_compile_rows(
            rows,
            status_key="transpile_status",
            field_order=("delta_compiled_count_2q", "delta_compiled_depth", "delta_compiled_size", "selected_backend_name"),
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
                source_mode="backend_transpile_v1",
                penalty_total=float("inf"),
                depth_surrogate=float("inf"),
                compile_gate_open=False,
                failure_reason="all_targets_failed",
                aggregation_mode=("single_backend" if str(self.config.mode) == "transpile_single_v1" else str(self.config.shortlist_reduction_mode)),
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
            source_mode="backend_transpile_v1",
            penalty_total=float(selected.get("penalty_total", float("inf"))),
            depth_surrogate=float(selected.get("penalty_total", float("inf"))),
            compile_gate_open=True,
            failure_reason=None,
            selected_backend_name=str(selected.get("selected_backend_name", "")) or None,
            selected_resolution_kind=(None if selected.get("resolution_kind") is None else str(selected.get("resolution_kind"))),
            aggregation_mode=("single_backend" if str(self.config.mode) == "transpile_single_v1" else str(self.config.shortlist_reduction_mode)),
            target_backend_names=[str(target.resolved_name) for target in self.targets],
            successful_target_count=int(successful_target_count),
            failed_target_count=int(failed_target_count),
            raw_delta_compiled_count_2q=(None if selected.get("raw_delta_compiled_count_2q") is None else float(selected.get("raw_delta_compiled_count_2q", 0.0))),
            delta_compiled_count_2q=(None if selected.get("delta_compiled_count_2q") is None else float(selected.get("delta_compiled_count_2q", 0.0))),
            raw_delta_compiled_depth=(None if selected.get("raw_delta_compiled_depth") is None else float(selected.get("raw_delta_compiled_depth", 0.0))),
            delta_compiled_depth=(None if selected.get("delta_compiled_depth") is None else float(selected.get("delta_compiled_depth", 0.0))),
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

    def estimate_insertion(
        self,
        snapshot: BackendCompileBaseSnapshot,
        *,
        candidate_term: AnsatzTerm,
        position_id: int,
        proxy_baseline: CompileCostEstimate | None = None,
    ) -> CompileCostEstimate:
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
        trial_rows = self._compile_structure(structure_key=str(trial_key), layout=trial_layout, ops=trial_ops)
        return self._estimate_from_rows(base_rows=snapshot.base_backend_rows, trial_rows=trial_rows, proxy_baseline=proxy_baseline)

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
            "row_hits": int(self.row_hits),
            "row_misses": int(self.row_misses),
            "compile_failures": int(self.compile_failures),
            "cache_entries": int(len(self.stats_cache)),
        }
