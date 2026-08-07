from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench import paper_i_hh_backend_compile_latency as diag
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, PauliPolynomial, PauliTerm


def _term(label: str, pauli: str) -> AnsatzTerm:
    return AnsatzTerm(label=str(label), polynomial=PauliPolynomial("JW", [PauliTerm(len(pauli), ps=pauli, pc=1.0)]))


def test_compile_latency_harness_uses_same_candidate_records_for_both_modes(monkeypatch) -> None:
    calls_by_mode: dict[str, list[tuple[str, int, int]]] = {}

    class _FakeOracle:
        def __init__(self, *, config, num_qubits: int, ref_state):
            self.config = config
            self.num_qubits = int(num_qubits)
            self.ref_state = ref_state
            self.resolution_audit = [{"resolved_name": "FakeMarrakesh", "success": True}]
            calls_by_mode.setdefault(str(config.mode), [])

        def snapshot_base(self, ops):
            return tuple(ops)

        def estimate_insertion(self, snapshot, *, candidate_term, position_id: int, proxy_baseline=None):
            mode = str(self.config.mode)
            calls_by_mode[mode].append((str(candidate_term.label), int(position_id), len(snapshot)))
            source_mode = (
                "backend_incremental_prefix_suffix_v1"
                if mode == "incremental_prefix_suffix_v1"
                else "backend_transpile_v1"
            )
            row = {
                "source_mode": source_mode,
                "hardware_cost_source": source_mode,
            }
            if mode == "incremental_prefix_suffix_v1":
                row["incremental_prefix_suffix"] = {"strict_no_proxy_fallback": True}
            return SimpleNamespace(
                compile_gate_open=True,
                source_mode=source_mode,
                hardware_cost_source=source_mode,
                selected_backend_name="FakeMarrakesh",
                penalty_total=1.0,
                delta_compiled_count_2q=2.0,
                delta_compiled_depth_2q=3.0,
                delta_compiled_depth=4.0,
                delta_compiled_size=5.0,
                selected_backend_row=row,
            )

        def cache_summary(self):
            return {
                "row_hits": 0,
                "row_misses": len(calls_by_mode[str(self.config.mode)]),
                "compile_failures": 0,
                "cache_entries": len(calls_by_mode[str(self.config.mode)]),
            }

    monkeypatch.setattr(diag, "BackendCompileOracle", _FakeOracle)
    pool_by_label = {"a": _term("a", "xeeeee"), "b": _term("b", "zxeeeee")}
    admissions = (
        diag.AdmissionRecord(step_index=1, label="a", position_id=0, source_path="unit"),
        diag.AdmissionRecord(step_index=2, label="b", position_id=0, source_path="unit"),
    )
    common = dict(
        admissions=admissions,
        pool_by_label=pool_by_label,
        num_qubits=6,
        ref_state=np.array([1.0] + [0.0] * 63, dtype=complex),
        backend_name="FakeMarrakesh",
        seed_transpiler=7,
        optimization_level=1,
        structure_theta_value=1.0,
        weight_2q=1.0,
        weight_depth=0.1,
        weight_size=0.01,
        record_limit=None,
    )

    full = diag.time_compile_mode(mode="transpile_single_v1", **common)
    incremental = diag.time_compile_mode(mode="incremental_prefix_suffix_v1", **common)

    assert calls_by_mode["transpile_single_v1"] == calls_by_mode["incremental_prefix_suffix_v1"]
    assert full["source_modes"] == ["backend_transpile_v1"]
    assert incremental["source_modes"] == ["backend_incremental_prefix_suffix_v1"]
    assert incremental["rows"][0]["incremental_prefix_suffix"]["strict_no_proxy_fallback"] is True
