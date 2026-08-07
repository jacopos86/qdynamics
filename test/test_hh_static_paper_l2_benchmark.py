#!/usr/bin/env python3
"""Tests for the paper-facing L=2 static benchmark wrapper."""

from __future__ import annotations

import json
from pathlib import Path

from pipelines.exact_bench import hh_static_paper_l2_benchmark as paper


class _FakeResolvedProblem:
    pass


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_classify_static_row_keeps_only_clean_paper_candidates() -> None:
    candidate = paper.classify_static_row(
        {
            "method_id": "hh_adapt_full_meta_legacy",
            "method_kind": "adapt_vqe",
            "quality_status": "ok_paper_candidate",
            "delta_E_abs": 7.0e-5,
            "benchmark_audit_flags": [],
        }
    )
    assert candidate["paper_role"] == "candidate_ours"
    assert candidate["paper_include"] is True
    assert candidate["qpu_compatibility"] == "qpu_compatible_state_preparation_benchmark"

    suspect = paper.classify_static_row(
        {
            "method_id": "hh_hva_termwise_vqe",
            "method_kind": "conventional_vqe",
            "quality_status": "ok_optimizer_suspect",
            "delta_E_abs": 0.003,
            "benchmark_audit_flags": ["optimizer_suspect"],
        }
    )
    assert suspect["paper_role"] == "diagnostic_optimizer_suspect"
    assert suspect["paper_include"] is False

    qsci = paper.classify_static_row(
        {
            "method_id": "hh_qsci_sq_lf_std",
            "method_kind": "compiled_operator_qsci",
            "quality_status": "ok_large_error",
            "delta_E_abs": 4.0,
            "benchmark_audit_flags": [],
        }
    )
    assert qsci["paper_role"] == "diagnostic_large_error"
    assert qsci["qpu_compatibility"] == "qpu_compatible_sampling_subspace_diagnostic"


def test_build_paper_l2_benchmark_enriches_rows_and_writes_artifacts(tmp_path, monkeypatch) -> None:
    payload_path = tmp_path / "runs" / "hh_L2_strong_canonical" / "hh_adapt_full_meta_legacy.json"
    _write_json(payload_path, {"benchmark_status": "ok", "parameterization": {"blocks": []}})
    rows_path = tmp_path / "hh_static_benchmark_rows.json"
    _write_json(
        rows_path,
        [
            {
                "case_id": "hh_L2_strong_canonical",
                "method_id": "hh_adapt_full_meta_legacy",
                "method_kind": "adapt_vqe",
                "quality_status": "ok_paper_candidate",
                "delta_E_abs": 7.0e-5,
                "energy": 0.1587,
                "artifact_json": str(payload_path),
                "benchmark_audit_flags": [],
            },
            {
                "case_id": "hh_L3_weak_current_success",
                "method_id": "hh_adapt_full_meta_legacy",
                "method_kind": "adapt_vqe",
                "quality_status": "ok_paper_candidate",
                "delta_E_abs": 1.0e-3,
                "artifact_json": str(payload_path),
                "benchmark_audit_flags": [],
            },
        ],
    )

    monkeypatch.setattr(paper, "resolve_problem_context", lambda request: _FakeResolvedProblem())
    monkeypatch.setattr(
        paper,
        "build_static_circuit_for_row",
        lambda **kwargs: (object(), "static_test_scope", "test note", 3),
    )
    monkeypatch.setattr(
        paper,
        "_compile_cost_payload",
        lambda **kwargs: {
            "transpile_status": "ok",
            "backend_name": "FakeMarrakesh",
            "seed_transpiler": 7,
            "optimization_level": 2,
            "abstract_size": 11,
            "abstract_depth": 9,
            "compiled_count_2q": 81,
            "compiled_depth": 120,
            "compiled_size": 300,
            "compiled_num_qubits": 156,
            "compiled_op_counts": {"cx": 81},
            "logical_to_physical": [0, 1, 2, 3, 4, 5],
            "error": None,
            "compile_audit_rows": [],
        },
    )

    result = paper.build_paper_l2_benchmark(
        output_dir=tmp_path / "paper",
        input_rows=rows_path,
        case_ids=("hh_L2_strong_canonical",),
        compile_policy="all",
    )

    assert Path(result["manifest_path"]).exists()
    rows = json.loads(Path(result["rows_path"]).read_text(encoding="utf-8"))
    assert len(rows) == 1
    row = rows[0]
    assert row["paper_role"] == "candidate_ours"
    assert row["paper_include"] is True
    assert row["static_compile_status"] == "ok"
    assert row["static_compile_scope"] == "static_test_scope"
    assert row["static_compiled_2q"] == 81
    assert row["static_compiled_operator_count"] == 3
    assert result["summary"]["paper_include_count"] == 1
    assert result["summary"]["static_pareto_by_case"]["hh_L2_strong_canonical"][0]["method_id"] == "hh_adapt_full_meta_legacy"


def test_pareto_front_uses_delta_e_and_static_two_qubit_cost() -> None:
    rows = [
        {"method_id": "best_quality", "delta_E_abs": 0.01, "static_compiled_2q": 100},
        {"method_id": "cheap", "delta_E_abs": 0.03, "static_compiled_2q": 30},
        {"method_id": "dominated", "delta_E_abs": 0.04, "static_compiled_2q": 120},
    ]

    front = paper._pareto_front(rows)

    assert [row["method_id"] for row in front] == ["best_quality", "cheap"]
