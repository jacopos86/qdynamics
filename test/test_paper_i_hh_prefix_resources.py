#!/usr/bin/env python3
"""Regression checks for Paper-I HH exact prefix-resource exporter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("qiskit")

from pipelines.exact_bench.hh_tableiii_prefix_resources import export_prefix_resources


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_prefix_resource_exporter_compiles_ready_rows_and_carries_blockers(tmp_path: Path) -> None:
    source_json = tmp_path / "generic_static_single.json"
    runtime_seed = tmp_path / "runtime_seed.json"
    audit_json = tmp_path / "audit.json"
    output_json = tmp_path / "prefix_resources.json"
    source_map = tmp_path / "source_map.json"

    _write_json(
        runtime_seed,
        {
            "schema": "runtime_seed_fixture_v1",
            "ansatz_input_state": {
                "nq_total": 2,
                "amplitudes_qn_to_q0": {"00": {"re": 1.0, "im": 0.0}},
            },
        },
    )
    _write_json(
        source_json,
        {
            "schema": "generic_static_single_fixture_v1",
            "runtime_seed_json": "runtime_seed.json",
            "result": {
                "adapt_history": [
                    {
                        "iteration": 1,
                        "selected_batch_labels": ["g0"],
                        "energy_after": -1.0,
                        "abs_delta_e_same_cutoff_after": 0.1,
                        "top_candidates": [{"label": "g0", "pauli_labels_exyz": ["xx"]}],
                    },
                    {
                        "iteration": 2,
                        "selected_batch_labels": ["g1"],
                        "energy_after": -1.1,
                        "abs_delta_e_same_cutoff_after": 0.01,
                        "top_candidates": [{"label": "g1", "pauli_labels_exyz": ["ez"]}],
                    },
                ]
            },
        },
    )
    _write_json(source_map, {"schema": "source_map_fixture_v1"})
    _write_json(
        audit_json,
        {
            "schema": "paper_i_hh_tableiii_prefix_replayability_audit_v1",
            "rows": [
                {
                    "regime": "weak_weak",
                    "method": "Append-ADAPT",
                    "classification": "exact-prefix-replay-ready",
                    "blockers": [],
                    "primary_source": {"path": str(source_json), "history_len": 2},
                    "visible_cells": {"DeltaE": "fixture"},
                },
                {
                    "regime": "strong_strong",
                    "method": "SNAKE",
                    "classification": "stdout-only-blocked",
                    "blockers": ["visible_source_is_stdout_or_ai_log_derived"],
                    "primary_source": {"path": "stdout.json", "history_len": 47},
                },
            ],
        },
    )

    manifest = export_prefix_resources(
        source_map_path=source_map,
        audit_json_path=audit_json,
        output_json_path=output_json,
        rebuild_audit_if_missing=False,
    )

    assert output_json.exists()
    assert manifest["schema"] == "paper_i_hh_tableiii_exact_prefix_resources_v1"
    assert manifest["row_count"] == 2
    assert manifest["compiled_ok_count"] == 2
    assert manifest["blocked_row_count"] == 1
    assert manifest["blocked_rows"][0]["classification"] == "stdout-only-blocked"
    first, second = manifest["rows"]
    assert first["prefix_k"] == 1
    assert second["prefix_k"] == 2
    assert first["logical_operator_prefix_len"] == 1
    assert second["logical_operator_prefix_len"] == 2
    assert first["N1q"] is not None
    assert first["N2q"] is not None
    assert first["D_circ"] is not None
    assert first["D2q"] is not None
    assert first["compiled_count_1q_semantics"] == "post_transpile_one_qubit_quantum_ops_excluding_barrier_delay_id_measure_reset"
