from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pipelines.reporting import (
    build_paper_i_hh_insertion_policy_overlays as overlays,
)
from pipelines.reporting import paper_i_hh_uniform_k50_qiskit as uniform


def _append_summary() -> dict:
    history = [
        {
            "controller_round": round_index,
            "energy_before": 102.0 - round_index,
            "energy_after": 101.0 - round_index,
        }
        for round_index in range(1, 51)
    ]
    return {
        "schema": "paper_i_append_run_summary_v1",
        "controller_rounds_completed": 50,
        "protocol_horizon": 50,
        "stop_reason": "maximum_controller_rounds",
        "final_energy": 51.0,
        "accepted_history": history,
        "estimator_accounting": {"S_alg": 500},
        "resources": {
            "terminal_observation_status": "ok",
            "terminal_compiled_resources": {
                "compiled_circuit_stats_status": "ok",
                "compiled_resource_qiskit_validated": True,
                "compile_convention": "table_i_basis_gate_transpile_v1",
                "compiled_basis_gates": [
                    "id",
                    "x",
                    "sx",
                    "rx",
                    "ry",
                    "rz",
                    "h",
                    "s",
                    "sdg",
                    "cx",
                    "cz",
                ],
                "qiskit_version": "2.3.1",
                "qiskit_transpile_optimization_level": 0,
                "qiskit_transpile_seed": 7,
                "grouped_exact_coefficient_tolerance": 1.0e-12,
                "grouped_exact_max_active_qubits": 5,
                "angle_convention": (
                    "structural_nonzero_placeholder_angles_v1"
                ),
                "compiled_circuit_scope": (
                    "ansatz_circuit_including_reference_state"
                ),
                "grouped_exact_synthesis_id": (
                    "commuting_pauli_or_active_support_unitary_exact_v1"
                ),
                "qiskit_basis_work_status": "ok",
                "qiskit_basis_work_schema": (
                    "qiskit_pretranspile_pauli_basis_work_v1"
                ),
                "qiskit_basis_work_non_attributable_operator_count": 0,
                "qiskit_pretranspile_basis_change_1q_total": 150,
                "compiled_count_2q_total": 100,
                "compiled_depth_2q_total": 80,
                "compiled_depth_total": 300,
                "qiskit_pretranspile_pauli_1q_work_total": 200,
                "num_qubits": 8,
                "logical_operator_count": 50,
                "runtime_rotation_count": 50,
                "generator_coefficients_sha256": "a" * 64,
            },
        },
    }


def _append_source() -> dict:
    return {
        "regime_id": "weak_weak",
        "candidate_representation": "macro_generator_v1",
        "route_id": "append_macro",
        "exact_same_cutoff_energy": 0.0,
        "terminal": {
            "k": 50,
            "status": "complete",
            "error": 51.0,
            "N2q": 100,
            "D2q": 80,
            "Dc": 300,
            "W1q": 200,
            "S_alg": 500,
        },
    }


def test_adoption_receipt_promotes_only_the_twelve_append_cells() -> None:
    receipt = json.loads(
        overlays.APPEND_ADOPTION_RECEIPT.read_text(encoding="utf-8")
    )
    adoption = receipt["adoption"]

    assert adoption["paper_evidence_adopted"] is True
    assert adoption["component_count"] == 12
    assert tuple(sorted(adoption["execution_ids"])) == (
        overlays._expected_append_execution_ids()
    )
    assert adoption["aggregate_partial_report_adopted"] is False
    assert adoption["ra_cells_adopted"] is False
    assert adoption["immutable_source_artifacts_modified"] is False
    assert adoption["decision_date"] == "2026-07-29"
    assert adoption["fixed_horizon_contract"]["resource_iteration"] == 50
    assert adoption["fixed_horizon_contract"]["trajectory_iterations"] == (
        list(range(51))
    )
    assert overlays._sha256(overlays.APPEND_ADOPTION_RECEIPT) == (
        "36b692255e73dd8287de5e309259e1f8703dd0ef4e2f5e889279e97c2c4d23ce"
    )


def test_adopted_production_source_closes_all_twelve_cells() -> None:
    dataset = overlays._load_adopted_append_cells()

    assert dataset["schema"] == (
        "paper_i_append_stationary_core_overlay_source_v1"
    )
    assert {
        kind: tuple(sorted(cells))
        for kind, cells in dataset["cells"].items()
    } == {
        "macro": tuple(sorted(overlays.PLOT_ORDER)),
        "singleton": tuple(sorted(overlays.PLOT_ORDER)),
    }
    expected_totals = {
        "macro": {
            "N2q": 14_932,
            "D2q": 11_882,
            "Dc": 65_049,
            "W1q": 29_480,
            "S_alg": 4_781_403,
        },
        "singleton": {
            "N2q": 1_532,
            "D2q": 1_318,
            "Dc": 6_087,
            "W1q": 2_832,
            "S_alg": 4_335_167,
        },
    }
    for kind, cells in dataset["cells"].items():
        assert all(
            [row["k"] for row in cell["trajectory"]] == list(range(51))
            for cell in cells.values()
        )
        assert {
            field: sum(
                int(cell["terminal_costs"][field])
                for cell in cells.values()
            )
            for field in overlays.RESOURCE_FIELDS
        } == expected_totals[kind]
    assert dataset["cells"]["macro"]["strong_strong_u8"][
        "trajectory"
    ][-1]["delta_E"] == pytest.approx(3.953301049830493e-5)
    assert dataset["cells"]["singleton"]["strong_strong_u8"][
        "trajectory"
    ][-1]["delta_E"] == pytest.approx(1.161507312552601e-8)


def test_append_source_preserves_k0_but_display_projects_to_k1_through_50() -> None:
    cell = overlays._append_cell_from_summary(
        _append_summary(),
        execution_id="fixture_append",
        source=_append_source(),
    )

    assert [row["k"] for row in cell["trajectory"]] == list(range(51))
    x, y = overlays._trajectory_points(
        cell["trajectory"],
        require_source_zero_to_terminal=True,
    )
    assert x == list(range(1, 51))
    assert len(y) == 50
    assert y[-1] == pytest.approx(51.0)
    assert cell["terminal_costs"] == {
        "N2q": 100,
        "D2q": 80,
        "Dc": 300,
        "W1q": 200,
        "S_alg": 500,
    }
    assert cell["compile_receipt"]["logical_operator_count"] == 50
    assert cell["compiler_fingerprint"]["qiskit_version"] == "2.3.1"


def test_append_source_rejects_report_terminal_cost_drift() -> None:
    source = copy.deepcopy(_append_source())
    source["terminal"]["S_alg"] = 499

    with pytest.raises(ValueError, match="report S_alg drifted"):
        overlays._append_cell_from_summary(
            _append_summary(),
            execution_id="fixture_append",
            source=source,
        )


def test_append_source_rejects_mixed_qiskit_fingerprint() -> None:
    summary = _append_summary()
    summary["resources"]["terminal_compiled_resources"][
        "qiskit_transpile_seed"
    ] = 8

    with pytest.raises(ValueError, match="compiler fingerprint drifted"):
        overlays._append_cell_from_summary(
            summary,
            execution_id="fixture_append",
            source=_append_source(),
        )


def test_macro_always_weak_weak_uses_runtime_term_order() -> None:
    evidence_path = overlays.OUTPUT_DIR / (
        f"{overlays.STEM}_macro_always_insertion_batch_page15_evidence.json"
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    row = next(
        item for item in evidence["rows"] if item["regime"] == "weak_weak"
    )

    endpoint = uniform.compile_compact_always_endpoint(
        row,
        repo_root=Path(overlays.REPO_ROOT),
    )

    assert endpoint["costs"] == {
        "N2q": 1_268,
        "D2q": 1_185,
        "Dc": 7_817,
        "W1q": 2_236,
        "S_alg": 2_024_560,
    }
    order = endpoint["compile_receipt"]["term_order"]
    assert order["reordered_operator_count"] == 46
    assert order["substantive_term_changes"] is False


def test_generated_uniform_dataset_closes_all_48_fingerprints() -> None:
    dataset = json.loads(
        overlays.UNIFORM_K50_QISKIT_COSTS.read_text(encoding="utf-8")
    )
    index = uniform.validate_uniform_k50_dataset(
        dataset,
        plot_order=overlays.PLOT_ORDER,
    )

    assert len(index) == 48
    assert {
        row["compiler_fingerprint_sha256"]
        for row in dataset["rows"]
    } == {dataset["compiler_fingerprint_sha256"]}
    assert dataset["fingerprint_guard"] == {
        "status": "all_endpoints_identical",
        "validated_endpoint_count": 48,
    }
    macro_always = [
        row
        for row in dataset["rows"]
        if row["representation_kind"] == "macro"
        and row["policy"] == "always"
    ]
    assert {
        field: sum(row["costs"][field] for row in macro_always)
        for field in overlays.RESOURCE_FIELDS
    } == {
        "N2q": 10_612,
        "D2q": 8_967,
        "Dc": 54_831,
        "W1q": 18_826,
        "S_alg": 12_676_972,
    }
    changed = [
        row
        for row in dataset["rows"]
        if any(
            row["uniform_minus_legacy"][field]
            for field in ("N2q", "D2q", "Dc", "W1q")
        )
    ]
    assert len(changed) == 18
    assert {
        (row["representation_kind"], row["policy"]) for row in changed
    } == {
        ("macro", "always"),
        ("macro", "plateau"),
        ("macro", "no_insertion"),
    }


def test_fixed_horizon_summary_uses_signed_append_minus_ra_differences() -> None:
    append = {
        regime: {
            "N2q": 100,
            "D2q": 100,
            "Dc": 100,
            "W1q": 100,
            "S_alg": 100,
        }
        for regime in overlays.PLOT_ORDER
    }
    comparator = {
        regime: {
            "N2q": 80,
            "D2q": 80,
            "Dc": 80,
            "W1q": 80,
            "S_alg": 120,
        }
        for regime in overlays.PLOT_ORDER
    }

    summary = overlays._fixed_horizon_resource_summary(
        representation="intact_macro",
        comparator_key="always",
        append_costs_by_regime=append,
        comparator_costs_by_regime=comparator,
    )

    first = summary["rows"][0]
    assert first["append_minus_ra"] == {
        "N2q": 20,
        "D2q": 20,
        "Dc": 20,
        "W1q": 20,
        "S_alg": -20,
    }
    assert first["percentage_reduction_relative_to_append"] == {
        "N2q": 20.0,
        "D2q": 20.0,
        "Dc": 20.0,
        "W1q": 20.0,
        "S_alg": -20.0,
    }
    assert first["dominance"]["all_fields"] == "mixed"
    assert summary["totals"]["append_minus_ra"]["N2q"] == 120
    assert summary["totals"]["append_minus_ra"]["S_alg"] == -120
