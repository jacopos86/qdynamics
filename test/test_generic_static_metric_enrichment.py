#!/usr/bin/env python3
"""Tests for post-hoc generic static metric enrichment."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.exact_bench import generic_static_metric_enrichment as enrich
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm


def _write_records(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _toy_hamiltonian() -> PauliPolynomial:
    return PauliPolynomial(
        "JW",
        [
            PauliTerm(2, ps="xe", pc=3.0),
            PauliTerm(2, ps="ez", pc=4.0),
            PauliTerm(2, ps="ee", pc=100.0),
        ],
    )


def test_metric_enrichment_writes_payload_missing_sidecar(tmp_path: Path) -> None:
    record = {
        "record_id": "static_table__hubbard__hubbard_L2__static_pos_geo_adapt_vqe",
        "family": "hubbard",
        "case_id": "hubbard_L2",
        "algorithm_id": "static_pos_geo_adapt_vqe",
    }

    payload = enrich.enrich_record(record=record, input_root=tmp_path / "missing", output_dir=tmp_path / "out" / record["record_id"] / "result")

    assert payload["status"] == "payload_missing"
    sidecar = tmp_path / "out" / record["record_id"] / "result" / "generic_static_metric_enrichment.json"
    assert sidecar.exists()
    loaded = json.loads(sidecar.read_text(encoding="utf-8"))
    assert loaded["guardrails"]["uses_exact_for_decision"] is False
    assert loaded["guardrails"]["raw_payload_mutated"] is False


def test_metric_enrichment_batch_summary_counts_records(tmp_path: Path) -> None:
    record = {
        "record_id": "static_table__bose_hubbard__bose_hubbard_L2__static_qiskit_adapt_vqe",
        "family": "bose_hubbard",
        "case_id": "bose_hubbard_L2",
        "algorithm_id": "static_qiskit_adapt_vqe",
    }
    records_path = tmp_path / "records.tsv"
    _write_records(records_path, [record])

    summary = enrich.run_batch(records_path=records_path, input_root=tmp_path / "missing", output_root=tmp_path / "enrichment")

    assert summary["record_count"] == 1
    assert summary["status_counts"] == {"payload_missing": 1}
    assert (tmp_path / "enrichment" / "metric_enrichment_summary.json").exists()


def test_metric_enrichment_uses_record_suite_profile_for_case_resolution(tmp_path: Path, monkeypatch) -> None:
    record = {
        "record_id": "static_table__hubbard__hubbard_L2_three_model_weak__static_full_meta_append_adapt_vqe",
        "family": "hubbard",
        "case_id": "hubbard_L2_three_model_weak",
        "algorithm_id": "static_full_meta_append_adapt_vqe",
        "suite_profile": "paper_i_three_model_main_20260525_v1",
    }
    result_dir = tmp_path / "input" / record["record_id"] / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "generic_static_single.json").write_text(
        json.dumps({"status": "ok", "result": {"status": "ok", "energy": 0.0}}) + "\n",
        encoding="utf-8",
    )

    calls: list[tuple[str, str, str | None]] = []

    def fake_spec_by_case_id(family: str, case_id: str, profile: str | None = None):
        calls.append((family, case_id, profile))
        raise RuntimeError("stop after profile check")

    monkeypatch.setattr(enrich, "_spec_by_case_id", fake_spec_by_case_id)

    payload = enrich.enrich_record(record=record, input_root=tmp_path / "input", output_dir=tmp_path / "out" / record["record_id"] / "result")

    assert payload["status"] == "failed"
    assert calls == [("hubbard", "hubbard_L2_three_model_weak", "paper_i_three_model_main_20260525_v1")]
    assert payload["suite_profile"] == "paper_i_three_model_main_20260525_v1"


def test_enrich_record_grouped_replay_preserves_explicit_s_var_components(tmp_path: Path) -> None:
    record = {
        "record_id": "static_table__hubbard__hubbard_L2__static_family_informed_vqe",
        "family": "hubbard",
        "case_id": "hubbard_L2",
        "algorithm_id": "static_family_informed_vqe",
    }
    result_dir = tmp_path / "input" / record["record_id"] / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "generic_static_single.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "result": {
                    "status": "ok",
                    "energy": 0.0,
                    "energy_eval_count_proxy": 1,
                    "S_var_H_outer": 2,
                    "S_var_grad": 3,
                    "S_var_metric": 5,
                    "S_var_H_refit": 7,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = enrich.enrich_record(record=record, input_root=tmp_path / "input", output_dir=tmp_path / "out" / record["record_id"] / "result")

    assert payload["metric_statuses"]["S_l2"] == "ok"
    assert payload["metric_statuses"]["S_var"] == "ok"
    assert payload["row_updates"]["S_var"] == 17.0
    assert payload["metrics"]["physical_measurement_work"]["S_var"]["status"] == "ok"


def test_metric_enrichment_does_not_fake_fidelity_without_ansatz_artifacts(tmp_path: Path) -> None:
    record = {
        "record_id": "static_table__bose_hubbard__bose_hubbard_L2__static_pos_geo_adapt_vqe",
        "family": "bose_hubbard",
        "case_id": "bose_hubbard_L2",
        "algorithm_id": "static_pos_geo_adapt_vqe",
    }
    result_dir = tmp_path / "input" / record["record_id"] / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "generic_static_single.json").write_text(
        json.dumps({"status": "ok", "result": {"status": "ok", "energy": 0.0}}) + "\n",
        encoding="utf-8",
    )

    payload = enrich.enrich_record(record=record, input_root=tmp_path / "input", output_dir=tmp_path / "out" / record["record_id"] / "result")

    assert payload["status"] == "completed"
    assert payload["metric_statuses"]["infidelity_same"] == "not_reconstructable_missing_ansatz_artifacts"
    assert "infidelity_same" not in payload["row_updates"]


def test_normalized_measurement_work_fixed_vqe_energy_only() -> None:
    metric, updates, statuses = enrich._normalized_measurement_work(
        algorithm_id="static_family_informed_vqe",
        row={"energy_eval_count_proxy": 17, "shots_total": 999},
    )

    assert statuses["S_norm"] == "ok"
    assert updates["S_norm"] == 17.0
    assert updates["S_norm_N_H_outer_eval"] == 17.0
    assert updates["S_norm_N_H_eval"] == 17.0
    assert updates["S_norm_N_grad"] == 0.0
    assert updates["S_norm_N_metric"] == 0.0
    assert updates["S_norm_N_H_refit_eval"] == 0.0
    assert updates["S_norm_N_refit_eval"] == 0.0
    assert metric["components"]["N_H_outer_eval"] == 17.0
    assert metric["legacy_raw_proxy"]["shots_total"] == 999.0


def test_normalized_measurement_work_qiskit_adapt_partitions_refit() -> None:
    metric, updates, statuses = enrich._normalized_measurement_work(
        algorithm_id="static_qiskit_adapt_vqe",
        row={
            "selected_operator_count": 2,
            "energy_eval_count_proxy": 11,
            "gradient_operator_probe_count_proxy": 23,
            "shots_total": 123456,
        },
    )

    assert statuses["S_norm"] == "ok"
    assert updates["S_norm"] == 34.0
    assert updates["S_norm_N_H_outer_eval"] == 0.0
    assert updates["S_norm_N_H_eval"] == 0.0
    assert updates["S_norm_N_grad"] == 23.0
    assert updates["S_norm_N_metric"] == 0.0
    assert updates["S_norm_N_H_refit_eval"] == 11.0
    assert updates["S_norm_N_refit_eval"] == 11.0
    assert metric["components"]["N_H_refit_eval"] == 11.0


def test_normalized_measurement_work_local_full_meta_append_partitions_refit_without_metric() -> None:
    metric, updates, statuses = enrich._normalized_measurement_work(
        algorithm_id="static_full_meta_append_adapt_vqe",
        row={
            "selected_operator_count": 2,
            "energy_eval_count_proxy": 11,
            "gradient_operator_probe_count_proxy": 23,
            "metric_operator_probe_count_proxy": 0,
        },
    )

    assert statuses["S_norm"] == "ok"
    assert updates["S_norm"] == 34.0
    assert updates["S_norm_N_H_outer_eval"] == 0.0
    assert updates["S_norm_N_grad"] == 23.0
    assert updates["S_norm_N_metric"] == 0.0
    assert updates["S_norm_N_H_refit_eval"] == 11.0
    assert metric["component_sources"]["N_metric"] == "metric_operator_probe_count_proxy"


def test_normalized_measurement_work_qiskit_adapt_counts_unexpected_positive_metric_telemetry() -> None:
    metric, updates, statuses = enrich._normalized_measurement_work(
        algorithm_id="static_qiskit_adapt_vqe",
        row={
            "selected_operator_count": 1,
            "energy_eval_count_proxy": 11,
            "gradient_operator_probe_count_proxy": 23,
            "metric_operator_probe_count_proxy": 5,
        },
    )

    assert statuses["S_norm"] == "ok"
    assert updates["S_norm"] == 39.0
    assert updates["S_norm_N_metric"] == 5.0
    assert metric["component_sources"]["N_metric"] == "metric_operator_probe_count_proxy"


def test_normalized_measurement_work_adaptive_no_selected_uses_h_eval() -> None:
    _, updates, statuses = enrich._normalized_measurement_work(
        algorithm_id="static_qubit_qeb_adapt_vqe",
        row={
            "selected_operator_count": 0,
            "energy_eval_count_proxy": 1,
            "gradient_operator_probe_count_proxy": 48,
            "metric_operator_probe_count_proxy": 0,
        },
    )

    assert statuses["S_norm"] == "ok"
    assert updates["S_norm"] == 49.0
    assert updates["S_norm_N_H_outer_eval"] == 1.0
    assert updates["S_norm_N_H_eval"] == 1.0
    assert updates["S_norm_N_H_refit_eval"] == 0.0
    assert updates["S_norm_N_refit_eval"] == 0.0


def test_normalized_measurement_work_pos_geo_includes_metric() -> None:
    _, updates, statuses = enrich._normalized_measurement_work(
        algorithm_id="static_pos_geo_adapt_vqe",
        row={
            "selected_operator_count": 3,
            "energy_eval_count_proxy": 100,
            "gradient_operator_probe_count_proxy": 20,
            "metric_operator_probe_count_proxy": 400,
        },
    )

    assert statuses["S_norm"] == "ok"
    assert updates["S_norm"] == 520.0
    assert updates["S_norm_N_metric"] == 400.0


def test_normalized_measurement_work_does_not_reconstruct_from_raw_scalar() -> None:
    metric, updates, statuses = enrich._normalized_measurement_work(
        algorithm_id="static_family_native_adapt_phase3",
        row={"shot_cost_proxy": 123.0},
    )

    assert statuses["S_norm"] == "missing_component_breakdown"
    assert updates == {}
    assert metric["S_norm"] is None
    assert metric["legacy_raw_proxy"]["shot_cost_proxy"] == 123.0


def test_normalized_measurement_work_rejects_negative_component() -> None:
    metric, updates, statuses = enrich._normalized_measurement_work(
        algorithm_id="static_qiskit_adapt_vqe",
        row={
            "selected_operator_count": 1,
            "energy_eval_count_proxy": 2,
            "gradient_operator_probe_count_proxy": -1,
        },
    )

    assert statuses["S_norm"] == "invalid_component_value"
    assert updates == {}
    assert metric["S_norm"] is None


def test_normalized_measurement_work_rejects_unassigned_other_quantum_work() -> None:
    metric, updates, statuses = enrich._normalized_measurement_work(
        algorithm_id="static_family_informed_vqe",
        row={"energy_eval_count_proxy": 17, "N_other_quantum": 1},
    )

    assert statuses["S_norm"] == "unassigned_other_quantum_work"
    assert updates == {}
    assert metric["S_norm"] is None
    assert metric["N_other_quantum"] == 1.0


def test_normalized_measurement_work_canonical_fields_win_over_legacy_aliases() -> None:
    metric, updates, statuses = enrich.normalized_measurement_work_from_explicit_row(
        row={
            "S_norm_N_H_outer_eval": 2,
            "S_norm_N_H_eval": 200,
            "S_norm_N_grad": 3,
            "S_norm_N_metric": 5,
            "S_norm_N_H_refit_eval": 7,
            "S_norm_N_refit_eval": 700,
        },
        raw_proxy={"shot_cost_proxy": 999},
    )

    assert statuses["S_norm"] == "ok"
    assert updates["S_norm"] == 17.0
    assert updates["S_norm_N_H_outer_eval"] == 2.0
    assert updates["S_norm_N_H_eval"] == 2.0
    assert updates["S_norm_N_H_refit_eval"] == 7.0
    assert updates["S_norm_N_refit_eval"] == 7.0
    assert metric["components"] == {
        "N_H_outer_eval": 2.0,
        "N_grad": 3.0,
        "N_metric": 5.0,
        "N_H_refit_eval": 7.0,
    }

    metric2, updates2, statuses2 = enrich.normalized_measurement_work_from_explicit_row(
        row={
            "S_norm_N_H_outer_eval": 2,
            "S_norm_N_H_eval": -200,
            "S_norm_N_grad": 3,
            "S_norm_N_metric": 5,
            "S_norm_N_H_refit_eval": 7,
            "S_norm_N_refit_eval": -700,
        },
        raw_proxy={},
    )

    assert statuses2["S_norm"] == "ok"
    assert updates2["S_norm"] == 17.0
    assert metric2["component_sources"]["N_H_outer_eval"] == "S_norm_N_H_outer_eval"


def test_grouped_measurement_proxy_requires_explicit_components() -> None:
    metric, updates, statuses = enrich.grouped_measurement_proxy_from_explicit_row(
        row={
            "S_grp_H_outer": 11,
            "S_grp_grad": 13,
            "S_grp_metric": 17,
            "S_grp_H_refit": 19,
        },
    )

    assert statuses["S_grp"] == "ok"
    assert updates["S_grp_total"] == 60.0
    assert updates["S_grp_H_outer"] == 11.0
    assert metric["components"]["S_grp_H_refit"] == 19.0

    missing, missing_updates, missing_statuses = enrich.grouped_measurement_proxy_from_explicit_row(
        row={
            "S_norm_N_H_outer_eval": 1,
            "S_norm_N_grad": 2,
            "S_norm_N_metric": 3,
            "S_norm_N_H_refit_eval": 4,
        },
    )

    assert missing_statuses["S_grp"] == "missing_grouped_measurement_breakdown"
    assert missing_updates == {}
    assert missing["S_grp_total"] is None


def test_hamiltonian_grouped_proxy_uses_qwc_coefficients_not_raw_term_count() -> None:
    proxy = enrich._hamiltonian_grouped_measurement_proxy(_toy_hamiltonian())

    assert proxy["term_count"] == 2
    assert proxy["group_count"] == 1
    assert proxy["group_basis_keys"] == ["xz"]
    assert proxy["C_grp"] == 25.0


def test_grouped_statevector_variance_single_pauli_variance() -> None:
    poly = PauliPolynomial("JW", [PauliTerm(1, ps="x", pc=1.0)])
    psi_zero = np.asarray([1.0, 0.0], dtype=complex)

    proxy = enrich._pauli_polynomial_grouped_statevector_variance_proxy(
        poly,
        psi_zero,
        observable_kind="unit_test_x",
    )

    assert proxy["model"] == "deterministic_greedy_qwc_grouped_statevector_variance_proxy_v1"
    assert proxy["term_count"] == 1
    assert proxy["group_count"] == 1
    assert proxy["group_basis_keys"] == ["x"]
    assert np.isclose(proxy["group_variances"][0], 1.0)
    assert np.isclose(proxy["C_var"], 1.0)


def test_grouped_statevector_variance_zero_for_eigenstate() -> None:
    poly = PauliPolynomial("JW", [PauliTerm(1, ps="z", pc=2.0)])
    psi_zero = np.asarray([1.0, 0.0], dtype=complex)

    proxy = enrich._pauli_polynomial_grouped_statevector_variance_proxy(
        poly,
        psi_zero,
        observable_kind="unit_test_z_eigenstate",
    )

    assert proxy["group_basis_keys"] == ["z"]
    assert np.isclose(proxy["group_variances"][0], 0.0)
    assert np.isclose(proxy["C_var"], 0.0)


def test_grouped_statevector_variance_multi_term_group() -> None:
    poly = PauliPolynomial(
        "JW",
        [
            PauliTerm(2, ps="ze", pc=1.0),
            PauliTerm(2, ps="ez", pc=2.0),
            PauliTerm(2, ps="ee", pc=99.0),
        ],
    )
    psi_plus_plus = np.ones(4, dtype=complex) / 2.0

    proxy = enrich._pauli_polynomial_grouped_statevector_variance_proxy(
        poly,
        psi_plus_plus,
        observable_kind="unit_test_grouped_z_sum",
    )

    assert proxy["term_count"] == 2
    assert proxy["group_count"] == 1
    assert proxy["group_basis_keys"] == ["zz"]
    assert np.isclose(proxy["group_expectations"][0], 0.0)
    assert np.isclose(proxy["group_second_moments"][0], 5.0)
    assert np.isclose(proxy["group_variances"][0], 5.0)
    assert np.isclose(proxy["C_var"], 5.0)


def test_grouped_measurement_proxy_fixed_vqe_promotes_hamiltonian_energy_work() -> None:
    context = SimpleNamespace(hamiltonian=_toy_hamiltonian())

    metric, updates, statuses = enrich.grouped_measurement_proxy_from_row_and_context(
        algorithm_id="static_family_informed_vqe",
        row={"energy_eval_count_proxy": 2, "shots_total": 999, "hamiltonian_pauli_term_count": 999},
        context=context,
    )

    assert statuses["S_grp"] == "ok"
    assert updates["S_grp_H_outer"] == 50.0
    assert updates["S_grp_grad"] == 0.0
    assert updates["S_grp_metric"] == 0.0
    assert updates["S_grp_H_refit"] == 0.0
    assert updates["S_grp_total"] == 50.0
    assert metric["hamiltonian_observable_proxy"]["C_grp"] == 25.0


def test_adaptive_gradient_component_reconstructs_non_geo_scan_schedule() -> None:
    spec = enrich._spec_by_case_id("bose_hubbard", "bose_hubbard_L2")
    context = enrich._resolve_context(spec)
    pool = enrich._adaptive_pool_for_grouped_measurement("static_qubit_qeb_adapt_vqe", context)

    cost, proxy = enrich._adaptive_gradient_grouped_component(
        algorithm_id="static_qubit_qeb_adapt_vqe",
        row={
            "gradient_scan_count_proxy": 1,
            "gradient_operator_probe_count_proxy": len(pool),
            "adapt_history": [],
        },
        context=context,
    )

    assert proxy["status"] == "ok"
    assert proxy["event_count"] == len(pool)
    assert cost is not None
    assert cost > 0.0


def test_grouped_measurement_proxy_non_geo_adaptive_promotes_with_reconstructed_gradient() -> None:
    spec = enrich._spec_by_case_id("bose_hubbard", "bose_hubbard_L2")
    context = enrich._resolve_context(spec)
    pool = enrich._adaptive_pool_for_grouped_measurement("static_qubit_qeb_adapt_vqe", context)

    metric, updates, statuses = enrich.grouped_measurement_proxy_from_row_and_context(
        algorithm_id="static_qubit_qeb_adapt_vqe",
        row={
            "selected_operator_count": 0,
            "energy_eval_count_proxy": 1,
            "gradient_scan_count_proxy": 1,
            "gradient_operator_probe_count_proxy": len(pool),
            "metric_operator_probe_count_proxy": 0,
            "adapt_history": [],
        },
        context=context,
    )

    assert statuses["S_grp"] == "ok"
    assert updates["S_grp_H_outer"] > 0.0
    assert updates["S_grp_grad"] > 0.0
    assert updates["S_grp_metric"] == 0.0
    assert updates["S_grp_H_refit"] == 0.0
    assert updates["S_grp_total"] == (
        updates["S_grp_H_outer"] + updates["S_grp_grad"]
    )
    assert metric["gradient_observable_proxy"]["event_count"] == len(pool)


def test_grouped_measurement_proxy_local_full_meta_append_promotes_with_reconstructed_gradient() -> None:
    spec = enrich._spec_by_case_id("bose_hubbard", "bose_hubbard_L2")
    context = enrich._resolve_context(spec)
    pool = enrich._adaptive_pool_for_grouped_measurement("static_full_meta_append_adapt_vqe", context)

    metric, updates, statuses = enrich.grouped_measurement_proxy_from_row_and_context(
        algorithm_id="static_full_meta_append_adapt_vqe",
        row={
            "selected_operator_count": 0,
            "energy_eval_count_proxy": 1,
            "gradient_scan_count_proxy": 1,
            "gradient_operator_probe_count_proxy": len(pool),
            "metric_operator_probe_count_proxy": 0,
            "adapt_history": [],
        },
        context=context,
    )

    assert statuses["S_grp"] == "ok"
    assert updates["S_grp_H_outer"] > 0.0
    assert updates["S_grp_grad"] > 0.0
    assert updates["S_grp_metric"] == 0.0
    assert updates["S_grp_H_refit"] == 0.0
    assert metric["gradient_observable_proxy"]["event_count"] == len(pool)
    assert metric["gradient_observable_proxy"]["event_model"] == "with_replacement_reconstructed_from_adapt_history"
    assert metric["metric_observable_proxy"]["status"] == "semantic_zero"


def test_grouped_measurement_proxy_pos_geo_promotes_with_explicit_metric_event_schedule() -> None:
    spec = enrich._spec_by_case_id("bose_hubbard", "bose_hubbard_L2")
    context = enrich._resolve_context(spec)
    pool = enrich._adaptive_pool_for_grouped_measurement("static_pos_geo_adapt_vqe", context)
    pool_labels = [str(candidate.label) for candidate in pool]
    selected = pool_labels[0]
    selector_pair_count = len(pool_labels) * (len(pool_labels) + 1) // 2

    metric, updates, statuses = enrich.grouped_measurement_proxy_from_row_and_context(
        algorithm_id="static_pos_geo_adapt_vqe",
        row={
            "selected_operator_count": 1,
            "energy_eval_count_proxy": 1,
            "gradient_scan_count_proxy": 1,
            "gradient_operator_probe_count_proxy": len(pool_labels),
            "metric_operator_probe_count_proxy": selector_pair_count + 1,
            "adapt_history": [
                {
                    "iteration": 0,
                    "candidate_count_scored": len(pool_labels),
                    "selected_batch_labels": [selected],
                    "selector_metric_candidate_labels": pool_labels,
                    "selector_metric_probe_count": selector_pair_count,
                    "qngd_metric_event_blocks": [
                        {
                            "block_kind": "pos_geo_position_trial_qngd_metric",
                            "selected_labels": [selected],
                            "metric_eval_count": 1,
                            "metric_operator_probe_count": 1,
                        }
                    ],
                }
            ],
        },
        context=context,
    )

    assert statuses["S_grp"] == "ok"
    assert updates["S_grp_grad"] > 0.0
    assert updates["S_grp_metric"] > 0.0
    assert updates["S_grp_total"] == (
        updates["S_grp_H_refit"] + updates["S_grp_grad"] + updates["S_grp_metric"]
    )
    assert metric["metric_observable_proxy"]["model"] == "symmetrized_generator_product_grouped_proxy_v1"
    assert metric["metric_observable_proxy"]["exactness"] == "proxy_not_dressed_circuit_metric_measurement_product_term_only"


def test_grouped_measurement_proxy_pos_geo_reconstructs_final_stop_metric_scan() -> None:
    spec = enrich._spec_by_case_id("bose_hubbard", "bose_hubbard_L2")
    context = enrich._resolve_context(spec)
    pool = enrich._adaptive_pool_for_grouped_measurement("static_pos_geo_adapt_vqe", context)
    pool_labels = [str(candidate.label) for candidate in pool]
    selected = pool_labels[0]
    selector_pair_count = len(pool_labels) * (len(pool_labels) + 1) // 2
    # The immediate-repeat rule is post-score, so the final Geo stop check
    # still solves the metric over the complete scored pool.
    final_labels = list(pool_labels)
    final_pair_count = len(final_labels) * (len(final_labels) + 1) // 2

    metric, updates, statuses = enrich.grouped_measurement_proxy_from_row_and_context(
        algorithm_id="static_pos_geo_adapt_vqe",
        row={
            "selected_operator_count": 1,
            "selected_operators": [selected],
            "energy_eval_count_proxy": 1,
            "gradient_scan_count_proxy": 1,
            "gradient_operator_probe_count_proxy": len(pool_labels),
            "metric_operator_probe_count_proxy": selector_pair_count + 1 + final_pair_count,
            "adapt_stop_reason": "geo_natural_gradient_norm_threshold",
            "adapt_history": [
                {
                    "iteration": 0,
                    "candidate_count_scored": len(pool_labels),
                    "selected_batch_labels": [selected],
                    "selector_metric_candidate_labels": pool_labels,
                    "selector_metric_probe_count": selector_pair_count,
                    "qngd_metric_event_blocks": [
                        {
                            "block_kind": "pos_geo_position_trial_qngd_metric",
                            "selected_labels": [selected],
                            "metric_eval_count": 1,
                            "metric_operator_probe_count": 1,
                        }
                    ],
                }
            ],
        },
        context=context,
    )

    assert statuses["S_grp"] == "ok"
    assert updates["S_grp_metric"] > 0.0
    blocks = metric["metric_observable_proxy"]["blocks"]
    assert blocks[-1]["block_kind"] == "geo_final_stop_selector_metric"
    assert blocks[-1]["label_count"] == len(pool_labels)
    assert blocks[-1]["metric_operator_probe_count"] == final_pair_count


def test_grouped_measurement_proxy_adaptive_explicit_grad_without_metric_stays_partial() -> None:
    context = SimpleNamespace(hamiltonian=_toy_hamiltonian())

    metric, updates, statuses = enrich.grouped_measurement_proxy_from_row_and_context(
        algorithm_id="static_pos_geo_adapt_vqe",
        row={
            "selected_operator_count": 1,
            "energy_eval_count_proxy": 2,
            "S_grp_grad": 7,
        },
        context=context,
    )

    assert statuses["S_grp"] == "partial_grouped_measurement_breakdown"
    assert updates["S_grp_grad"] == 7.0
    assert "S_grp_metric" not in updates
    assert "S_grp_total" not in updates
    assert metric["missing_components"] == ["S_grp_metric"]


def test_grouped_measurement_proxy_adaptive_promotes_when_gradient_metric_are_explicit() -> None:
    context = SimpleNamespace(hamiltonian=_toy_hamiltonian())

    metric, updates, statuses = enrich.grouped_measurement_proxy_from_row_and_context(
        algorithm_id="static_qiskit_adapt_vqe",
        row={
            "selected_operator_count": 1,
            "energy_eval_count_proxy": 2,
            "S_grp_grad": 7,
        },
        context=context,
    )

    assert statuses["S_grp"] == "ok"
    assert updates["S_grp_total"] == 57.0
    assert metric["components"]["S_grp_H_refit"] == 50.0


def test_grouped_measurement_proxy_qiskit_adapt_positive_metric_count_stays_partial_without_metric_component() -> None:
    context = SimpleNamespace(hamiltonian=_toy_hamiltonian())

    metric, updates, statuses = enrich.grouped_measurement_proxy_from_row_and_context(
        algorithm_id="static_qiskit_adapt_vqe",
        row={
            "selected_operator_count": 1,
            "energy_eval_count_proxy": 2,
            "S_grp_grad": 7,
            "metric_operator_probe_count_proxy": 1,
        },
        context=context,
    )

    assert statuses["S_grp"] == "partial_grouped_measurement_breakdown"
    assert updates["S_grp_grad"] == 7.0
    assert "S_grp_metric" not in updates
    assert "S_grp_total" not in updates
    assert metric["metric_observable_proxy"]["status"] == "unexpected_metric_probe_count_for_metricless_method"


def test_grouped_measurement_proxy_adaptive_rejects_negative_selected_count() -> None:
    context = SimpleNamespace(hamiltonian=_toy_hamiltonian())

    metric, updates, statuses = enrich.grouped_measurement_proxy_from_row_and_context(
        algorithm_id="static_qiskit_adapt_vqe",
        row={
            "selected_operator_count": -1,
            "energy_eval_count_proxy": 2,
            "S_grp_grad": 7,
            "S_grp_metric": 0,
        },
        context=context,
    )

    assert statuses["S_grp"] == "invalid_grouped_measurement_value"
    assert updates == {}
    assert metric["reason"] == "negative_selected_operator_count"


def test_algorithmic_measurement_work_uses_only_event_granular_fields() -> None:
    metric, updates, statuses = enrich.algorithmic_measurement_work_from_row(
        row={
            "S_alg_N_H_outer_eval": 2,
            "S_alg_N_grad_probe": 3,
            "S_alg_N_metric_probe": 5,
            "S_alg_N_H_refit_eval": 7,
            "S_alg_N_other_quantum": 0,
            "S_norm": 999,
        },
        raw_proxy={"shots_total": 1234},
    )

    assert statuses["S_alg"] == "ok"
    assert updates["S_alg"] == 17.0
    assert metric["components"] == {
        "N_H_outer_eval": 2.0,
        "N_grad_probe": 3.0,
        "N_metric_probe": 5.0,
        "N_H_refit_eval": 7.0,
    }
    assert metric["legacy_raw_proxy"]["shots_total"] == 1234.0


def test_algorithmic_measurement_work_rejects_legacy_s_norm_as_table_cost() -> None:
    metric, updates, statuses = enrich.algorithmic_measurement_work_from_row(
        row={"S_norm": 17, "shots_total": 1234},
        raw_proxy={"shots_total": 1234},
    )

    assert statuses["S_alg"] == "legacy_proxy_not_event_ledger"
    assert updates == {}
    assert metric["S_alg"] is None


def test_table_i_event_ledger_replay_fixed_vqe_promotes_s_alg() -> None:
    row = {"energy_eval_count_proxy": 17, "shots_total": 999}
    ledger, status = enrich._table_i_event_ledger_from_comparator_row(
        algorithm_id="static_family_informed_vqe",
        row=row,
    )

    assert status == "ok"
    assert ledger is not None
    assert ledger["schema"] == enrich.TABLE_I_EVENT_LEDGER_SCHEMA
    assert ledger["component_totals"] == {
        "N_H_outer_eval": 17.0,
        "N_grad_probe": 0.0,
        "N_metric_probe": 0.0,
        "N_H_refit_eval": 0.0,
        "N_other_quantum": 0.0,
    }
    metric, updates, statuses = enrich.algorithmic_measurement_work_from_row(
        row={"table_i_measurement_event_ledger": ledger, "shots_total": 999},
        raw_proxy={"shots_total": 999},
    )
    assert statuses["S_alg"] == "ok"
    assert updates["S_alg"] == 17.0
    assert metric["source_kind"] == "event_ledger"


def test_table_i_event_ledger_replay_adapt_partitions_selection_work() -> None:
    ledger, status = enrich._table_i_event_ledger_from_comparator_row(
        algorithm_id="static_pos_geo_adapt_vqe",
        row={
            "selected_operator_count": 3,
            "energy_eval_count_proxy": 100,
            "gradient_operator_probe_count_proxy": 20,
            "metric_operator_probe_count_proxy": 400,
        },
    )

    assert status == "ok"
    assert ledger is not None
    assert ledger["component_totals"] == {
        "N_H_outer_eval": 0.0,
        "N_grad_probe": 20.0,
        "N_metric_probe": 400.0,
        "N_H_refit_eval": 100.0,
        "N_other_quantum": 0.0,
    }
    metric, updates, statuses = enrich.algorithmic_measurement_work_from_row(
        row={"table_i_measurement_event_ledger": ledger},
        raw_proxy={},
    )
    assert statuses["S_alg"] == "ok"
    assert updates["S_alg"] == 520.0


def test_table_i_event_ledger_replay_local_full_meta_append_is_adaptive_metricless() -> None:
    ledger, status = enrich._table_i_event_ledger_from_comparator_row(
        algorithm_id="static_full_meta_append_adapt_vqe",
        row={
            "selected_operator_count": 2,
            "energy_eval_count_proxy": 11,
            "gradient_operator_probe_count_proxy": 13,
            "metric_operator_probe_count_proxy": 0,
        },
    )

    assert status == "ok"
    assert ledger is not None
    assert ledger["component_totals"] == {
        "N_H_outer_eval": 0.0,
        "N_grad_probe": 13.0,
        "N_metric_probe": 0.0,
        "N_H_refit_eval": 11.0,
        "N_other_quantum": 0.0,
    }


def test_table_i_threshold_cost_accepts_native_first_hit_components() -> None:
    cost = enrich.table_i_threshold_cost_from_row(
        algorithm_id="static_pos_geo_adapt_vqe",
        threshold=1e-6,
        record={"record_id": "r1", "case_id": "c1"},
        row={
            "source": "native_adaptive_iteration",
            "first_hit_semantics": "native_first_crossing_after_adapt_iteration",
            "abs_delta_e": 5e-7,
            "selected_operator_count": 2,
            "energy_eval_count_proxy": 11,
            "gradient_operator_probe_count_proxy": 13,
            "metric_operator_probe_count_proxy": 17,
            "shots_total": 999999,
        },
    )

    assert cost["threshold_status"] == "ok_native_first_hit"
    assert cost["S_alg"] == 41.0
    assert cost["S_var"] is None
    assert cost["S_var_status"] == "missing_threshold_state"
    assert cost["N_metric"] == 17.0
    assert cost["metric_fraction"] == 17.0 / 41.0
    assert cost["record_id"] == "r1"


def test_table_i_threshold_cost_promotes_threshold_statevector_variance_components() -> None:
    cost = enrich.table_i_threshold_cost_from_row(
        algorithm_id="static_pos_geo_adapt_vqe",
        threshold=1e-6,
        record={"record_id": "r1", "case_id": "c1"},
        row={
            "source": "native_adaptive_iteration",
            "first_hit_semantics": "native_first_crossing_after_adapt_iteration",
            "abs_delta_e": 5e-7,
            "selected_operator_count": 2,
            "energy_eval_count_proxy": 11,
            "gradient_operator_probe_count_proxy": 13,
            "metric_operator_probe_count_proxy": 17,
            "statevector_variance_metric": {
                "status": "ok",
                "state_scope": "threshold_first_hit_state",
                "components": {
                    "S_var_H_outer": 2,
                    "S_var_grad": 3,
                    "S_var_metric": 5,
                    "S_var_H_refit": 7,
                },
            },
        },
    )

    assert cost["threshold_status"] == "ok_native_first_hit"
    assert cost["S_alg"] == 41.0
    assert cost["S_var"] == 17.0
    assert cost["S_phys_var"] == 17.0
    assert cost["S_var_status"] == "ok"
    assert cost["S_var_components"]["S_var_metric"] == 5.0



def test_table_i_threshold_cost_rejects_terminal_s_var_as_native_first_hit_cost() -> None:
    cost = enrich.table_i_threshold_cost_from_row(
        algorithm_id="static_pos_geo_adapt_vqe",
        threshold=1e-6,
        row={
            "source": "native_adaptive_iteration",
            "first_hit_semantics": "native_first_crossing_after_adapt_iteration",
            "abs_delta_e": 5e-7,
            "selected_operator_count": 2,
            "energy_eval_count_proxy": 11,
            "gradient_operator_probe_count_proxy": 13,
            "metric_operator_probe_count_proxy": 17,
            "statevector_variance_metric": {
                "status": "ok",
                "state_scope": "terminal_final_state",
                "components": {
                    "S_var_H_outer": 2,
                    "S_var_grad": 3,
                    "S_var_metric": 5,
                    "S_var_H_refit": 7,
                },
            },
        },
    )

    assert cost["threshold_status"] == "ok_native_first_hit"
    assert cost["S_var"] is None
    assert cost["S_var_status"] == "missing_threshold_state"


def test_table_i_threshold_cost_treats_adaptive_terminal_fallback_as_upper_bound() -> None:
    cost = enrich.table_i_threshold_cost_from_row(
        algorithm_id="static_pos_geo_adapt_vqe",
        threshold=1e-6,
        row={
            "source": "terminal_row_fallback",
            "first_hit_semantics": "terminal_upper_bound_not_native_first_hit",
            "abs_delta_e": 5e-7,
            "selected_operator_count": 2,
            "energy_eval_count_proxy": 11,
            "gradient_operator_probe_count_proxy": 13,
            "metric_operator_probe_count_proxy": 17,
        },
    )

    assert cost["threshold_status"] == "terminal_upper_bound_missing_native_first_hit"
    assert cost["S_alg"] is None
    assert cost["S_var"] is None
    assert cost["S_var_status"] == "missing_threshold_state"


def test_table_i_threshold_cost_accepts_adaptive_final_ansatz_cost_for_miss() -> None:
    cost = enrich.table_i_threshold_cost_from_row(
        algorithm_id="static_pos_geo_adapt_vqe",
        threshold=1e-6,
        row={
            "source": "terminal_row_fallback",
            "first_hit_semantics": "terminal_upper_bound_not_native_first_hit",
            "abs_delta_e": 5e-4,
            "selected_operator_count": 2,
            "energy_eval_count_proxy": 11,
            "gradient_operator_probe_count_proxy": 13,
            "metric_operator_probe_count_proxy": 17,
            "compiled_count_2q_total": 8,
            "compiled_depth_2q_total": 8,
            "compiled_depth_total": 44,
            "compiled_circuit_stats_status": "ok",
            "first_hit_cost_source_kind": "qiskit_compiled_final_ansatz_circuit",
            "compiled_resource_qiskit_validated": True,
        },
    )

    assert cost["threshold_status"] == "not_reached_final_ansatz"
    assert cost["resource_display_allowed"] is True
    assert cost["compiled_resource_validation_status"] == "ok"
    assert cost["first_hit_cost_source_kind"] == "qiskit_compiled_final_ansatz_circuit"
    assert cost["S_alg"] == 41.0
    assert cost["count_2q"] == 8.0


def test_table_i_threshold_cost_accepts_fixed_terminal_method() -> None:
    cost = enrich.table_i_threshold_cost_from_row(
        algorithm_id="static_family_informed_vqe",
        threshold=1e-6,
        row={"abs_delta_e": 5e-7, "energy_eval_count_proxy": 19, "shots_total": 12345},
    )

    assert cost["threshold_status"] == "ok_terminal_only_method"
    assert cost["S_alg"] == 19.0
    assert cost["components"]["N_H_outer_eval"] == 19.0


def test_table_i_threshold_cost_rejects_complete_proxy_resources() -> None:
    cost = enrich.table_i_threshold_cost_from_row(
        algorithm_id="static_pos_geo_adapt_vqe",
        threshold=1e-6,
        row={
            "source": "native_adaptive_iteration",
            "first_hit_semantics": "native_first_crossing_after_adapt_iteration",
            "abs_delta_e": 5e-7,
            "selected_operator_count": 2,
            "energy_eval_count_proxy": 11,
            "gradient_operator_probe_count_proxy": 13,
            "metric_operator_probe_count_proxy": 17,
            "compiled_count_2q_total": 8,
            "compiled_depth_2q_total": 6,
            "compiled_depth_total": 44,
            "compiled_circuit_stats_status": "deterministic_pauli_rotation_proxy",
            "first_hit_cost_source_kind": "qiskit_compiled_first_hit_ansatz_circuit",
        },
    )

    assert cost["threshold_status"] == "ok_native_first_hit"
    assert cost["compiled_resource_validation_status"] == "invalid"
    assert cost["compiled_resource_validation_reason"] == "compiled_circuit_stats_status=deterministic_pauli_rotation_proxy"
    assert cost["resource_display_allowed"] is False


def test_table_i_threshold_cost_rejects_raw_proxy_only() -> None:
    cost = enrich.table_i_threshold_cost_from_row(
        algorithm_id="static_pos_geo_adapt_vqe",
        threshold=1e-6,
        row={
            "source": "native_adaptive_iteration",
            "abs_delta_e": 5e-7,
            "shots_total": 12345,
        },
    )

    assert cost["threshold_status"] == "raw_proxy_rejected"
    assert cost["S_alg"] is None


def _snake_valid_sidecar_row(**sidecar_overrides):
    sidecar = {
        "schema": "snake_first_crossing_compiled_cost_v1",
        "benchmark_id": "hh_L2_clean_weak",
        "history_position_tau": 3,
        "current_target_threshold": 2e-4,
        "primary_error_at_crossing": 1e-4,
        "compiled_count_2q_total": 64,
        "compiled_depth_2q_total": 52,
        "compiled_depth_total": 237,
        "compiled_circuit_stats_status": "ok",
        "first_hit_cost_source_kind": "snake_qiskit_compiled_first_hit_ansatz_circuit",
        "source_result_sha256": "test-hash",
        "S_alg": 97,
    }
    sidecar.update(sidecar_overrides)
    return {
        "source_result_sha256": "test-hash",
        "paper_i_first_crossing": {
            "schema": "paper_i_first_crossing_v1",
            "status": "reached",
            "reached": True,
            "tau_phys": 2e-4,
            "benchmark_id": "hh_L2_clean_weak",
            "history_position_tau": 3,
            "primary_error_at_crossing": 1e-4,
        },
        "paper_i_first_crossing_compiled_cost": sidecar,
    }


def test_table_i_threshold_cost_snake_reached_without_compiled_sidecar_is_not_promoted() -> None:
    cost = enrich.table_i_threshold_cost_from_row(
        algorithm_id="static_family_native_adapt_phase3",
        threshold=2e-4,
        record={"record_id": "r1", "case_id": "hh_L2_clean_weak"},
        row={
            "abs_delta_e": 1e-4,
            "paper_i_first_crossing": {
                "schema": "paper_i_first_crossing_v1",
                "status": "reached",
                "reached": True,
                "tau_phys": 2e-4,
                "benchmark_id": "hh_L2_clean_weak",
                "history_position_tau": 3,
                "primary_error_at_crossing": 1e-4,
            },
            "compiled_count_2q_total": 999,
            "compiled_depth_total": 888,
        },
    )

    assert cost["threshold_status"] == "snake_audited_first_crossing_cost_missing"
    assert cost["S_alg"] is None
    assert cost["cost_source"] is None


def test_table_i_threshold_cost_snake_sidecar_history_mismatch_is_invalid() -> None:
    cost = enrich.table_i_threshold_cost_from_row(
        algorithm_id="static_family_native_adapt_phase3",
        threshold=2e-4,
        record={"record_id": "r1", "case_id": "hh_L2_clean_weak"},
        row={
            "paper_i_first_crossing": {
                "schema": "paper_i_first_crossing_v1",
                "status": "reached",
                "reached": True,
                "tau_phys": 2e-4,
                "benchmark_id": "hh_L2_clean_weak",
                "history_position_tau": 3,
                "primary_error_at_crossing": 1e-4,
            },
            "paper_i_first_crossing_compiled_cost": {
                "schema": "snake_first_crossing_compiled_cost_v1",
                "benchmark_id": "hh_L2_clean_weak",
                "history_position_tau": 4,
                "current_target_threshold": 2e-4,
                "primary_error_at_crossing": 1e-4,
                "compiled_count_2q_total": 64,
                "compiled_depth_2q_total": 52,
                "compiled_depth_total": 237,
                "compiled_circuit_stats_status": "ok",
                "first_hit_cost_source_kind": "snake_qiskit_compiled_first_hit_ansatz_circuit",
                "source_result_sha256": "test-hash",
                "S_alg": 97,
            },
            "source_result_sha256": "test-hash",
        },
    )

    assert cost["threshold_status"] == "snake_audited_first_crossing_cost_invalid"
    assert cost["reason"] == "compiled_cost_history_position_tau_mismatch"


def test_table_i_threshold_cost_snake_noninteger_history_position_is_invalid() -> None:
    cost = enrich.table_i_threshold_cost_from_row(
        algorithm_id="static_family_native_adapt_phase3",
        threshold=2e-4,
        record={"record_id": "r1", "case_id": "hh_L2_clean_weak"},
        row={
            "paper_i_first_crossing": {
                "schema": "paper_i_first_crossing_v1",
                "status": "reached",
                "reached": True,
                "tau_phys": 2e-4,
                "benchmark_id": "hh_L2_clean_weak",
                "history_position_tau": 3.1,
                "primary_error_at_crossing": 1e-4,
            },
            "paper_i_first_crossing_compiled_cost": {
                "schema": "snake_first_crossing_compiled_cost_v1",
                "benchmark_id": "hh_L2_clean_weak",
                "history_position_tau": 3.1,
                "primary_error_at_crossing": 1e-4,
                "compiled_count_2q_total": 64,
                "compiled_depth_total": 237,
                "S_alg": 97,
            },
        },
    )

    assert cost["threshold_status"] == "snake_audited_first_crossing_cost_invalid"
    assert cost["reason"] == "paper_i_first_crossing_history_position_tau_missing"


def test_table_i_threshold_cost_snake_valid_sidecar_promotes_audited_costs() -> None:
    cost = enrich.table_i_threshold_cost_from_row(
        algorithm_id="static_family_native_adapt_phase3",
        threshold=2e-4,
        record={"record_id": "r1", "case_id": "hh_L2_clean_weak"},
        row={
            "paper_i_first_crossing": {
                "schema": "paper_i_first_crossing_v1",
                "status": "reached",
                "reached": True,
                "tau_phys": 2e-4,
                "benchmark_id": "hh_L2_clean_weak",
                "history_position_tau": 3,
                "primary_error_at_crossing": 1e-4,
            },
            "paper_i_first_crossing_compiled_cost": {
                "schema": "snake_first_crossing_compiled_cost_v1",
                "benchmark_id": "hh_L2_clean_weak",
                "history_position_tau": 3,
                "current_target_threshold": 2e-4,
                "primary_error_at_crossing": 1e-4,
                "compiled_count_2q_total": 64,
                "compiled_depth_2q_total": 52,
                "compiled_depth_total": 237,
                "compiled_circuit_stats_status": "ok",
                "first_hit_cost_source_kind": "snake_qiskit_compiled_first_hit_ansatz_circuit",
                "source_result_sha256": "test-hash",
                "S_alg": 97,
                "N_metric": 11,
            },
            "source_result_sha256": "test-hash",
        },
    )

    assert cost["threshold_status"] == "ok_native_first_hit"
    assert cost["cost_source"] == "snake_audited_first_crossing_compiled_cost"
    assert cost["first_hit_semantics"] == "snake_audited_history_position_tau"
    assert cost["count_2q"] == 64.0
    assert cost["depth_2q"] == 52.0
    assert cost["circuit_depth"] == 237.0
    assert cost["S_alg"] == 97.0
    assert cost["metric_fraction"] == 11.0 / 97.0
    assert cost["sidecar_validation_status"] == "ok"
    assert cost["sidecar_hash_verified"] is True
    assert cost["resource_display_allowed"] is True


def test_table_i_threshold_cost_snake_sidecar_rejects_s_norm_only_work() -> None:
    cost = enrich.table_i_threshold_cost_from_row(
        algorithm_id="static_family_native_adapt_phase3",
        threshold=2e-4,
        record={"record_id": "r1", "case_id": "hh_L2_clean_weak"},
        row=_snake_valid_sidecar_row(S_alg=None, S_norm=97),
    )

    assert cost["threshold_status"] == "snake_audited_first_crossing_cost_invalid"
    assert cost["reason"] == "compiled_cost_S_alg_missing_without_reason"
    assert cost["S_alg"] is None


def test_table_i_threshold_cost_snake_sidecar_rejects_non_snake_qiskit_source_kind() -> None:
    cost = enrich.table_i_threshold_cost_from_row(
        algorithm_id="static_family_native_adapt_phase3",
        threshold=2e-4,
        record={"record_id": "r1", "case_id": "hh_L2_clean_weak"},
        row=_snake_valid_sidecar_row(first_hit_cost_source_kind="qiskit_compiled_first_hit_ansatz_circuit"),
    )

    assert cost["threshold_status"] == "snake_audited_first_crossing_cost_invalid"
    assert cost["reason"] == "compiled_cost_source_kind_mismatch"
    assert cost["resource_display_allowed"] is False


def test_table_i_threshold_cost_snake_sidecar_rejects_primary_error_mismatch() -> None:
    cost = enrich.table_i_threshold_cost_from_row(
        algorithm_id="static_family_native_adapt_phase3",
        threshold=2e-4,
        record={"record_id": "r1", "case_id": "hh_L2_clean_weak"},
        row=_snake_valid_sidecar_row(primary_error_at_crossing=1.5e-4),
    )

    assert cost["threshold_status"] == "snake_audited_first_crossing_cost_invalid"
    assert cost["reason"] == "compiled_cost_primary_error_mismatch"
    assert cost["resource_display_allowed"] is False


def test_table_i_threshold_cost_snake_sidecar_rejects_missing_two_qubit_depth() -> None:
    row = _snake_valid_sidecar_row()
    del row["paper_i_first_crossing_compiled_cost"]["compiled_depth_2q_total"]

    cost = enrich.table_i_threshold_cost_from_row(
        algorithm_id="static_family_native_adapt_phase3",
        threshold=2e-4,
        record={"record_id": "r1", "case_id": "hh_L2_clean_weak"},
        row=row,
    )

    assert cost["threshold_status"] == "snake_audited_first_crossing_cost_invalid"
    assert cost["reason"] == "compiled_cost_required_resource_missing"
    assert cost["resource_display_allowed"] is False


def test_table_i_threshold_cost_snake_sidecar_rejects_synthetic_source() -> None:
    cost = enrich.table_i_threshold_cost_from_row(
        algorithm_id="static_family_native_adapt_phase3",
        threshold=2e-4,
        record={"record_id": "r1", "case_id": "hh_L2_clean_weak"},
        row={
            "source_result_sha256": "test-hash",
            "paper_i_first_crossing": {
                "schema": "paper_i_first_crossing_v1",
                "status": "reached",
                "reached": True,
                "tau_phys": 2e-4,
                "benchmark_id": "hh_L2_clean_weak",
                "history_position_tau": 3,
                "primary_error_at_crossing": 1e-4,
            },
            "paper_i_first_crossing_compiled_cost": {
                "schema": "snake_first_crossing_compiled_cost_v1",
                "benchmark_id": "hh_L2_clean_weak",
                "history_position_tau": 3,
                "current_target_threshold": 2e-4,
                "primary_error_at_crossing": 1e-4,
                "compiled_count_2q_total": 64,
                "compiled_depth_2q_total": 52,
                "compiled_depth_total": 237,
                "compiled_circuit_stats_status": "ok",
                "source": "live_snake_overlay_current_best_first_crossing",
                "source_result_sha256": "test-hash",
                "S_alg": 97,
            },
        },
    )

    assert cost["threshold_status"] == "snake_audited_first_crossing_cost_invalid"
    assert cost["reason"] == "forbidden_resource_source_live_snake_overlay"
    assert cost["resource_display_allowed"] is False


def test_table_i_threshold_cost_snake_sidecar_rejects_missing_hash() -> None:
    cost = enrich.table_i_threshold_cost_from_row(
        algorithm_id="static_family_native_adapt_phase3",
        threshold=2e-4,
        record={"record_id": "r1", "case_id": "hh_L2_clean_weak"},
        row={
            "paper_i_first_crossing": {
                "schema": "paper_i_first_crossing_v1",
                "status": "reached",
                "reached": True,
                "tau_phys": 2e-4,
                "benchmark_id": "hh_L2_clean_weak",
                "history_position_tau": 3,
                "primary_error_at_crossing": 1e-4,
            },
            "paper_i_first_crossing_compiled_cost": {
                "schema": "snake_first_crossing_compiled_cost_v1",
                "benchmark_id": "hh_L2_clean_weak",
                "history_position_tau": 3,
                "current_target_threshold": 2e-4,
                "primary_error_at_crossing": 1e-4,
                "compiled_count_2q_total": 64,
                "compiled_depth_2q_total": 52,
                "compiled_depth_total": 237,
                "compiled_circuit_stats_status": "ok",
                "first_hit_cost_source_kind": "snake_qiskit_compiled_first_hit_ansatz_circuit",
                "S_alg": 97,
            },
        },
    )

    assert cost["threshold_status"] == "snake_audited_first_crossing_cost_invalid"
    assert cost["reason"] == "compiled_cost_source_result_sha256_missing"


def test_table_i_threshold_cost_snake_sidecar_rejects_impossible_depth_ordering() -> None:
    cost = enrich.table_i_threshold_cost_from_row(
        algorithm_id="static_family_native_adapt_phase3",
        threshold=2e-4,
        record={"record_id": "r1", "case_id": "hh_L2_clean_weak"},
        row={
            "source_result_sha256": "test-hash",
            "paper_i_first_crossing": {
                "schema": "paper_i_first_crossing_v1",
                "status": "reached",
                "reached": True,
                "tau_phys": 2e-4,
                "benchmark_id": "hh_L2_clean_weak",
                "history_position_tau": 3,
                "primary_error_at_crossing": 1e-4,
            },
            "paper_i_first_crossing_compiled_cost": {
                "schema": "snake_first_crossing_compiled_cost_v1",
                "benchmark_id": "hh_L2_clean_weak",
                "history_position_tau": 3,
                "current_target_threshold": 2e-4,
                "primary_error_at_crossing": 1e-4,
                "compiled_count_2q_total": 64,
                "compiled_depth_2q_total": 300,
                "compiled_depth_total": 237,
                "compiled_circuit_stats_status": "ok",
                "first_hit_cost_source_kind": "snake_qiskit_compiled_first_hit_ansatz_circuit",
                "source_result_sha256": "test-hash",
                "S_alg": 97,
            },
        },
    )

    assert cost["threshold_status"] == "snake_audited_first_crossing_cost_invalid"
    assert cost["reason"] == "compiled_depth_total_less_than_two_qubit_depth"


def test_state_reconstruction_treats_full_meta_append_as_local_adapt(monkeypatch) -> None:
    candidate = enrich.build_pairwise_qubit_excitation_pool(2)[0]
    context = SimpleNamespace(
        layout=SimpleNamespace(total_qubits=2),
        reference_state=SimpleNamespace(build_state=lambda: np.eye(4, dtype=complex)[1]),
    )

    monkeypatch.setattr(enrich, "build_full_meta_candidate_pool", lambda ctx: (candidate,))

    def _fake_compile(ctx, row, *, selected=None, source_kind=None):  # noqa: ANN001, ANN201
        assert [item.label for item in selected] == [candidate.label]
        assert source_kind == "qiskit_compiled_final_ansatz_circuit"
        return (
            {"status": "ok", "compiled_depth_2q_total": 4, "depth_2q_semantics": "test"},
            {"compiled_depth_2q_total": 4.0},
            {"compiled_depth_2q_total": "ok"},
        )

    monkeypatch.setattr(enrich, "_qiskit_compile_selected_pauli_groups", _fake_compile)

    psi, depth_metric, updates, statuses = enrich._reconstruct_state_and_depth(
        "static_full_meta_append_adapt_vqe",
        context,
        {"selected_operators": [candidate.label], "theta": [0.0]},
    )

    assert psi is not None
    assert depth_metric["status"] == "ok"
    assert updates["compiled_depth_2q_total"] == 4.0
    assert statuses["compiled_depth_2q_total"] == "ok"


def test_metric_enrichment_preserves_grouped_exact_mode_for_qiskit_compile(monkeypatch) -> None:
    candidate = SimpleNamespace(
        label="grouped_parent",
        polynomial=PauliPolynomial(
            "JW",
            [
                PauliTerm(1, ps="x", pc=0.5),
                PauliTerm(1, ps="z", pc=-0.25),
            ],
        ),
        pauli_labels_exyz=("x", "z"),
        execution_mode="grouped_exact",
    )
    context = SimpleNamespace(
        layout=SimpleNamespace(total_qubits=1),
        reference_state=SimpleNamespace(build_state=lambda: np.asarray([1.0, 0.0], dtype=complex)),
    )

    def fake_compile(*, ops, num_qubits, reference_state, source_kind):  # noqa: ANN001, ANN201
        assert len(ops) == 1
        assert ops[0].execution_mode == "grouped_exact"
        return {
            "compiled_depth_total": 7,
            "compiled_depth_2q_total": 3,
            "compiled_count_2q_total": 4,
            "compiled_op_counts": {"cx": 4},
            "compiled_basis_gates": ["cx", "rz"],
            "compiled_depth_2q_semantics": "test",
            "grouped_exact_synthesis_id": "exact_test_v1",
            "generator_coefficients_sha256": "coeff-hash",
            "operator_synthesis": [{"synthesis": "active_support_unitary_exact"}],
        }

    monkeypatch.setattr(enrich, "compile_table_i_ansatz_terms", fake_compile)
    metric, updates, statuses = enrich._qiskit_compile_selected_pauli_groups(
        context,
        {"theta": [0.0]},
        selected=(candidate,),
    )

    assert metric["status"] == "ok"
    assert metric["grouped_exact_synthesis_id"] == "exact_test_v1"
    assert metric["generator_coefficients_sha256"] == "coeff-hash"
    assert updates["compiled_resource_qiskit_validated"] is True
    assert statuses["compiled_depth_2q_total"] == "ok"


def test_metric_enrichment_refuses_label_only_grouped_exact_cost_substitution() -> None:
    context = SimpleNamespace(
        layout=SimpleNamespace(total_qubits=1),
        reference_state=SimpleNamespace(build_state=lambda: np.asarray([1.0, 0.0], dtype=complex)),
    )
    with pytest.raises(enrich.NotReconstructable) as exc_info:
        enrich._qiskit_compile_selected_pauli_groups(
            context,
            {
                "theta": [0.0],
                "selected_operator_pauli_labels_exyz": [["x", "z"]],
                "selected_operator_execution_modes": ["grouped_exact"],
            },
            selected=None,
        )
    assert exc_info.value.status == "not_reconstructable_grouped_exact_coefficients_missing"


def test_metric_enrichment_classifies_full_meta_geo_as_local_adapt() -> None:
    assert "static_geo_adapt_vqe" in enrich._ADAPT_VARIANT_IDS


def test_physical_measurement_work_from_grouped_replay_promotes_s_l2_only() -> None:
    metric, updates, statuses = enrich.physical_measurement_work_from_grouped_replay(
        grouped_metric={
            "schema": enrich.GROUPED_MEASUREMENT_PROXY_SCHEMA,
            "status": "ok",
            "components": {
                "S_grp_H_outer": 2,
                "S_grp_grad": 3,
                "S_grp_metric": 5,
                "S_grp_H_refit": 7,
            },
            "component_sources": {"S_grp_grad": "unit"},
        }
    )

    assert statuses == {
        "S_phys": "missing_fresh_variance_event_components",
        "S_l2": "ok",
        "S_var": "missing_statevector_variance_event_components",
        "S_phys_var": "missing_statevector_variance_event_components",
    }
    assert updates["S_l2"] == 17.0
    assert updates["S_l2_grad"] == 3.0
    assert "S_var" not in updates
    assert metric["S_l2"]["source_kind"] == "exact_bench_noiseless_grouped_l2_replay_from_context"
    assert metric["S_var"]["status"] == "missing_statevector_variance_event_components"


def test_physical_measurement_work_from_grouped_replay_promotes_s_var_when_event_components_exist() -> None:
    metric, updates, statuses = enrich.physical_measurement_work_from_grouped_replay(
        grouped_metric={
            "schema": enrich.GROUPED_MEASUREMENT_PROXY_SCHEMA,
            "status": "ok",
            "components": {
                "S_grp_H_outer": 2,
                "S_grp_grad": 3,
                "S_grp_metric": 5,
                "S_grp_H_refit": 7,
            },
        },
        statevector_variance_metric={
            "status": "ok",
            "components": {
                "S_var_H_outer": 11,
                "S_var_grad": 13,
                "S_var_metric": 17,
                "S_var_H_refit": 19,
            },
            "component_sources": {"S_var_grad": "unit_variance_event_replay"},
        },
    )

    assert statuses["S_l2"] == "ok"
    assert statuses["S_var"] == "ok"
    assert statuses["S_phys_var"] == "ok"
    assert updates["S_l2"] == 17.0
    assert updates["S_var"] == 60.0
    assert updates["S_var_grad"] == 13.0
    assert metric["S_var"]["schema"] == enrich.STATEVECTOR_VARIANCE_METRIC_SCHEMA
    assert metric["S_var"]["source_kind"] == "exact_bench_noiseless_grouped_statevector_variance_replay_from_event_statevectors"
