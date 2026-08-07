from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.reporting.paper_i_geo_compact_method_overlays import (
    OverlayBlocked,
    _aligned_runtime_split_execution_modes,
    _append_plateau_overlay,
    _derive_execution_mode_from_contract,
    _native_curve_and_reconstruction,
    _parse_snake_checkpoint_curve,
    _snake_history_round_audit,
    _validate_record_contract,
    index_explicit_roots,
    select_explicit_source,
)


def _contract_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "method_key": "snake",
        "algorithm_id": "static_family_native_adapt_phase3",
        "optimizer": "POWELL",
        "budget": "200",
        "child_policy": "macro_only",
        "pool_contract": "full_meta_unfiltered",
        "parent_generator_policy": "full_meta_parent_macro_generators_only_all_methods",
        "generic_adapt_runtime_split_mode": "off",
        "shared_pauli_pool_mode": "off",
        "expected_horizon": "30",
        "record_id": "record",
    }
    row.update(overrides)
    return row


def test_record_contract_separates_parent_only_and_l2_child_set() -> None:
    parent = _validate_record_contract(
        {"row": _contract_row()},
        method_key="snake",
        expected_horizon=30,
        case_id="hubbard_L3_scaling_weak",
    )
    assert parent["overlay_policy_comparability"] == "parent_macro_only_matched"
    assert parent["common_pauli_child_policy"] is True

    child = _contract_row(
        child_policy="native_phase3_singleton",
        child_subset_size="1",
        parent_generator_policy=None,
        generic_adapt_runtime_split_mode="",
        expected_horizon=None,
        max_depth="50",
        snake_phase3_runtime_split_mode="shortlist_pauli_children_v1",
        snake_phase3_runtime_split_selection_mode="archival_child_set_forward_v1",
        snake_phase3_runtime_split_max_subset_size="1",
        snake_phase3_runtime_split_child_set_symmetry_policy="hard_guard",
    )
    validated = _validate_record_contract(
        {"row": child},
        method_key="snake",
        expected_horizon=50,
        case_id="hh_L2_nph2_three_model_sym_weak_weak",
    )
    assert validated["overlay_policy_comparability"] == "mixed_child_set_diagnostic"
    assert validated["common_pauli_child_policy"] is False

    legacy_append = {
        **_contract_row(),
        "method_key": "append",
        "algorithm_id": "static_full_meta_append_adapt_vqe",
        "parent_generator_policy": None,
        "expected_horizon": None,
        "max_depth": "50",
        "record_id": "legacy__fullmeta_parent",
    }
    validated_append = _validate_record_contract(
        {
            "row": legacy_append,
            "env_overlay": {
                "GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_MODE": "off",
                "GENERIC_STATIC_TABLE_HH_ADAPTIVE_POOL_PROFILE": "full_meta_unfiltered",
            },
        },
        method_key="append",
        expected_horizon=50,
        case_id="hh_L2_nph2_three_model_sym_weak_weak",
    )
    assert (
        validated_append["overlay_policy_comparability"]
        == "legacy_l2_parent_macro_only_env_verified"
    )


def test_native_l2_index_uses_internal_regime_not_duplicated_source_case_id(
    tmp_path: Path,
) -> None:
    root = tmp_path / "explicit"
    for regime in ("intermediate_weak", "strong_weak"):
        record = root / regime
        (record / "json").mkdir(parents=True)
        (record / "json" / "result.json").write_text("{}\n", encoding="utf-8")
        manifest = {
            "status": "ok",
            "returncode": 0,
            "row": _contract_row(
                case_id="hh_L2_nph2_three_model_sym_strong_weak",
                internal_regime=regime,
                child_policy="native_phase3_singleton",
            ),
        }
        (record / "cell_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
    indexed, audit = index_explicit_roots([root], method_key="snake")
    assert audit[0]["native_only_result_count"] == 2
    assert set(indexed) == {
        "hh_L2_nph2_three_model_sym_strong_weak",
        "hh_L2_nph2_three_model_sym_u8_strong_weak",
    }
    assert all(len(rows) == 1 for rows in indexed.values())


def test_snake_checkpoint_curve_requires_named_terminal_events(tmp_path: Path) -> None:
    path = tmp_path / "stdout.log"
    rows = [
        {
            "event": "hardcoded_adapt_current_checkpoint_written",
            "reason": "beam_round_done",
            "depth": k,
            "benchmark_target_abs_delta_e_current": 1.0 / k,
        }
        for k in range(1, 4)
    ]
    path.write_text(
        "noise\n" + "\n".join("AI_LOG " + json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    curve = _parse_snake_checkpoint_curve(path, initial_error=2.0)
    assert [point["k"] for point in curve] == [0, 1, 2, 3]
    assert curve[-1]["error_raw"] == pytest.approx(1.0 / 3.0)

    path.write_text(
        "\n".join("AI_LOG " + json.dumps(row) for row in (rows[0], rows[0])) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(OverlayBlocked, match="duplicate SNAKE checkpoint"):
        _parse_snake_checkpoint_curve(path, initial_error=2.0)


def test_native_terminal_structure_allows_unambiguous_singletons_only() -> None:
    history = [
        {
            "depth": 1,
            "delta_abs_prev": 2.0,
            "delta_abs_current": 1.0,
        }
    ]
    native = {
        "adapt_vqe": {
            "success": True,
            "abs_delta_e": 1.0,
            "exact_abs_delta_e_from_final_state": 1.0,
            "ansatz_depth": 1,
            "history": history,
            "operators": ["A"],
            "logical_optimal_point": [0.1],
            "optimal_point": [0.1],
            "parameterization": {
                "logical_operator_count": 1,
                "runtime_parameter_count": 1,
                "blocks": [
                    {
                        "candidate_label": "A",
                        "logical_index": 0,
                        "runtime_start": 0,
                        "runtime_count": 1,
                        "runtime_terms_exyz": [
                            {
                                "pauli_exyz": "xe",
                                "coeff_re": 0.5,
                                "coeff_im": 0.0,
                                "nq": 2,
                            }
                        ],
                    }
                ],
            },
        }
    }
    curve, audit, reconstruction = _native_curve_and_reconstruction(
        native, expected_horizon=1
    )
    assert [point["k"] for point in curve] == [0, 1]
    assert audit["status"] == "pass"
    assert reconstruction is not None
    assert reconstruction["selected_generator_semantics"][0]["execution_mode"] == "termwise_product"

    native["adapt_vqe"]["optimal_point"] = [0.1, 0.2]
    native["adapt_vqe"]["parameterization"]["runtime_parameter_count"] = 2
    block = native["adapt_vqe"]["parameterization"]["blocks"][0]
    block["runtime_count"] = 2
    block["runtime_terms_exyz"].append(
        {"pauli_exyz": "ex", "coeff_re": -0.5, "coeff_im": 0.0, "nq": 2}
    )
    _, audit, reconstruction = _native_curve_and_reconstruction(native, expected_horizon=1)
    assert reconstruction is None
    assert audit["local_table_i_compile_status"].startswith("blocked:")


def test_native_singleton_prefers_aligned_runtime_split_mode_over_equivalence() -> None:
    native = {
        "adapt_vqe": {
            "success": True,
            "abs_delta_e": 1.0,
            "exact_abs_delta_e_from_final_state": 1.0,
            "ansatz_depth": 1,
            "history": [
                {"depth": 1, "delta_abs_prev": 2.0, "delta_abs_current": 1.0}
            ],
            "operators": ["A::child_set[0]"],
            "logical_optimal_point": [0.1],
            "optimal_point": [0.1],
            "parameterization": {
                "logical_operator_count": 1,
                "runtime_parameter_count": 1,
                "blocks": [
                    {
                        "candidate_label": "A::child_set[0]",
                        "logical_index": 0,
                        "runtime_start": 0,
                        "runtime_count": 1,
                        "runtime_terms_exyz": [
                            {
                                "pauli_exyz": "xe",
                                "coeff_re": 0.5,
                                "coeff_im": 0.0,
                                "nq": 2,
                            }
                        ],
                    }
                ],
            },
            "continuation": {
                "selected_generator_metadata": [
                    {
                        "candidate_label": "A::child_set[0]",
                        "compile_metadata": {
                            "runtime_split": {
                                "mode": "shortlist_pauli_children_v1",
                                "representation": "child_set",
                                "recommended_execution_mode": "grouped_exact",
                            }
                        },
                    }
                ]
            },
        }
    }
    _, audit, reconstruction = _native_curve_and_reconstruction(native, expected_horizon=1)
    assert reconstruction is not None
    semantic = reconstruction["selected_generator_semantics"][0]
    assert semantic["execution_mode"] == "grouped_exact"
    assert semantic["execution_mode_source"] == "native_runtime_split_metadata_aligned_v1"
    assert audit["execution_mode_counts"] == {"grouped_exact": 1}


def test_snake_history_audit_uses_full_retained_history(tmp_path: Path) -> None:
    current = {
        "adapt_vqe": {
            "history_count": 3,
            "history_tail_count": 3,
            "history_tail": [
                {"post_admission_prune": {"permission_reason": "has_min_scaffold"}},
                {
                    "post_admission_prune": {
                        "permission_reason": "zero_gain_duplicate_structural_rollback"
                    }
                },
                {
                    "post_admission_prune": {
                        "permission_reason": "zero_gain_duplicate_structural_rollback"
                    }
                },
            ],
            "ansatz_depth": 1,
        }
    }
    (tmp_path / "current.json").write_text(json.dumps(current), encoding="utf-8")
    audit = _snake_history_round_audit(
        record_dir=tmp_path,
        expected_horizon=3,
        terminal_generator_count=1,
    )
    assert audit["trajectory_status"] == "ok"
    assert "zero_gain_duplicate_structural_rollback_count" not in audit
    assert audit["trajectory_semantics"] == "outer_history_rounds_not_committed_admission_count"


def test_retrieved_hh_execution_modes_match_base_pool_legal_subspace_contract() -> None:
    root = Path(
        "output/chtc_retrievals/paper_i_append_snake_completed_20260711/compact_native"
    )
    if not root.is_dir():
        pytest.skip("retrieved compact-native evidence is not present")
    expected = {
        "hh_L3_nph2_scaling_weak_weak": (2, 4),
        "hh_L3_nph2_scaling_intermediate_weak": (0, 5),
        "hh_L3_nph2_scaling_strong_weak": (0, 5),
        "hh_L3_nph2_scaling_weak_strong": (0, 4),
        "hh_L3_nph2_scaling_intermediate_strong": (4, 4),
        "hh_L3_nph2_scaling_strong_strong": (0, 5),
        "hh_L4_nph1_scaling_weak_weak": (0, 41),
        "hh_L4_nph1_scaling_intermediate_weak": (0, 15),
        "hh_L4_nph1_scaling_strong_weak": (0, 13),
        "hh_L4_nph1_scaling_weak_strong": (0, 54),
        "hh_L4_nph1_scaling_intermediate_strong": (0, 10),
        "hh_L4_nph1_scaling_strong_strong": (0, 14),
    }
    for case_id, expected_counts in expected.items():
        matches = list(root.glob(f"*__{case_id}__*__native_compact.json"))
        assert len(matches) == 1, case_id
        native = json.loads(matches[0].read_text(encoding="utf-8"))["payload"]
        modes: list[str] = []
        for block in native["adapt_vqe"]["parameterization"]["blocks"]:
            mode, source, details = _derive_execution_mode_from_contract(
                native,
                label=str(block["candidate_label"]),
                terms=block["runtime_terms_exyz"],
            )
            assert source == "hh_base_full_meta_selected_legal_subspace_execution_replay_v1"
            assert details is not None and details["schema"] == "hh_base_full_meta_selected_execution_replay_v1"
            modes.append(str(mode))
        assert (modes.count("grouped_exact"), modes.count("termwise_product")) == expected_counts


def test_retrieved_l2_modes_use_aligned_nested_runtime_split_metadata() -> None:
    root = Path(
        "output/chtc_retrievals/paper_i_append_snake_completed_20260711/compact_native"
    )
    if not root.is_dir():
        pytest.skip("retrieved compact-native evidence is not present")
    expected = {
        "intermediate_strong": (2, 2),
        "intermediate_weak": (2, 1),
        "strong_strong": (3, 1),
        "strong_weak": (3, 1),
        "weak_strong": (2, 1),
        "weak_weak": (2, 2),
    }
    for regime, expected_counts in expected.items():
        matches = list(root.glob(f"*__{regime}__snake__current_forward*__native_compact.json"))
        assert len(matches) == 1, regime
        native = json.loads(matches[0].read_text(encoding="utf-8"))["payload"]
        labels = [str(value) for value in native["adapt_vqe"]["operators"]]
        aligned = _aligned_runtime_split_execution_modes(native, labels=labels)
        modes = [str(mode) for mode, _ in aligned]
        assert (modes.count("grouped_exact"), modes.count("termwise_product")) == expected_counts


def test_explicit_repair_supersedes_source_at_same_root_precedence() -> None:
    base = {
        "result_path": Path("base.json"),
        "root_precedence": 0,
        "record_contract": {"row": {"record_id": "base"}},
    }
    repair = {
        "result_path": Path("repair.json"),
        "root_precedence": 0,
        "record_contract": {
            "row": {"record_id": "repair", "repair_source_record_id": "base"}
        },
    }
    selected = select_explicit_source({"case": [base, repair]}, "case")
    assert selected["result_path"] == Path("repair.json")

    with pytest.raises(OverlayBlocked, match="ambiguous_results"):
        select_explicit_source({"case": [base, {**repair, "record_contract": {"row": {"record_id": "other"}}}]}, "case")


def test_append_overlay_compiles_its_own_first_plateau_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result_path = tmp_path / "result" / "generic_static_single.json"
    result_path.parent.mkdir(parents=True)
    history = [
        {
            "abs_delta_e_same_cutoff_after": 1.05,
            "energy_before": 3.0,
            "energy_after": 1.05,
            "depth_after": 1,
            "selected_batch_labels": ["A"],
            "appended_operator_count": 1,
            "selected_insertion_position": None,
            "outer_hamiltonian_eval_count": 1,
            "optimizer_nfev": 2,
            "selector_gradient_probe_count": 3,
            "selector_metric_probe_count": 4,
            "optimizer_effective_success": True,
        },
        {
            "abs_delta_e_same_cutoff_after": 1.0,
            "energy_before": 1.05,
            "energy_after": 1.0,
            "depth_after": 2,
            "selected_batch_labels": ["B"],
            "appended_operator_count": 1,
            "selected_insertion_position": None,
            "outer_hamiltonian_eval_count": 1,
            "optimizer_nfev": 2,
            "selector_gradient_probe_count": 3,
            "selector_metric_probe_count": 4,
            "optimizer_effective_success": True,
        },
    ]
    result_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "case_id": "hubbard_L3_scaling_weak",
                "result": {
                    "adapt_history": history,
                    "optimizer_success_all": True,
                    "same_cutoff_exact_gs_energy": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    seed_path = result_path.parent / "runtime_seed.json"
    seed_path.write_text("{}\n", encoding="utf-8")
    observed: dict[str, int] = {}

    def fake_reconstruct(**kwargs: object) -> dict[str, object]:
        observed["history_position"] = int(kwargs["history_position"])
        return {"selected_generator_semantics": [], "selected_generator_semantics_sha256": "a"}

    monkeypatch.setattr(
        "pipelines.reporting.paper_i_geo_compact_method_overlays.reconstruct_structural_prefix",
        fake_reconstruct,
    )
    monkeypatch.setattr(
        "pipelines.reporting.paper_i_geo_compact_method_overlays.compile_prefix_qiskit",
        lambda **_: {"status": "ok", "N2q": 1, "D2q": 1, "Dcirc": 2},
    )
    overlay = _append_plateau_overlay(
        {
            "result_path": result_path,
            "record_dir": tmp_path,
            "record_contract": {
                "path": "cell_manifest.json",
                "sha256": "b" * 64,
                "row": {
                    "method_key": "append",
                    "algorithm_id": "static_full_meta_append_adapt_vqe",
                    "optimizer": "POWELL",
                    "budget": "200",
                    "child_policy": "macro_only",
                    "pool_contract": "full_meta_unfiltered",
                    "parent_generator_policy": "full_meta_parent_macro_generators_only_all_methods",
                    "generic_adapt_runtime_split_mode": "off",
                    "shared_pauli_pool_mode": "off",
                    "expected_horizon": "2",
                    "record_id": "append",
                },
            },
        },
        expected_horizon=2,
        expected_exact_energy=0.0,
        grouped_exact_max_active_qubits=8,
    )
    assert overlay["status"] == "ok"
    assert overlay["marker"]["label"] == "k_pl"
    assert overlay["marker"]["k"] == 1
    assert observed["history_position"] == 0
    assert overlay["query_ledger"]["S"] == 10
