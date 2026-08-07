from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path



ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_script_module(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_script_module(
    "paper_i_hh_spsa_budget_ladder_cell_runner",
    "chtc/phase3_optuna/run_paper_i_hh_spsa_budget_ladder_cell.py",
)
report = _load_script_module(
    "paper_i_hh_spsa_budget_ladder_report_builder",
    "pipelines/reporting/build_paper_i_hh_spsa_budget_ladder_report.py",
)


def _single_flag(args: list[str], flag: str) -> str | bool:
    values = runner.flag_values(args).get(flag)
    assert values, flag
    assert len(values) == 1
    return values[0]


def _minimal_snake_payload(*, s_norm: float = 999.0) -> dict[str, object]:
    def phase(count: int) -> dict[str, object]:
        return {
            "records_with_group_keys": count,
            "groups_total": count,
            "actual_operator_probe_count": count,
            "operator_probe_charge_basis": "logical_estimator_request_pre_grouping_v1",
            "common_exposure_stage": "post_common_eligibility_post_expansion_pre_method_filter",
            "common_exposure_policy_id": "trajectory_conditioned_full_child_common_exposure_v1",
            "expansion_policy_id": "test_expansion_policy",
            "eligibility_policy_id": "test_eligibility_policy",
            "deduplication_policy_id": "test_deduplication_policy",
            "probe_enumerator_id": "test_probe_enumerator",
        }

    return {
        "adapt_vqe": {
            "benchmark_target_abs_delta_e_current": 1.0e-3,
            "energy": -1.0,
            "exact_gs_energy": -1.001,
            "ansatz_depth": 3,
            "S_norm": s_norm,
            "controller_measurement_work_summary": {
                "legacy_fallback_used": False,
                "source_kind": "native_controller_work",
                "candidate_work_ledger_schema": "controller_candidate_work_ledger_v1",
                "candidate_work_ledger_status": "explicit_candidate_work_ledger_v1",
                "candidate_work_event_count": 4,
                "candidate_work_missing_event_count": 0,
                "candidate_count_total": 10,
                "evaluated_count_total": 10,
                "pre_shortlist_count_total": 10,
                "shortlist_size_total": 10,
                "retained_count_total": 10,
                "rejected_count_total": 0,
                "candidate_work_ledger_scope": "event_records_measured_v1",
                "candidate_work_ledger_scopes": {"event_records_measured_v1": 4},
                "by_phase": {
                    "phase0": phase(3),
                    "phase1": phase(4),
                    "phase2": phase(2),
                    "phase3": phase(1),
                },
            },
            "history": [{"depth": 1, "energy_after": -1.0, "nfev_opt": 5}],
            "resume_boundary_refit": {"executed": False},
            "final_full_refit": {"executed": False},
            "nfev_total": 20,
        }
    }


def test_row_inner_optimizer_accepts_rotosolve_for_snake_rows() -> None:
    assert runner.row_inner_optimizer({"adapt_optimizer_kind": "rotosolve"}) == "ROTOSOLVE"


def test_snake_source_locked_command_uses_row_inner_optimizer_by_default(tmp_path: Path) -> None:
    source_json = tmp_path / "source_result.json"
    source_json.write_text(json.dumps({"adapt_vqe": {"ansatz_depth": 5}}), encoding="utf-8")
    source_args = [
        "/usr/bin/python3",
        "-u",
        "-m",
        "pipelines.static_adapt.adapt_pipeline",
        "--adapt-max-depth",
        "30",
        "--adapt-maxiter",
        "800",
        "--adapt-final-refit-maxiter",
        "800",
        "--adapt-inner-optimizer",
        "SPSA",
        "--output-json",
        "/old/result.json",
        "--adapt-current-json",
        "/old/current.json",
    ]
    row = {
        "record_id": "r",
        "method_key": "snake",
        "display_regime": "weak-weak",
        "engine_key": "native_forced",
        "budget": "200",
        "source_json": str(source_json),
        "source_command_args_json": json.dumps(source_args),
        "adapt_optimizer_kind": "powell",
        "adapt_schur_warm_start_mode": "append-prune",
    }

    _source_cmd, effective_cmd, audit = runner.build_snake_source_locked_command(row, tmp_path / "out")

    assert audit["status"] == "pass"
    assert _single_flag(effective_cmd, "--adapt-inner-optimizer") == "POWELL"
    assert _single_flag(effective_cmd, "--adapt-schur-warm-start-mode") == "append-prune"
    assert audit["diagnostic_inner_optimizer"] == "POWELL"
    assert audit["diagnostic_schur_warm_start_mode"] == "append-prune"


def test_snake_source_locked_command_accepts_explicit_rotosolve_override(tmp_path: Path) -> None:
    source_json = tmp_path / "source_result.json"
    source_json.write_text(json.dumps({"adapt_vqe": {"ansatz_depth": 5}}), encoding="utf-8")
    source_args = [
        "/usr/bin/python3",
        "-u",
        "-m",
        "pipelines.static_adapt.adapt_pipeline",
        "--adapt-inner-optimizer",
        "SPSA",
        "--output-json",
        "/old/result.json",
        "--adapt-current-json",
        "/old/current.json",
    ]
    row = {
        "record_id": "r",
        "method_key": "snake",
        "display_regime": "weak-weak",
        "engine_key": "native_forced",
        "budget": "200",
        "source_json": str(source_json),
        "source_command_args_json": json.dumps(source_args),
        "adapt_optimizer_kind": "powell",
    }

    _source_cmd, effective_cmd, audit = runner.build_snake_source_locked_command(
        row,
        tmp_path / "out",
        inner_optimizer="ROTOSOLVE",
    )

    assert audit["status"] == "pass"
    assert _single_flag(effective_cmd, "--adapt-inner-optimizer") == "ROTOSOLVE"
    assert audit["diagnostic_inner_optimizer"] == "ROTOSOLVE"


def test_snake_source_locked_command_applies_explicit_cli_overrides(tmp_path: Path) -> None:
    source_json = tmp_path / "source_result.json"
    source_json.write_text(json.dumps({"adapt_vqe": {"ansatz_depth": 5}}), encoding="utf-8")
    source_args = [
        "/usr/bin/python3",
        "-u",
        "-m",
        "pipelines.static_adapt.adapt_pipeline",
        "--adapt-max-depth",
        "30",
        "--adapt-maxiter",
        "800",
        "--adapt-final-refit-maxiter",
        "800",
        "--phase2-rho",
        "0.25",
        "--phase2-w-shot",
        "0.08",
        "--phase-live-hysteresis-enabled",
        "--phase1-prune-collapse-ratio",
        "0.2",
        "--output-json",
        "/old/result.json",
        "--adapt-current-json",
        "/old/current.json",
    ]
    row = {
        "record_id": "r",
        "method_key": "snake",
        "display_regime": "weak-weak",
        "engine_key": "native_forced",
        "budget": "200",
        "source_json": str(source_json),
        "source_command_args_json": json.dumps(source_args),
        "snake_cli_overrides_json": json.dumps(
            {
                "set_flags": {
                    "--phase2-rho": "0.5",
                    "--phase2-w-shot": "0.05",
                    "--phase1-lambda-theta": "0.001",
                },
                "remove_bool_flags": ["--phase-live-hysteresis-enabled"],
                "remove_value_flags": ["--phase1-prune-collapse-ratio"],
                "enable_flags": ["--phase-live-hysteresis-disabled"],
            }
        ),
    }

    _source_cmd, effective_cmd, audit = runner.build_snake_source_locked_command(row, tmp_path / "out")

    assert audit["status"] == "pass"
    assert _single_flag(effective_cmd, "--phase2-rho") == "0.5"
    assert _single_flag(effective_cmd, "--phase2-w-shot") == "0.05"
    assert _single_flag(effective_cmd, "--phase1-lambda-theta") == "0.001"
    assert "--phase-live-hysteresis-enabled" not in effective_cmd
    assert "--phase-live-hysteresis-disabled" in effective_cmd
    assert "--phase1-prune-collapse-ratio" not in effective_cmd
    assert "--phase2-rho" in audit["allowed_flag_changes"]
    assert audit["diagnostic_snake_cli_overrides"]["set_flags"]["--phase2-rho"] == "0.5"


def test_snake_source_locked_command_preserves_resume_segment_depth_and_backend(tmp_path: Path) -> None:
    source_json = tmp_path / "source_result.json"
    source_json.write_text(json.dumps({"adapt_vqe": {"ansatz_depth": 21}}), encoding="utf-8")
    source_args = [
        "/usr/bin/python3",
        "-u",
        "-m",
        "pipelines.static_adapt.adapt_pipeline",
        "--adapt-max-depth",
        "30",
        "--adapt-maxiter",
        "800",
        "--adapt-final-refit-maxiter",
        "800",
        "--phase3-backend-cost-mode",
        "marrakesh_graph_span_v1",
        "--adapt-resume-scaffold-json",
        f"{runner.LOCAL_REPO_PREFIX}/raw_outputs/resume/current.json",
        "--adapt-resume-mode",
        "scaffold_v1",
        "--adapt-segment-id",
        "u8_ss_resume_from_k24_20260616",
        "--adapt-segment-target-depth",
        "30",
        "--adapt-segment-max-new-admissions",
        "8",
        "--adapt-resume-compile-smoke",
        "required",
        "--phase2-enable-batching",
        "--phase3-enable-batching",
        "--output-json",
        "/old/result.json",
        "--adapt-current-json",
        "/old/current.json",
    ]
    row = {
        "record_id": "r",
        "method_key": "snake",
        "display_regime": "strong-strong",
        "engine_key": "native_forced",
        "budget": "200",
        "source_json": str(source_json),
        "source_command_args_json": json.dumps(source_args),
        "spsa_refit_engine": runner.NATIVE_ENGINE,
    }

    source_cmd, effective_cmd, audit = runner.build_snake_source_locked_command(row, tmp_path / "out")

    assert source_cmd[0] == sys.executable
    assert audit["status"] == "pass"
    assert _single_flag(effective_cmd, "--adapt-maxiter") == "200"
    assert _single_flag(effective_cmd, "--adapt-final-refit-maxiter") == "200"
    assert _single_flag(effective_cmd, "--adapt-max-depth") == "30"
    assert _single_flag(effective_cmd, "--phase3-backend-cost-mode") == "marrakesh_graph_span_v1"
    assert _single_flag(effective_cmd, "--adapt-resume-mode") == "scaffold_v1"
    assert _single_flag(effective_cmd, "--adapt-segment-target-depth") == "30"
    assert _single_flag(effective_cmd, "--adapt-segment-max-new-admissions") == "8"
    assert _single_flag(effective_cmd, "--adapt-resume-compile-smoke") == "required"
    assert _single_flag(effective_cmd, "--adapt-resume-scaffold-json") == str(
        runner.ROOT / "raw_outputs/resume/current.json"
    )


def test_snake_source_locked_command_applies_global_child_pool_override(tmp_path: Path) -> None:
    source_json = tmp_path / "source_result.json"
    source_json.write_text(json.dumps({"adapt_vqe": {"ansatz_depth": 5}}), encoding="utf-8")
    source_args = [
        "/usr/bin/python3",
        "-u",
        "-m",
        "pipelines.static_adapt.adapt_pipeline",
        "--adapt-max-depth",
        "30",
        "--adapt-maxiter",
        "800",
        "--adapt-final-refit-maxiter",
        "800",
        "--phase3-runtime-split-mode",
        "shortlist_pauli_children_v1",
        "--allow-archival-phase3-runtime-split",
        "--output-json",
        "/old/result.json",
        "--adapt-current-json",
        "/old/current.json",
    ]
    row = {
        "record_id": "r",
        "method_key": "snake",
        "display_regime": "weak-weak",
        "engine_key": "native_forced",
        "budget": "200",
        "source_json": str(source_json),
        "source_command_args_json": json.dumps(source_args),
        "snake_phase3_runtime_split_mode": "off",
        "snake_adapt_child_pool_expansion_mode": "global_pauli_child_sets_v1",
        "snake_adapt_child_pool_expansion_symmetry_policy": "off",
        "snake_adapt_child_pool_expansion_max_subset_size": "3",
    }

    _source_cmd, effective_cmd, audit = runner.build_snake_source_locked_command(row, tmp_path / "out")

    assert audit["status"] == "pass"
    assert _single_flag(effective_cmd, "--phase3-runtime-split-mode") == "off"
    assert "--allow-archival-phase3-runtime-split" not in effective_cmd
    assert _single_flag(effective_cmd, "--adapt-child-pool-expansion-mode") == "global_pauli_child_sets_v1"
    assert _single_flag(effective_cmd, "--adapt-child-pool-expansion-symmetry-policy") == "off"
    assert _single_flag(effective_cmd, "--adapt-child-pool-expansion-max-subset-size") == "3"
    assert audit["diagnostic_snake_adapt_child_pool_expansion_mode"] == "global_pauli_child_sets_v1"


def test_snake_source_locked_command_applies_shared_pauli_pool_override(tmp_path: Path) -> None:
    source_json = tmp_path / "source_result.json"
    source_json.write_text(json.dumps({"adapt_vqe": {"ansatz_depth": 5}}), encoding="utf-8")
    source_args = [
        "/usr/bin/python3",
        "-u",
        "-m",
        "pipelines.static_adapt.adapt_pipeline",
        "--adapt-max-depth",
        "30",
        "--adapt-maxiter",
        "800",
        "--adapt-final-refit-maxiter",
        "800",
        "--phase3-runtime-split-mode",
        "shortlist_pauli_children_v1",
        "--allow-archival-phase3-runtime-split",
        "--adapt-child-pool-expansion-mode",
        "global_pauli_child_sets_v1",
        "--output-json",
        "/old/result.json",
        "--adapt-current-json",
        "/old/current.json",
    ]
    row = {
        "record_id": "r",
        "method_key": "snake",
        "display_regime": "weak-weak",
        "engine_key": "native_forced",
        "budget": "200",
        "source_json": str(source_json),
        "source_command_args_json": json.dumps(source_args),
        "shared_pauli_pool_mode": "shared_pauli_child_sets_v1",
        "shared_pauli_pool_symmetry_policy": "hard_guard",
        "shared_pauli_pool_max_subset_size": "3",
    }

    _source_cmd, effective_cmd, audit = runner.build_snake_source_locked_command(row, tmp_path / "out")

    assert audit["status"] == "pass"
    assert _single_flag(effective_cmd, "--shared-pauli-pool-mode") == "shared_pauli_child_sets_v1"
    assert _single_flag(effective_cmd, "--shared-pauli-pool-symmetry-policy") == "hard_guard"
    assert _single_flag(effective_cmd, "--shared-pauli-pool-max-subset-size") == "3"
    assert _single_flag(effective_cmd, "--phase3-runtime-split-mode") == "off"
    assert "--allow-archival-phase3-runtime-split" not in effective_cmd
    assert _single_flag(effective_cmd, "--adapt-child-pool-expansion-mode") == "off"
    assert audit["diagnostic_shared_pauli_pool_mode"] == "shared_pauli_child_sets_v1"


def test_snake_source_locked_command_applies_shared_pauli_pool_no_guard_override(tmp_path: Path) -> None:
    source_json = tmp_path / "source_result.json"
    source_json.write_text(json.dumps({"adapt_vqe": {"ansatz_depth": 5}}), encoding="utf-8")
    source_args = [
        "/usr/bin/python3",
        "-u",
        "-m",
        "pipelines.static_adapt.adapt_pipeline",
        "--adapt-max-depth",
        "30",
        "--adapt-maxiter",
        "800",
        "--adapt-final-refit-maxiter",
        "800",
        "--phase3-runtime-split-mode",
        "shortlist_pauli_children_v1",
        "--allow-archival-phase3-runtime-split",
        "--adapt-child-pool-expansion-mode",
        "global_pauli_child_sets_v1",
        "--output-json",
        "/old/result.json",
        "--adapt-current-json",
        "/old/current.json",
    ]
    row = {
        "record_id": "r",
        "method_key": "snake",
        "display_regime": "weak-weak",
        "engine_key": "native_forced",
        "budget": "200",
        "source_json": str(source_json),
        "source_command_args_json": json.dumps(source_args),
        "shared_pauli_pool_mode": "shared_pauli_child_sets_v1",
        "shared_pauli_pool_symmetry_policy": "off",
        "shared_pauli_pool_max_subset_size": "1",
    }

    _source_cmd, effective_cmd, audit = runner.build_snake_source_locked_command(row, tmp_path / "out")

    assert audit["status"] == "pass"
    assert _single_flag(effective_cmd, "--shared-pauli-pool-mode") == "shared_pauli_child_sets_v1"
    assert _single_flag(effective_cmd, "--shared-pauli-pool-symmetry-policy") == "off"
    assert _single_flag(effective_cmd, "--shared-pauli-pool-max-subset-size") == "1"
    assert _single_flag(effective_cmd, "--phase3-runtime-split-mode") == "off"
    assert "--allow-archival-phase3-runtime-split" not in effective_cmd
    assert _single_flag(effective_cmd, "--adapt-child-pool-expansion-mode") == "off"
    assert audit["diagnostic_shared_pauli_pool_mode"] == "shared_pauli_child_sets_v1"
    assert audit["diagnostic_shared_pauli_pool_symmetry_policy"] == "off"
    assert audit["diagnostic_shared_pauli_pool_max_subset_size"] == "1"
    assert "--shared-pauli-pool-symmetry-policy" in audit["allowed_flag_changes"]
    assert "--adapt-child-pool-expansion-mode" in audit["allowed_flag_changes"]


def test_snake_source_locked_command_applies_minus_hva_class_filter(tmp_path: Path) -> None:
    source_json = tmp_path / "source_result.json"
    source_json.write_text(json.dumps({"adapt_vqe": {"ansatz_depth": 5}}), encoding="utf-8")
    source_args = [
        "/usr/bin/python3",
        "-u",
        "-m",
        "pipelines.static_adapt.adapt_pipeline",
        "--adapt-max-depth",
        "30",
        "--adapt-maxiter",
        "800",
        "--adapt-final-refit-maxiter",
        "800",
        "--adapt-pool",
        "full_meta",
        "--output-json",
        "/old/result.json",
        "--adapt-current-json",
        "/old/current.json",
    ]
    class_filter = "agent_guidance/static-adapt/hh_full_meta_minus_hva_class_filter.json"
    row = {
        "record_id": "r",
        "method_key": "snake",
        "display_regime": "weak-weak",
        "engine_key": "native_forced",
        "budget": "200",
        "source_json": str(source_json),
        "source_command_args_json": json.dumps(source_args),
        "adapt_pool_class_filter_json": class_filter,
    }

    _source_cmd, effective_cmd, audit = runner.build_snake_source_locked_command(row, tmp_path / "out")

    assert audit["status"] == "pass"
    assert _single_flag(effective_cmd, "--adapt-pool") == "full_meta"
    assert _single_flag(effective_cmd, "--adapt-pool-class-filter-json") == class_filter
    assert "--adapt-pool-class-filter-json" in audit["allowed_flag_changes"]
    assert audit["diagnostic_adapt_pool_class_filter_json"] == class_filter


def test_snake_source_locked_command_removes_class_filter_when_off(tmp_path: Path) -> None:
    source_json = tmp_path / "source_result.json"
    source_json.write_text(json.dumps({"adapt_vqe": {"ansatz_depth": 5}}), encoding="utf-8")
    class_filter = "agent_guidance/static-adapt/hh_full_meta_minus_hva_class_filter.json"
    source_args = [
        "/usr/bin/python3",
        "-u",
        "-m",
        "pipelines.static_adapt.adapt_pipeline",
        "--adapt-max-depth",
        "30",
        "--adapt-maxiter",
        "800",
        "--adapt-final-refit-maxiter",
        "800",
        "--adapt-pool",
        "full_meta",
        "--adapt-pool-class-filter-json",
        class_filter,
        "--output-json",
        "/old/result.json",
        "--adapt-current-json",
        "/old/current.json",
    ]
    row = {
        "record_id": "r",
        "method_key": "snake",
        "display_regime": "weak-weak",
        "engine_key": "native_forced",
        "budget": "200",
        "source_json": str(source_json),
        "source_command_args_json": json.dumps(source_args),
        "adapt_pool_class_filter_json": "off",
    }

    _source_cmd, effective_cmd, audit = runner.build_snake_source_locked_command(row, tmp_path / "out")

    assert audit["status"] == "pass"
    assert "--adapt-pool-class-filter-json" not in effective_cmd
    assert "--adapt-pool-class-filter-json" in audit["allowed_flag_changes"]
    assert audit["diagnostic_adapt_pool_class_filter_json"] == "off"


def test_append_geo_env_applies_shared_pauli_pool_overlay(tmp_path: Path) -> None:
    row = {
        "method_key": "append",
        "suite_profile": "paper_i_main_tables_spsa",
        "adapt_optimizer_kind": "powell",
        "max_depth": "30",
        "budget": "200",
        "same_cutoff_exact_gs_energy": "-1.0",
        "exact_reference_energy": "-1.0",
        "exact_reference_n_ph_max": "5",
        "shared_pauli_pool_mode": "shared_pauli_child_sets_v1",
        "shared_pauli_pool_symmetry_policy": "hard_guard",
        "shared_pauli_pool_max_subset_size": "3",
        "pool_contract": "full_meta_unfiltered",
        "adapt_pool_class_filter_json": "off",
    }

    env = runner.append_geo_env(row, tmp_path / "out")

    assert env["GENERIC_STATIC_TABLE_SHARED_PAULI_POOL_MODE"] == "shared_pauli_child_sets_v1"
    assert env["GENERIC_STATIC_TABLE_SHARED_PAULI_POOL_SYMMETRY_POLICY"] == "hard_guard"
    assert env["GENERIC_STATIC_TABLE_SHARED_PAULI_POOL_MAX_SUBSET_SIZE"] == "3"
    assert env["GENERIC_STATIC_TABLE_HH_ADAPTIVE_POOL_PROFILE"] == "full_meta_unfiltered"
    assert env["GENERIC_STATIC_TABLE_HH_FULL_META_CLASS_FILTER_JSON"] == "off"
    assert "GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_MODE" not in env


def test_append_geo_env_applies_shared_pauli_pool_no_guard_overlay(tmp_path: Path) -> None:
    row = {
        "method_key": "geo",
        "suite_profile": "paper_i_main_tables_spsa",
        "adapt_optimizer_kind": "powell",
        "max_depth": "30",
        "budget": "200",
        "same_cutoff_exact_gs_energy": "-1.0",
        "exact_reference_energy": "-1.0",
        "exact_reference_n_ph_max": "5",
        "shared_pauli_pool_mode": "shared_pauli_child_sets_v1",
        "shared_pauli_pool_symmetry_policy": "off",
        "shared_pauli_pool_max_subset_size": "1",
        "pool_contract": "full_meta_unfiltered",
        "adapt_pool_class_filter_json": "off",
    }

    env = runner.append_geo_env(row, tmp_path / "out")

    assert env["GENERIC_STATIC_TABLE_SHARED_PAULI_POOL_MODE"] == "shared_pauli_child_sets_v1"
    assert env["GENERIC_STATIC_TABLE_SHARED_PAULI_POOL_SYMMETRY_POLICY"] == "off"
    assert env["GENERIC_STATIC_TABLE_SHARED_PAULI_POOL_MAX_SUBSET_SIZE"] == "1"
    assert env["GENERIC_STATIC_TABLE_HH_ADAPTIVE_POOL_PROFILE"] == "full_meta_unfiltered"
    assert env["GENERIC_STATIC_TABLE_HH_FULL_META_CLASS_FILTER_JSON"] == "off"
    assert "GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_MODE" not in env


def test_snake_source_locked_command_applies_phase3_child_set_hard_guard(tmp_path: Path) -> None:
    source_json = tmp_path / "source_result.json"
    source_json.write_text(json.dumps({"adapt_vqe": {"ansatz_depth": 5}}), encoding="utf-8")
    source_args = [
        "/usr/bin/python3",
        "-u",
        "-m",
        "pipelines.static_adapt.adapt_pipeline",
        "--adapt-max-depth",
        "30",
        "--adapt-maxiter",
        "800",
        "--adapt-final-refit-maxiter",
        "800",
        "--phase3-runtime-split-mode",
        "shortlist_pauli_children_v1",
        "--allow-archival-phase3-runtime-split",
        "--output-json",
        "/old/result.json",
        "--adapt-current-json",
        "/old/current.json",
    ]
    row = {
        "record_id": "r",
        "method_key": "snake",
        "display_regime": "weak-weak",
        "engine_key": "native_forced",
        "budget": "200",
        "source_json": str(source_json),
        "source_command_args_json": json.dumps(source_args),
        "snake_phase3_runtime_split_mode": "shortlist_pauli_children_v1",
        "snake_phase3_runtime_split_selection_mode": "archival_child_set_forward_v1",
        "snake_phase3_runtime_split_child_set_symmetry_policy": "hard_guard",
        "snake_phase3_runtime_split_max_subset_size": "7",
    }

    _source_cmd, effective_cmd, audit = runner.build_snake_source_locked_command(row, tmp_path / "out")

    assert audit["status"] == "pass"
    assert _single_flag(effective_cmd, "--phase3-runtime-split-mode") == "shortlist_pauli_children_v1"
    assert "--allow-archival-phase3-runtime-split" in effective_cmd
    assert _single_flag(effective_cmd, "--phase3-runtime-split-selection-mode") == "archival_child_set_forward_v1"
    assert _single_flag(effective_cmd, "--phase3-runtime-split-child-set-symmetry-policy") == "hard_guard"
    assert _single_flag(effective_cmd, "--phase3-runtime-split-max-subset-size") == "7"
    assert audit["diagnostic_snake_phase3_runtime_split_selection_mode"] == "archival_child_set_forward_v1"
    assert audit["diagnostic_snake_phase3_runtime_split_child_set_symmetry_policy"] == "hard_guard"
    assert audit["diagnostic_snake_phase3_runtime_split_max_subset_size"] == "7"


def test_snake_source_locked_command_applies_phase3_child_set_off(tmp_path: Path) -> None:
    source_json = tmp_path / "source_result.json"
    source_json.write_text(json.dumps({"adapt_vqe": {"ansatz_depth": 5}}), encoding="utf-8")
    source_args = [
        "/usr/bin/python3",
        "-u",
        "-m",
        "pipelines.static_adapt.adapt_pipeline",
        "--adapt-max-depth",
        "30",
        "--adapt-maxiter",
        "800",
        "--adapt-final-refit-maxiter",
        "800",
        "--phase3-runtime-split-mode",
        "shortlist_pauli_children_v1",
        "--allow-archival-phase3-runtime-split",
        "--output-json",
        "/old/result.json",
        "--adapt-current-json",
        "/old/current.json",
    ]
    row = {
        "record_id": "r",
        "method_key": "snake",
        "display_regime": "weak-weak",
        "engine_key": "native_forced",
        "budget": "200",
        "source_json": str(source_json),
        "source_command_args_json": json.dumps(source_args),
        "snake_phase3_runtime_split_mode": "shortlist_pauli_children_v1",
        "snake_phase3_runtime_split_selection_mode": "archival_child_set_forward_v1",
        "snake_phase3_runtime_split_child_set_symmetry_policy": "off",
        "snake_phase3_runtime_split_max_subset_size": "1",
    }

    _source_cmd, effective_cmd, audit = runner.build_snake_source_locked_command(row, tmp_path / "out")

    assert audit["status"] == "pass"
    assert _single_flag(effective_cmd, "--phase3-runtime-split-child-set-symmetry-policy") == "off"
    assert audit["diagnostic_snake_phase3_runtime_split_child_set_symmetry_policy"] == "off"


def test_snake_algorithmic_work_sidecar_is_reconstructable(tmp_path: Path) -> None:
    result_path = tmp_path / "json" / "result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(_minimal_snake_payload()), encoding="utf-8")
    sidecar_path = tmp_path / "snake_algorithmic_work.json"
    row = {
        "record_id": "r",
        "display_regime": "weak-weak",
        "engine_key": "native_forced",
        "budget": "800",
        "result_json_rel": str(result_path),
        "snake_algorithmic_work_rel": str(sidecar_path),
    }

    sidecar = runner.write_snake_algorithmic_work_sidecar(row, tmp_path)

    assert sidecar["S_alg_status"] == "ok"
    assert sidecar["S_alg"] == 30.0
    assert sidecar["component_counts"] == {
        "N_H_outer_eval": 15.0,
        "N_grad_probe": 7.0,
        "N_metric_probe": 3.0,
        "N_H_refit_eval": 5.0,
    }
    assert sidecar_path.exists()


def test_snake_fair_sidecar_blocks_legacy_group_common_but_keeps_actual(tmp_path: Path) -> None:
    result_path = tmp_path / "json" / "result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(_minimal_snake_payload()), encoding="utf-8")
    row = {
        "record_id": "r",
        "display_regime": "weak-weak",
        "engine_key": "native_forced",
        "budget": "800",
        "result_json_rel": str(result_path),
    }

    sidecar = runner.write_snake_fair_shot_work_sidecar(row, tmp_path)

    assert sidecar["S_actual"] == 30
    assert sidecar["S_actual_status"] == "ok"
    assert sidecar["S_common_exposure"] is None
    assert sidecar["S_common_exposure_status"] == "missing_common_exposure_ledger"
    assert sidecar["S_fair"] is None
    assert sidecar["S_fair_status"] == "missing_common_exposure_ledger"
    assert sidecar["fair_work_currency"] == "expanded_common_candidate_probe_event_count_v1"
    assert sidecar["operator_probe_charge_basis"] == "logical_estimator_request_pre_grouping_v1"
    assert (tmp_path / "snake_fair_shot_work.json").exists()


def test_report_uses_snake_sidecar_s_alg_not_legacy_s_norm(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(_minimal_snake_payload(s_norm=999.0)), encoding="utf-8")
    sidecar_path = tmp_path / "snake_algorithmic_work.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "schema": "paper_i_hh_spsa_budget_ladder_snake_algorithmic_work_sidecar_v1",
                "S_alg": 32.0,
                "S_alg_status": "ok",
                "algorithmic_measurement_work": {
                    "schema": "algorithmic_measurement_work_v1",
                    "status": "ok",
                    "S_alg": 32.0,
                    "components": {},
                    "component_sources": {},
                },
            }
        ),
        encoding="utf-8",
    )
    records_tsv = tmp_path / "records.tsv"
    fieldnames = [
        "record_id",
        "method_key",
        "method_label",
        "engine_key",
        "engine_label",
        "budget",
        "display_regime",
        "result_json_rel",
        "snake_algorithmic_work_rel",
        "cell_manifest_rel",
        "record_output_dir",
        "same_cutoff_exact_gs_energy",
    ]
    with records_tsv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "record_id": "r",
                "method_key": "snake",
                "method_label": "SNAKE",
                "engine_key": "native_forced",
                "engine_label": "native forced full budget",
                "budget": "800",
                "display_regime": "weak-weak",
                "result_json_rel": str(result_path),
                "snake_algorithmic_work_rel": str(sidecar_path),
                "cell_manifest_rel": str(tmp_path / "missing_manifest.json"),
                "record_output_dir": str(tmp_path),
                "same_cutoff_exact_gs_energy": "-1.001",
            }
        )

    loaded = report.load_records(records_tsv)

    assert len(loaded) == 1
    assert loaded[0].s_alg == 32.0
    assert loaded[0].s_alg_status == "ok"
