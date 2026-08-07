from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
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


generator = _load_script_module(
    "paper_i_hh_recovery_candidate_run_stock_records",
    "chtc/phase3_optuna/generate_paper_i_hh_recovery_candidate_run_stock_records.py",
)
preflight_submit = _load_script_module(
    "phase3_preflight_submit_for_recovery_candidate",
    "chtc/phase3_optuna/preflight_submit.py",
)
runner = _load_script_module(
    "paper_i_hh_spsa_budget_ladder_cell_for_recovery_candidate",
    "chtc/phase3_optuna/run_paper_i_hh_spsa_budget_ladder_cell.py",
)


def _arg_value(args: list[str], flag: str) -> str:
    assert flag in args
    return args[args.index(flag) + 1]


def test_recovery_candidate_powell_wave0_nobatch_anchor_records() -> None:
    rows = generator.build_records(
        "paper_i_hh_recovery_candidate_unit_powell_wave0",
        stage_name="powell_nobatch_anchor",
        wave_index=0,
    )

    assert len(rows) == 2
    assert {row["display_regime"] for row in rows} == {"weak-weak", "weak-strong"}
    assert {row["method_key"] for row in rows} == {"snake"}
    assert {row["run_class"] for row in rows} == {"candidate"}
    assert {row["optimizer"] for row in rows} == {"POWELL"}
    assert {row["adapt_optimizer_kind"] for row in rows} == {"powell"}
    assert {row["budget"] for row in rows} == {"200"}
    assert {row["max_depth"] for row in rows} == {"30"}
    assert {row["pool_contract"] for row in rows} == {"full_meta_unfiltered"}
    assert {row["adapt_pool_class_filter_json"] for row in rows} == {"off"}
    assert {row["snake_phase3_runtime_split_selection_mode"] for row in rows} == {
        "archival_child_set_forward_v1"
    }
    assert {row["snake_phase3_runtime_split_child_set_symmetry_policy"] for row in rows} == {
        "hard_guard"
    }
    assert {row["child_subset_size"] for row in rows} == {"3"}
    assert {row["snake_phase3_runtime_split_max_subset_size"] for row in rows} == {"3"}
    assert {row["route_variant"] for row in rows} == {
        "nobatch_anchor_cap3_metricprune_beam0p005"
    }
    assert {row["ordered_batch_beam_enabled"] for row in rows} == {"false"}
    assert {row["provenance_layer"] for row in rows} == {"visible_row"}
    assert all(row["latex_report_stem"] == "paper_i_hh_recovery_candidate_run_stock_20260705" for row in rows)
    assert all("hh_full_meta_minus_hva_class_filter.json" not in json.dumps(row) for row in rows)

    for row in rows:
        overrides = json.loads(row["snake_cli_overrides_json"])
        assert overrides["set_flags"]["--phase3-runtime-split-max-subset-size"] == "3"
        assert overrides["set_flags"]["--phase1-prune-schur-nomination-route"] == "metric_regularized_v1"
        assert overrides["set_flags"]["--adapt-beam-lambda"] == "0.005"
        assert "--phase2-no-batching" in overrides["enable_flags"]
        assert "--phase3-no-batching" in overrides["enable_flags"]
        assert "--phase2-enable-batching" in overrides["remove_bool_flags"]
        assert "--phase3-enable-batching" in overrides["remove_bool_flags"]
        assert "--phase3-source-lock-preferred-sequence" in overrides["remove_value_flags"]
        assert preflight_submit._is_hh_fullmeta_phase3_singleton(row)
        assert preflight_submit._hh_fullmeta_phase3_singleton_contract_blockers(row) == []


def test_recovery_candidate_effective_command_is_source_locked_nobatch(tmp_path: Path) -> None:
    row = generator.build_records(
        "paper_i_hh_recovery_candidate_unit_command",
        stage_name="powell_nobatch_anchor",
        wave_index=0,
    )[0]

    source_cmd, effective_cmd, audit = runner.build_snake_source_locked_command(
        row,
        tmp_path / "record",
    )

    assert source_cmd
    assert audit["status"] == "pass"
    assert audit["non_allowed_flag_changes"] == []
    assert _arg_value(effective_cmd, "--adapt-inner-optimizer") == "POWELL"
    assert _arg_value(effective_cmd, "--adapt-maxiter") == "200"
    assert _arg_value(effective_cmd, "--adapt-final-refit-maxiter") == "200"
    assert _arg_value(effective_cmd, "--adapt-max-depth") == "30"
    assert _arg_value(effective_cmd, "--phase3-runtime-split-max-subset-size") == "3"
    assert _arg_value(effective_cmd, "--phase1-prune-schur-nomination-route") == "metric_regularized_v1"
    assert _arg_value(effective_cmd, "--adapt-beam-lambda") == "0.005"
    assert "--phase2-no-batching" in effective_cmd
    assert "--phase3-no-batching" in effective_cmd
    assert "--phase2-enable-batching" not in effective_cmd
    assert "--phase3-enable-batching" not in effective_cmd
    assert "--phase3-source-lock-preferred-sequence" not in effective_cmd
    assert "--adapt-pool-class-filter-json" not in effective_cmd


def test_recovery_candidate_batch_stage_is_gated_and_explicit() -> None:
    rows = generator.build_records(
        "paper_i_hh_recovery_candidate_unit_powell_batch_wave0",
        stage_name="powell_batch_gated",
        wave_index=0,
        include_batch_variants=("greedy_batch_cap3", "combinatorial_batch_cap3"),
    )

    assert len(rows) == 4
    assert {row["display_regime"] for row in rows} == {"weak-weak", "weak-strong"}
    assert {row["ordered_batch_beam_enabled"] for row in rows} == {"true"}
    assert {row["anchor_gate_status"] for row in rows} == {"requires_matching_nobatch_anchor_pass"}
    assert {row["batch_variant_gate"] for row in rows} == {"gated_after_anchor"}
    assert {row["route_variant"] for row in rows} == {
        "greedy_batch_cap3",
        "combinatorial_batch_cap3",
    }
    assert {row["phase2_batch_selection_mode"] for row in rows} == {
        "greedy_reduced_plane",
        "combinatorial_reduced_plane",
    }
    assert {row["phase2_batch_target_size"] for row in rows} == {"3"}
    assert {row["phase2_batch_size_cap"] for row in rows} == {"3"}
    assert {row["adapt_beam_lambda"] for row in rows} == {"0.005"}

    for row in rows:
        overrides = json.loads(row["snake_cli_overrides_json"])
        assert "--phase2-enable-batching" in overrides["enable_flags"]
        assert "--phase3-enable-batching" in overrides["enable_flags"]
        assert overrides["set_flags"]["--static-route-id"] == "unspecified"
        assert overrides["set_flags"]["--phase2-batch-target-size"] == "3"
        assert overrides["set_flags"]["--phase2-batch-size-cap"] == "3"
        assert overrides["set_flags"]["--adapt-beam-live-branches"] == "3"
        assert overrides["set_flags"]["--adapt-beam-children-per-parent"] == "3"
        assert preflight_submit._hh_fullmeta_phase3_singleton_contract_blockers(row) == []


def test_recovery_candidate_write_records_keeps_provenance_fields() -> None:
    batch_id = "paper_i_hh_recovery_candidate_unit_write"
    rows = generator.build_records(batch_id, stage_name="powell_nobatch_anchor", wave_index=0)
    manifest = generator.write_records(
        batch_id,
        rows,
        request_cpus=1,
        request_memory_mb=32768,
        request_disk_mb=61440,
        max_runtime_s=172800,
        stage_name="powell_nobatch_anchor",
        wave_index=0,
    )

    assert manifest["schema"] == "paper_i_hh_recovery_candidate_run_stock_manifest_v1"
    assert manifest["record_count"] == 2
    assert manifest["run_stock"]["stage"] == "powell_nobatch_anchor"
    assert manifest["run_stock"]["wave_index"] == 0
    assert manifest["run_stock"]["report_update_policy"] == (
        "every_completed_result_batch_updates_latex_pdf_json_csv_md_sidecars"
    )
    assert manifest["source_contract"]["provenance_layer"] == "visible_row"
    assert manifest["source_contract"]["snake_child_policy"]["phase3_runtime_split_max_subset_size"] == 3
    assert (
        manifest["source_contract"]["matrix_policies"][0]["matrix_role"]
        == "visible-row recovery/candidate cap-3 route"
    )

    input_dir = ROOT / "chtc" / "phase3_optuna" / "input" / batch_id
    with (input_dir / "paper_i_hh_spsa_budget_ladder_records.tsv").open(newline="", encoding="utf-8") as fh:
        read_back = list(csv.DictReader(fh, delimiter="\t"))
    assert len(read_back) == 2
    assert "provenance_layer" in read_back[0]
    assert "settings_changed_json" in read_back[0]
    assert "work_semantics_expected_json" in read_back[0]
    assert "latex_report_stem" in read_back[0]
    assert read_back[0]["provenance_layer"] == "visible_row"
    assert json.loads(read_back[0]["work_semantics_expected_json"])["S_alg_work_scope"] == (
        "winner_lineage_terminal"
    )

    submit_path = ROOT / "chtc" / "phase3_optuna" / f"submit_{batch_id}.sub"
    submit_contract = preflight_submit.run_task.parse_submit_contract(submit_path)
    assert submit_contract["queue_record_id_file"] == (
        f"chtc/phase3_optuna/input/{batch_id}/paper_i_hh_spsa_budget_ladder_record_queue.tsv"
    )

    preflight = preflight_submit.build_preflight_bundle(
        submit_path=submit_path,
        records_path=input_dir / "paper_i_hh_spsa_budget_ladder_records.tsv",
    )
    assert preflight["ok"], preflight["blocking_reasons"]
    assert preflight["record_count"] == 2
