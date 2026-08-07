from __future__ import annotations

import csv
import importlib.util
import json
import sys
from collections import Counter
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
    "paper_i_hh_weak_weak_snake_mechanism_ablation_records",
    "chtc/phase3_optuna/generate_paper_i_hh_weak_weak_snake_mechanism_ablation_records.py",
)
preflight_submit = _load_script_module(
    "phase3_preflight_submit_for_weak_weak_mechanism_ablation",
    "chtc/phase3_optuna/preflight_submit.py",
)
runner = _load_script_module(
    "paper_i_hh_spsa_budget_ladder_cell_for_weak_weak_mechanism_ablation",
    "chtc/phase3_optuna/run_paper_i_hh_spsa_budget_ladder_cell.py",
)


def _arg_value(args: list[str], flag: str) -> str:
    assert flag in args
    idx = args.index(flag)
    assert idx + 1 < len(args)
    return args[idx + 1]


def _rows() -> list[dict[str, str]]:
    return generator.build_records("paper_i_hh_weak_weak_snake_mechanism_ablation_unit")


def test_mechanism_ablation_matrix_is_doubled_and_source_locked() -> None:
    rows = _rows()
    runnable = [row for row in rows if row["runnable"] == "true"]

    assert len(rows) == 28
    assert len(runnable) == 20
    assert Counter(row["source_anchor_family"] for row in rows) == {
        "batch_cap3_combinatorial": 14,
        "physical_operator_lane": 14,
    }
    assert Counter(row["source_anchor_family"] for row in runnable) == {
        "batch_cap3_combinatorial": 9,
        "physical_operator_lane": 11,
    }
    assert {row["display_regime"] for row in rows} == {"weak-weak"}
    assert {row["method_key"] for row in rows} == {"snake"}
    assert {row["optimizer"] for row in rows} == {"POWELL"}
    assert {row["adapt_optimizer_kind"] for row in rows} == {"powell"}
    assert {row["budget"] for row in rows} == {"200"}
    assert {row["max_depth"] for row in rows} == {"30"}
    assert {row["pool_contract"] for row in rows} == {"full_meta_unfiltered"}
    assert {row["adapt_pool_class_filter_json"] for row in rows} == {"off"}
    assert {row["adapt_beam_lambda"] for row in rows} == {"0.005"}
    assert {row["adapt_beam_children_per_parent"] for row in rows} == {"2"}
    assert all("hh_full_meta_minus_hva_class_filter.json" not in json.dumps(row) for row in rows)


def test_mechanism_ablation_keeps_child_cap_separate_from_batch_cap() -> None:
    rows = _rows()
    native_rows = [row for row in rows if row["child_policy"] == "native_phase3_singleton"]
    batch_rows = [
        row
        for row in rows
        if row["runnable"] == "true" and row["hh_mechanism_ablation_variant"] in {"greedy_cap3", "combinatorial_cap3"}
    ]

    assert {row["child_subset_size"] for row in native_rows} == {"1"}
    assert {row["snake_phase3_runtime_split_max_subset_size"] for row in native_rows} == {"1"}
    assert len(batch_rows) == 2
    assert {row["source_anchor_family"] for row in batch_rows} == {"physical_operator_lane"}
    assert {row["phase2_batch_target_size"] for row in batch_rows} == {"3"}
    assert {row["phase2_batch_size_cap"] for row in batch_rows} == {"3"}
    assert {row["phase2_batch_selection_mode"] for row in batch_rows} == {
        "greedy_reduced_plane",
        "combinatorial_reduced_plane",
    }
    for row in batch_rows:
        overrides = json.loads(row["snake_cli_overrides_json"])
        assert overrides["set_flags"]["--phase3-runtime-split-max-subset-size"] == "1"
        assert overrides["set_flags"]["--phase2-batch-target-size"] == "3"
        assert overrides["set_flags"]["--phase2-batch-size-cap"] == "3"
        assert overrides["set_flags"]["--adapt-beam-children-per-parent"] == "2"
        assert "--phase2-enable-batching" in overrides["enable_flags"]
        assert "--phase3-enable-batching" in overrides["enable_flags"]
    physical_combo = next(
        row
        for row in batch_rows
        if row["source_anchor_family"] == "physical_operator_lane"
        and row["hh_mechanism_ablation_variant"] == "combinatorial_cap3"
    )
    assert physical_combo["source_anchor_role"] == "physical_operator_lane_combinatorial_cap3_source_anchor_rebuild"
    assert physical_combo["hh_mechanism_ablation_role"] == "physical_operator_lane_source_anchor_rebuild"
    assert physical_combo["hh_mechanism_ablation_expected_status"] == "queued_source_anchor"


def test_mechanism_ablation_variant_overrides_are_fail_closed() -> None:
    rows = {row["hh_mechanism_ablation_variant"]: row for row in _rows() if row["source_anchor_family"] == "batch_cap3_combinatorial"}

    no_cost = json.loads(rows["no_cost_term"]["snake_cli_overrides_json"])["set_flags"]
    assert no_cost["--adapt-beam-lambda"] == "0.005"
    assert no_cost["--phase2-w-shot"] == "0.0"
    assert no_cost["--phase2-w-depth"] == "0.0"
    assert no_cost["--phase3-backend-w-depth"] == "0.0"
    assert no_cost["--phase3-backend-cost-mode"] == "proxy"

    phase2_only = json.loads(rows["phase2_novelty_only_no_second_order"]["snake_cli_overrides_json"])["set_flags"]
    assert phase2_only["--adapt-continuation-mode"] == "phase2_v1"
    assert phase2_only["--phase3-backend-cost-mode"] == "proxy"
    assert phase2_only["--phase2-selector-gain-mode"] == "unit_gain_v1"

    phase1_macro = rows["phase1_only_macro_pool"]
    macro_flags = json.loads(phase1_macro["snake_cli_overrides_json"])["set_flags"]
    assert phase1_macro["child_policy"] == "macro_only"
    assert phase1_macro["snake_phase3_runtime_split_mode"] == "off"
    assert macro_flags["--adapt-continuation-mode"] == "phase1_v1"

    phase1_singleton = rows["phase1_only_singleton_pool"]
    singleton_flags = json.loads(phase1_singleton["snake_cli_overrides_json"])["set_flags"]
    assert phase1_singleton["child_policy"] == "common_phase0_singleton"
    assert phase1_singleton["shared_pauli_pool_mode"] == "shared_pauli_child_sets_v1"
    assert phase1_singleton["shared_pauli_pool_max_subset_size"] == "1"
    assert singleton_flags["--adapt-continuation-mode"] == "phase1_v1"


def test_physical_anchor_rows_preserve_physical_source_lane() -> None:
    physical_rows = [row for row in _rows() if row["source_anchor_family"] == "physical_operator_lane"]
    assert physical_rows
    for row in physical_rows:
        source_args = json.loads(row["source_command_args_json"])
        assert _arg_value(source_args, "--static-lane-route") == "physical_operator_type"
        assert _arg_value(source_args, "--physical-lane-shortlist-aggressiveness") == "3"


def test_mechanism_ablation_rows_pass_dedicated_preflight_contract() -> None:
    for row in _rows():
        assert preflight_submit._is_hh_weak_weak_snake_mechanism_ablation(row)
        blockers = preflight_submit._hh_weak_weak_snake_mechanism_ablation_contract_blockers(row)
        assert blockers == [], (row["record_id"], blockers)


def test_mechanism_ablation_effective_physical_batch_command(tmp_path: Path) -> None:
    rows = _rows()
    row = next(
        row
        for row in rows
        if row["source_anchor_family"] == "physical_operator_lane"
        and row["hh_mechanism_ablation_variant"] == "combinatorial_cap3"
    )

    _source_cmd, effective_cmd, audit = runner.build_snake_source_locked_command(row, tmp_path / "record")

    assert audit["status"] == "pass", audit
    assert audit["non_allowed_flag_changes"] == []
    assert _arg_value(effective_cmd, "--adapt-inner-optimizer") == "POWELL"
    assert _arg_value(effective_cmd, "--adapt-maxiter") == "200"
    assert _arg_value(effective_cmd, "--adapt-final-refit-maxiter") == "200"
    assert _arg_value(effective_cmd, "--adapt-max-depth") == "30"
    assert _arg_value(effective_cmd, "--static-lane-route") == "physical_operator_type"
    assert _arg_value(effective_cmd, "--phase3-runtime-split-max-subset-size") == "1"
    assert _arg_value(effective_cmd, "--phase2-batch-selection-mode") == "combinatorial_reduced_plane"
    assert _arg_value(effective_cmd, "--phase2-batch-target-size") == "3"
    assert _arg_value(effective_cmd, "--phase2-batch-size-cap") == "3"
    assert _arg_value(effective_cmd, "--adapt-beam-lambda") == "0.005"
    assert _arg_value(effective_cmd, "--adapt-beam-children-per-parent") == "2"
    assert "--phase2-enable-batching" in effective_cmd
    assert "--phase3-enable-batching" in effective_cmd


def test_mechanism_ablation_write_records_and_preflight_bundle() -> None:
    batch_id = "paper_i_hh_weak_weak_snake_mechanism_ablation_unit_write"
    rows = generator.build_records(batch_id)
    manifest = generator.write_records(
        batch_id,
        rows,
        request_cpus=1,
        request_memory_mb=32768,
        request_disk_mb=61440,
        max_runtime_s=172800,
    )

    assert manifest["schema"] == "paper_i_hh_weak_weak_snake_mechanism_ablation_manifest_v1"
    assert manifest["record_count"] == 28
    assert manifest["runnable_record_count"] == 20
    assert manifest["expected_runnable_rows"] == 20
    assert any(
        item.endswith(
            "raw_outputs/paper_i_hh_powell_weak_weak_snake_batchcap3_ablation_20260707/"
            "weak_weak__snake__powell200__phase3_archival_subset1__beam0p005_metricprune__combinatorial_cap3/"
            "json/result.json"
        )
        for item in manifest["source_transfer_inputs"]
    )
    assert any(
        item.endswith("raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708/weak_weak/json/result.json")
        for item in manifest["source_transfer_inputs"]
    )

    input_dir = ROOT / "chtc" / "phase3_optuna" / "input" / batch_id
    records_tsv = input_dir / "paper_i_hh_spsa_budget_ladder_records.tsv"
    with records_tsv.open(newline="", encoding="utf-8") as fh:
        read_back = list(csv.DictReader(fh, delimiter="\t"))
    assert len(read_back) == 28
    assert "source_anchor_family" in read_back[0]
    assert "hh_mechanism_ablation_variant" in read_back[0]

    submit_path = ROOT / "chtc" / "phase3_optuna" / f"submit_{batch_id}.sub"
    submit_text = submit_path.read_text(encoding="utf-8")
    assert "raw_outputs/paper_i_hh_powell_weak_weak_snake_batchcap3_ablation_20260707" in submit_text
    assert "raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708/weak_weak/json/result.json" in submit_text
    preflight = preflight_submit.build_preflight_bundle(
        submit_path=submit_path,
        records_path=records_tsv,
        record_id_file=input_dir / "paper_i_hh_spsa_budget_ladder_record_ids.txt",
    )
    assert preflight["ok"], preflight["blocking_reasons"]
    assert preflight["record_count"] == 20
