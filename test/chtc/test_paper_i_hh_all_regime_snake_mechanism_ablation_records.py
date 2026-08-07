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
    "paper_i_hh_all_regime_snake_mechanism_ablation_records",
    "chtc/phase3_optuna/generate_paper_i_hh_all_regime_snake_mechanism_ablation_records.py",
)
preflight_submit = _load_script_module(
    "phase3_preflight_submit_for_all_regime_mechanism_ablation",
    "chtc/phase3_optuna/preflight_submit.py",
)
runner = _load_script_module(
    "paper_i_hh_spsa_budget_ladder_cell_for_all_regime_mechanism_ablation",
    "chtc/phase3_optuna/run_paper_i_hh_spsa_budget_ladder_cell.py",
)


def _arg_value(args: list[str], flag: str) -> str:
    assert flag in args
    idx = args.index(flag)
    assert idx + 1 < len(args)
    return args[idx + 1]


def _rows() -> list[dict[str, str]]:
    return generator.build_records("paper_i_hh_all_regime_snake_mechanism_ablation_unit")


def test_all_regime_mechanism_ablation_matrix_shape_and_contract() -> None:
    rows = _rows()
    assert len(rows) == 78
    assert {row["runnable"] for row in rows} == {"true"}
    assert {row["source_anchor_family"] for row in rows} == {"physical_operator_lane"}
    assert Counter(row["display_regime"] for row in rows) == {
        "weak-weak": 13,
        "intermediate-weak": 13,
        "strong-weak": 13,
        "weak-strong": 13,
        "intermediate-strong": 13,
        "strong-strong": 13,
    }
    assert {row["method_key"] for row in rows} == {"snake"}
    assert {row["optimizer"] for row in rows} == {"POWELL"}
    assert {row["adapt_optimizer_kind"] for row in rows} == {"powell"}
    assert {row["budget"] for row in rows} == {"200"}
    assert {row["max_depth"] for row in rows} == {"30"}
    assert {row["pool_contract"] for row in rows} == {"full_meta_unfiltered"}
    assert {row["adapt_pool_class_filter_json"] for row in rows} == {"off"}
    assert all("hh_full_meta_minus_hva_class_filter.json" not in json.dumps(row) for row in rows)
    assert "no_shortlisting" not in {row["hh_mechanism_ablation_variant"] for row in rows}
    assert "full_geometry_window" not in {row["hh_mechanism_ablation_variant"] for row in rows}


def test_all_regime_batching_is_phase3_only_and_child_cap_stays_one() -> None:
    rows = _rows()
    batch_rows = [row for row in rows if row["phase3_batch_selection_mode"]]
    assert batch_rows
    assert {row["child_subset_size"] for row in rows if row["child_policy"] == "native_phase3_singleton"} == {"1"}
    assert {row["snake_phase3_runtime_split_max_subset_size"] for row in rows if row["child_policy"] == "native_phase3_singleton"} == {"1"}
    assert {row["phase3_batch_target_size"] for row in batch_rows} == {"3"}
    assert {row["phase3_batch_size_cap"] for row in batch_rows} == {"3"}
    assert {row["phase2_batch_selection_mode"] for row in rows} == {""}
    assert {row["phase2_batch_target_size"] for row in rows} == {""}
    assert {row["phase2_batch_size_cap"] for row in rows} == {""}
    for row in batch_rows:
        overrides = json.loads(row["snake_cli_overrides_json"])
        assert "--phase3-enable-batching" in overrides["enable_flags"]
        assert "--phase2-no-batching" in overrides["enable_flags"]
        assert "--phase2-enable-batching" not in overrides["enable_flags"]
        assert overrides["set_flags"]["--phase3-runtime-split-max-subset-size"] == "1"
        assert overrides["set_flags"]["--phase3-batch-target-size"] == "3"
        assert overrides["set_flags"]["--phase3-batch-size-cap"] == "3"
        assert "--phase2-batch-selection-mode" in overrides["remove_value_flags"]


def test_all_regime_new_ablation_overrides() -> None:
    rows = {(row["display_regime"], row["hh_mechanism_ablation_variant"]): row for row in _rows()}
    no_beam = rows[("weak-weak", "no_beam")]
    no_beam_flags = json.loads(no_beam["snake_cli_overrides_json"])["set_flags"]
    assert no_beam["adapt_beam_live_branches"] == "1"
    assert no_beam["adapt_beam_children_per_parent"] == "1"
    assert no_beam_flags["--adapt-beam-live-branches"] == "1"
    assert no_beam_flags["--adapt-beam-children-per-parent"] == "1"

    no_lane = rows[("weak-weak", "no_lane_global_pool")]
    no_lane_overrides = json.loads(no_lane["snake_cli_overrides_json"])
    assert no_lane["static_lane_route"] == "algebraic"
    assert no_lane_overrides["set_flags"]["--static-lane-route"] == "algebraic"
    assert "--physical-lane-shortlist-aggressiveness" in no_lane_overrides["remove_value_flags"]


def test_all_regime_rows_pass_dedicated_preflight_contract() -> None:
    for row in _rows():
        assert preflight_submit._is_hh_all_regime_snake_mechanism_ablation(row)
        blockers = preflight_submit._hh_all_regime_snake_mechanism_ablation_contract_blockers(row)
        assert blockers == [], (row["record_id"], blockers)


def test_all_regime_effective_commands_for_anchor_no_beam_and_no_lane(tmp_path: Path) -> None:
    rows = {(row["display_regime"], row["hh_mechanism_ablation_variant"]): row for row in _rows()}
    anchor = rows[("weak-weak", "combinatorial_cap3_anchor")]
    _source_cmd, effective_cmd, audit = runner.build_snake_source_locked_command(anchor, tmp_path / "anchor")
    assert audit["status"] == "pass", audit
    assert _arg_value(effective_cmd, "--adapt-inner-optimizer") == "POWELL"
    assert _arg_value(effective_cmd, "--adapt-maxiter") == "200"
    assert _arg_value(effective_cmd, "--adapt-final-refit-maxiter") == "200"
    assert _arg_value(effective_cmd, "--adapt-max-depth") == "30"
    assert _arg_value(effective_cmd, "--adapt-reopt-policy") == "full"
    assert _arg_value(effective_cmd, "--adapt-window-size") == "99"
    assert _arg_value(effective_cmd, "--phase3-geometry-window-size") == "99"
    assert _arg_value(effective_cmd, "--static-lane-route") == "physical_operator_type"
    assert _arg_value(effective_cmd, "--phase3-runtime-split-max-subset-size") == "1"
    assert _arg_value(effective_cmd, "--phase3-batch-selection-mode") == "combinatorial_reduced_plane"
    assert _arg_value(effective_cmd, "--phase3-batch-target-size") == "3"
    assert _arg_value(effective_cmd, "--phase3-batch-size-cap") == "3"
    assert "--phase3-enable-batching" in effective_cmd
    assert "--phase2-no-batching" in effective_cmd
    assert "--phase2-enable-batching" not in effective_cmd

    no_beam = rows[("weak-weak", "no_beam")]
    _source_cmd, no_beam_cmd, audit = runner.build_snake_source_locked_command(no_beam, tmp_path / "no_beam")
    assert audit["status"] == "pass", audit
    assert _arg_value(no_beam_cmd, "--adapt-beam-live-branches") == "1"
    assert _arg_value(no_beam_cmd, "--adapt-beam-children-per-parent") == "1"

    no_lane = rows[("weak-weak", "no_lane_global_pool")]
    _source_cmd, no_lane_cmd, audit = runner.build_snake_source_locked_command(no_lane, tmp_path / "no_lane")
    assert audit["status"] == "pass", audit
    assert _arg_value(no_lane_cmd, "--static-lane-route") == "algebraic"
    assert "--physical-lane-shortlist-aggressiveness" not in no_lane_cmd


def test_all_regime_write_records_and_preflight_bundle() -> None:
    batch_id = "paper_i_hh_all_regime_snake_mechanism_ablation_unit_write"
    rows = generator.build_records(batch_id)
    manifest = generator.write_records(
        batch_id,
        rows,
        request_cpus=1,
        request_memory_mb=32768,
        request_disk_mb=61440,
        max_runtime_s=172800,
    )
    assert manifest["schema"] == "paper_i_hh_all_regime_snake_mechanism_ablation_manifest_v1"
    assert manifest["record_count"] == 78
    assert manifest["runnable_record_count"] == 78
    assert manifest["expected_runnable_rows"] == 78

    input_dir = ROOT / "chtc" / "phase3_optuna" / "input" / batch_id
    records_tsv = input_dir / "paper_i_hh_spsa_budget_ladder_records.tsv"
    with records_tsv.open(newline="", encoding="utf-8") as fh:
        read_back = list(csv.DictReader(fh, delimiter="\t"))
    assert len(read_back) == 78
    assert "phase3_batch_selection_mode" in read_back[0]

    submit_path = ROOT / "chtc" / "phase3_optuna" / f"submit_{batch_id}.sub"
    submit_text = submit_path.read_text(encoding="utf-8")
    assert "raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708/weak_weak/json/result.json" in submit_text
    assert "raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708/strong_strong/json/result.json" in submit_text
    preflight = preflight_submit.build_preflight_bundle(
        submit_path=submit_path,
        records_path=records_tsv,
        record_id_file=input_dir / "paper_i_hh_spsa_budget_ladder_record_ids.txt",
    )
    assert preflight["ok"], preflight["blocking_reasons"]
    assert preflight["record_count"] == 78
