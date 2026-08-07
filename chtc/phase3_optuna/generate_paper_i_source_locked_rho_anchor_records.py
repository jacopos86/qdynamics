#!/usr/bin/env python3
"""Generate source-locked Paper-I rho anchor CHTC records.

This generator is intentionally not an Optuna/oracle-grid generator.  It stages
the original visible-row command scripts and only authorizes source-value rho
anchors that can be replayed from those commands.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "chtc" / "phase3_optuna"
BATCH_ID = "paper_i_source_locked_rho_anchors_20260609_v1"
INPUT_DIR = SCRIPT_DIR / "input" / BATCH_ID
SOURCES_DIR = INPUT_DIR / "sources"
RHO_SOURCE_VALUE = 0.25


@dataclass(frozen=True)
class ExecutableAnchor:
    case_id: str
    table_row: str
    hamiltonian_family: str
    regime: str
    source_class: str
    source_result_json: str | None
    source_command_sh: str
    source_effective_manifest_json: str | None
    source_selected_logical_json: str | None = None
    visible_source_json: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class BlockedRow:
    case_id: str
    table_row: str
    hamiltonian_family: str
    regime: str
    reason: str
    visible_source: str


EXECUTABLE_ANCHORS = [
    ExecutableAnchor(
        case_id="hubbard_strong_rho0p25_anchor",
        table_row="Table I SNAKE Hubbard strong",
        hamiltonian_family="Hubbard",
        regime="strong",
        source_class="result_json_with_command_and_effective_manifest",
        source_result_json="/Users/jakestrobel/LocalProjects/Holstein_test_fullclone_3_local_outputs/raw_outputs/chtc_fetches/rho_sweep_7566650_fixed_settings_20260608/raw_outputs/routeA_paper_i_fixed_settings_rho_sweep_20260607_v1_hubbard_strong_rho0p25/run/hubbard_L2_three_model_strong/trial_0000/hubbard_L2_three_model_strong/json/result.json",
        source_command_sh="/Users/jakestrobel/LocalProjects/Holstein_test_fullclone_3_local_outputs/raw_outputs/chtc_fetches/rho_sweep_7566650_fixed_settings_20260608/raw_outputs/routeA_paper_i_fixed_settings_rho_sweep_20260607_v1_hubbard_strong_rho0p25/run/hubbard_L2_three_model_strong/trial_0000/hubbard_L2_three_model_strong/logs/command.sh",
        source_effective_manifest_json="/Users/jakestrobel/LocalProjects/Holstein_test_fullclone_3_local_outputs/raw_outputs/chtc_fetches/rho_sweep_7566650_fixed_settings_20260608/raw_outputs/routeA_paper_i_fixed_settings_rho_sweep_20260607_v1_hubbard_strong_rho0p25/run/hubbard_L2_three_model_strong/trial_0000/effective_trial_manifest.json",
        source_selected_logical_json="/Users/jakestrobel/LocalProjects/Holstein_test_fullclone_3_local_outputs/raw_outputs/chtc_fetches/rho_sweep_7566650_fixed_settings_20260608/raw_outputs/routeA_paper_i_fixed_settings_rho_sweep_20260607_v1_hubbard_strong_rho0p25/run/selected_logical_sources/hubbard_L2_three_model_strong.selected_logical.json",
        visible_source_json="MATH/paper_facing/paper_I_static_scaffold/paper_i_hubbard_strong_settings_on_weak_replay_audit_20260608.json",
    ),
    ExecutableAnchor(
        case_id="spin_boson_strong_rho0p25_anchor",
        table_row="Table II SNAKE spin-boson strong",
        hamiltonian_family="Spin-boson",
        regime="strong",
        source_class="result_json_with_command_and_effective_manifest",
        source_result_json="artifacts/agent_runs/spin_boson_snake_pauli_children_no_shots_local_optuna_20260527_v1/strong/run/spin_boson_L2_nph2_three_model_strong/trial_0737/spin_boson_L2_nph2_three_model_strong/json/result.json",
        source_command_sh="artifacts/agent_runs/spin_boson_snake_pauli_children_no_shots_local_optuna_20260527_v1/strong/run/spin_boson_L2_nph2_three_model_strong/trial_0737/spin_boson_L2_nph2_three_model_strong/logs/command.sh",
        source_effective_manifest_json="artifacts/agent_runs/spin_boson_snake_pauli_children_no_shots_local_optuna_20260527_v1/strong/run/spin_boson_L2_nph2_three_model_strong/trial_0737/effective_trial_manifest.json",
        visible_source_json="MATH/paper_facing/paper_I_static_scaffold/paper_i_spin_boson_strong_settings_on_weak_replay_audit_20260608.json",
        notes="Original command relied on the code default for phase2_rho; the anchor sets the same source value explicitly.",
    ),
    ExecutableAnchor(
        case_id="hh_weak_strong_rho0p25_anchor",
        table_row="Table III SNAKE HH weak-strong",
        hamiltonian_family="Hubbard-Holstein",
        regime="weak-strong",
        source_class="stdout_history_with_command_no_result_json",
        source_result_json=None,
        source_command_sh="raw_outputs/chtc_fetches/hh_snake_structural_continue_7096352_live_logs_20260601/extracted/raw_outputs/routeA_paper_i_hh_snake_structural_continue_20260531_v1/weak_strong/logs/command.sh",
        source_effective_manifest_json=None,
        visible_source_json="MATH/paper_facing/paper_I_static_scaffold/history_sources/hh_tableiii_weak_strong_snake_structural_continue_7096352_stdout_20260601.json",
        notes="Weaker anchor: visible source is stdout-derived, so the runner can compare scalar energy/depth only.",
    ),
    ExecutableAnchor(
        case_id="hh_strong_strong_rho0p25_anchor",
        table_row="Table III SNAKE HH strong-strong",
        hamiltonian_family="Hubbard-Holstein",
        regime="strong-strong",
        source_class="stdout_history_with_command_no_result_json",
        source_result_json=None,
        source_command_sh="raw_outputs/chtc_fetches/hh_snake_structural_continue_7096352_live_logs_20260601/extracted/raw_outputs/routeA_paper_i_hh_snake_structural_continue_20260531_v1/strong_strong/logs/command.sh",
        source_effective_manifest_json=None,
        visible_source_json="MATH/paper_facing/paper_I_static_scaffold/history_sources/hh_tableiii_strong_strong_snake_structural_continue_7096352_stdout_20260601.json",
        notes="Weaker anchor: visible source is stdout-derived, so the runner can compare scalar energy/depth only.",
    ),
]


BLOCKED_ROWS = [
    BlockedRow(
        case_id="hubbard_weak",
        table_row="Table I SNAKE Hubbard weak",
        hamiltonian_family="Hubbard",
        regime="weak",
        reason="Visible source has result JSON only; no adjacent command.sh or effective manifest was found, so a source-locked candidate anchor would require reconstruction.",
        visible_source="tmp/paper_i_hubbard_strong_settings_on_weak_20260608_v1/run/hubbard_L2_three_model_weak/trial_0000/hubbard_L2_three_model_weak/json/result.json",
    ),
    BlockedRow(
        case_id="spin_boson_weak",
        table_row="Table II SNAKE spin-boson weak",
        hamiltonian_family="Spin-boson",
        regime="weak",
        reason="Visible source has result JSON only; no adjacent command.sh or effective manifest was found, so a source-locked candidate anchor would require reconstruction.",
        visible_source="tmp/paper_i_spin_boson_strong_settings_on_weak_20260608_v1/run/spin_boson_L2_nph1_three_model_weak/trial_0000/spin_boson_L2_nph1_three_model_weak/json/result.json",
    ),
    BlockedRow(
        case_id="hh_weak_weak",
        table_row="Table III SNAKE HH weak-weak",
        hamiltonian_family="Hubbard-Holstein",
        regime="weak-weak",
        reason="Visible source is a live checkpoint/current promotion with strict replay JSON but no original command.sh/effective manifest found locally.",
        visible_source="output/pdf/paper_i_table_iii_snake_weak_weak_live_prefix_promotion_20260530.json",
    ),
    BlockedRow(
        case_id="hh_strong_weak",
        table_row="Table III SNAKE HH strong-weak",
        hamiltonian_family="Hubbard-Holstein",
        regime="strong-weak",
        reason="Current visible source is stdout-held continuation with no replayable current/result JSON; previous JSON-backed compiled-resource source lacks adjacent command/effective manifest locally.",
        visible_source="MATH/paper_facing/paper_I_static_scaffold/history_sources/hh_tableiii_strong_weak_snake_continue_7582403_stdout_20260609.json",
    ),
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _repo_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def _copy_required(src: str, dest: Path) -> dict[str, Any]:
    src_path = _repo_path(src)
    if not src_path.exists():
        raise FileNotFoundError(f"missing required source: {src_path}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dest)
    return {
        "original_path": str(src_path),
        "staged_path": str(dest.relative_to(REPO_ROOT)),
        "sha256": _sha256(dest),
    }


def _copy_optional(src: str | None, dest: Path) -> dict[str, Any] | None:
    if not src:
        return None
    return _copy_required(src, dest)


def main() -> None:
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)

    specs: list[dict[str, Any]] = []
    planning_errors: list[dict[str, str]] = []
    for anchor in EXECUTABLE_ANCHORS:
        case_dir = SOURCES_DIR / anchor.case_id
        try:
            staged_command = _copy_required(anchor.source_command_sh, case_dir / "source_command.sh")
            staged_result = _copy_optional(anchor.source_result_json, case_dir / "source_result.json")
            staged_manifest = _copy_optional(anchor.source_effective_manifest_json, case_dir / "source_effective_manifest.json")
            staged_selected_logical = _copy_optional(
                anchor.source_selected_logical_json,
                case_dir / "source_selected_logical.json",
            )
            staged_visible_source = _copy_optional(anchor.visible_source_json, case_dir / "visible_source.json")
        except FileNotFoundError as exc:
            planning_errors.append({"case_id": anchor.case_id, "error": str(exc)})
            continue

        spec = asdict(anchor)
        spec.update(
            {
                "batch_id": BATCH_ID,
                "rho_variable": "phase2_rho",
                "rho_source_value": RHO_SOURCE_VALUE,
                "rho_grid_after_anchor": [0.05, 0.1, 0.25, 0.5, 1.0],
                "run_class": "source_value_anchor",
                "allowed_non_output_mutations": [
                    "python executable path normalization",
                    "--output-json path",
                    "--adapt-current-json path",
                    "explicit --phase2-rho source value if the original command used the code default",
                    "portable rewrite of staged selected-logical source path",
                ],
                "staged_command": staged_command,
                "staged_result": staged_result,
                "staged_effective_manifest": staged_manifest,
                "staged_selected_logical": staged_selected_logical,
                "staged_visible_source": staged_visible_source,
            }
        )
        specs.append(spec)

    if planning_errors:
        raise SystemExit(f"planning errors: {planning_errors}")

    specs_path = INPUT_DIR / "source_locked_rho_anchor_specs.json"
    specs_path.write_text(json.dumps({"schema": "paper_i_source_locked_rho_anchor_specs_v1", "batch_id": BATCH_ID, "anchors": specs}, indent=2) + "\n")

    case_ids_path = INPUT_DIR / "source_locked_rho_anchor_case_ids.txt"
    case_ids_path.write_text("\n".join(spec["case_id"] for spec in specs) + "\n")

    audit = {
        "schema": "source_locked_sensitivity_planning_audit_v1",
        "batch_id": BATCH_ID,
        "purpose": "Paper-I rho source-value anchors before any five-point rho fanout.",
        "forbidden_wrappers": ["phase3_policy_optuna", "oracle-grid", "search/grid generators"],
        "enforced_policy": [
            "no Optuna or oracle-grid runner",
            "source-value anchor first",
            "do not reconstruct missing replay commands from result JSON settings",
            "only phase2_rho may be varied after anchors pass",
        ],
        "eligible_anchor_count": len(specs),
        "eligible_anchor_case_ids": [spec["case_id"] for spec in specs],
        "blocked_row_count": len(BLOCKED_ROWS),
        "blocked_rows": [asdict(row) for row in BLOCKED_ROWS],
    }
    (INPUT_DIR / "source_locked_rho_anchor_planning_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    submit_path = SCRIPT_DIR / f"submit_{BATCH_ID}.sub"
    submit_path.write_text(
        "\n".join(
            [
                "universe = vanilla",
                "executable = chtc/phase3_optuna/run_source_locked_rho_anchor_task_apptainer.sh",
                "arguments = $(case_id)",
                "should_transfer_files = YES",
                "when_to_transfer_output = ON_EXIT_OR_EVICT",
                "transfer_executable = True",
                "preserve_relative_paths = True",
                "transfer_input_files = pipelines, src, docs, test_support, chtc/phase3_optuna",
                "transfer_output_files = raw_outputs, logs",
                f"log = logs/{BATCH_ID}.$(Cluster).$(Process).log",
                f"output = logs/{BATCH_ID}.$(Cluster).$(Process).out",
                f"error = logs/{BATCH_ID}.$(Cluster).$(Process).err",
                "stream_output = False",
                "stream_error = False",
                "requirements = TARGET.HasSIF",
                "request_cpus = 10",
                "request_memory = 32GB",
                "request_disk = 122880MB",
                "+MaxRuntime = 172800",
                f'+JobBatchName = "holstein-{BATCH_ID}"',
                "environment = \"PHASE3_TERMINATE_ON_STALE_PROGRESS=1 PHASE3_REQUIRE_FIRST_PROGRESS_WITHIN_SEC=3600 PHASE3_PROGRESS_STALE_AFTER_SEC=3600 PHASE3_HEARTBEAT_INTERVAL_SEC=60 PHASE3_SHELL_HEARTBEAT_SEC=60\"",
                f"queue case_id from chtc/phase3_optuna/input/{BATCH_ID}/source_locked_rho_anchor_case_ids.txt",
                "",
            ]
        )
    )

    print(json.dumps({"batch_id": BATCH_ID, "specs": str(specs_path), "submit": str(submit_path), "anchors": len(specs)}, indent=2))


if __name__ == "__main__":
    main()
