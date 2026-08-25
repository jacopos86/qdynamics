from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

from chtc.phase3_optuna import generic_static_table_runner as runner
from chtc.phase3_optuna.generic_static_table_runner import (
    _clear_stale_env,
    env_overlay_from_record,
    run_command_with_heartbeat,
)


def test_env_overlay_from_record_exports_table_i_fields() -> None:
    overlay = env_overlay_from_record(
        {
            "suite_profile": "nph2_ref3_v1",
            "energy_stop_target": "1e-6",
            "first_hit_thresholds": "1e-6,1e-8",
            "generic_adapt_stop_policy": "fixed_horizon_no_target_v1",
            "powell_maxiter_cap_policy": "accept_finite_nonincreasing_v1",
            "generic_adapt_runtime_split_mode": "shortlist_pauli_children_v1",
            "generic_adapt_runtime_split_symmetry_policy": "hard_guard",
            "generic_adapt_runtime_split_max_subset_size": "4",
            "phase2_novelty_mode": "legacy_pairwise_v1",
            "shots_per_pauli_term_proxy": "1024",
            "hea_spsa_learning_rate": "0.04",
            "hea_spsa_perturbation": "0.01",
            "family_informed_spsa_a": "0.05",
            "adapt_spsa_c": "0.02",
            "hardware_resolution_mode": "ideal",
            "static_route_id": "route_a",
            "selected_logical_route": "historical_selected",
            "selected_logical_source_json": "chtc/phase3_optuna/input/selected/hk.json",
            "selected_logical_transfer_mode": "exact_match_v1",
            "phase3_policy_json": "chtc/phase3_optuna/input/policies/snake.json",
            "phase3_adapt_max_depth": "2",
            "phase3_adapt_spsa_a": "0.05",
            "phase3_adapt_spsa_c": "0.02",
            "phase3_adapt_spsa_big_a": "50.0",
            "phase3_adapt_allow_repeats": "true",
            "phase3_adapt_parallel_gradient_workers": "2",
            "benchmark_value_noise_model": "",
        }
    )

    assert overlay["TABLE_I_STATIC_SUITE_PROFILE"] == "nph2_ref3_v1"
    assert overlay["GENERIC_STATIC_TABLE_ENERGY_STOP_TARGET"] == "1e-6"
    assert overlay["GENERIC_STATIC_TABLE_FIRST_HIT_THRESHOLDS"] == "1e-6,1e-8"
    assert overlay["GENERIC_STATIC_TABLE_GENERIC_ADAPT_STOP_POLICY"] == "fixed_horizon_no_target_v1"
    assert overlay["GENERIC_STATIC_TABLE_POWELL_MAXITER_CAP_POLICY"] == "accept_finite_nonincreasing_v1"
    assert overlay["GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_MODE"] == "shortlist_pauli_children_v1"
    assert overlay["GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_SYMMETRY_POLICY"] == "hard_guard"
    assert overlay["GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_MAX_SUBSET_SIZE"] == "4"
    assert overlay["PHASE3_POLICY_PHASE2_NOVELTY_MODE"] == "legacy_pairwise_v1"
    assert overlay["GENERIC_STATIC_TABLE_SHOTS_PER_PAULI_TERM_PROXY"] == "1024"
    assert overlay["GENERIC_STATIC_TABLE_HEA_SPSA_LEARNING_RATE"] == "0.04"
    assert overlay["GENERIC_STATIC_TABLE_HEA_SPSA_PERTURBATION"] == "0.01"
    assert overlay["GENERIC_STATIC_TABLE_FAMILY_INFORMED_SPSA_A"] == "0.05"
    assert overlay["GENERIC_STATIC_TABLE_ADAPT_SPSA_C"] == "0.02"
    assert overlay["GENERIC_STATIC_TABLE_HARDWARE_RESOLUTION_MODE"] == "ideal"
    assert overlay["GENERIC_STATIC_TABLE_STATIC_ROUTE_ID"] == "route_a"
    assert overlay["GENERIC_STATIC_TABLE_SELECTED_LOGICAL_ROUTE"] == "historical_selected"
    assert overlay["GENERIC_STATIC_TABLE_SELECTED_LOGICAL_SOURCE_JSON"] == "chtc/phase3_optuna/input/selected/hk.json"
    assert overlay["GENERIC_STATIC_TABLE_SELECTED_LOGICAL_TRANSFER_MODE"] == "exact_match_v1"
    assert overlay["GENERIC_STATIC_TABLE_PHASE3_POLICY_JSON"] == "chtc/phase3_optuna/input/policies/snake.json"
    assert overlay["GENERIC_STATIC_TABLE_PHASE3_ADAPT_MAX_DEPTH"] == "2"
    assert overlay["GENERIC_STATIC_TABLE_PHASE3_ADAPT_SPSA_A"] == "0.05"
    assert overlay["GENERIC_STATIC_TABLE_PHASE3_ADAPT_SPSA_C"] == "0.02"
    assert overlay["GENERIC_STATIC_TABLE_PHASE3_ADAPT_SPSA_BIG_A"] == "50.0"
    assert overlay["GENERIC_STATIC_TABLE_PHASE3_ADAPT_ALLOW_REPEATS"] == "true"
    assert overlay["GENERIC_STATIC_TABLE_PHASE3_ADAPT_PARALLEL_GRADIENT_WORKERS"] == "2"
    assert "GENERIC_STATIC_TABLE_BENCHMARK_VALUE_NOISE_MODEL" not in overlay


def test_clear_stale_env_always_removes_static_route_without_row_column() -> None:
    env = {
        "GENERIC_STATIC_TABLE_STATIC_ROUTE_ID": "route_a",
        "STATIC_ROUTE_ID": "route_a",
        "GENERIC_STATIC_TABLE_SELECTED_LOGICAL_ROUTE": "historical_selected",
        "GENERIC_STATIC_TABLE_SELECTED_LOGICAL_SOURCE_JSON": "old.json",
        "GENERIC_STATIC_TABLE_SELECTED_LOGICAL_TRANSFER_MODE": "boundary_v1",
        "SELECTED_LOGICAL_ROUTE": "historical_selected",
        "GENERIC_STATIC_TABLE_PHASE3_POLICY_JSON": "stale.json",
        "PHASE3_POLICY_PHASE2_NOVELTY_MODE": "legacy_pairwise_v1",
        "PHASE3_POLICY_INNER_OPTIMIZER": "POWELL",
        "GENERIC_STATIC_TABLE_GENERIC_ADAPT_STOP_POLICY": "fixed_horizon_no_target_v1",
        "GENERIC_STATIC_TABLE_POWELL_MAXITER_CAP_POLICY": "accept_finite_nonincreasing_v1",
        "GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_MODE": "shortlist_pauli_children_v1",
        "GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_SYMMETRY_POLICY": "hard_guard",
        "GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_MAX_SUBSET_SIZE": "4",
        "GENERIC_STATIC_TABLE_SHOTS_PER_PAULI_TERM_PROXY": "stale",
        "GENERIC_STATIC_TABLE_HEA_SPSA_LEARNING_RATE": "0.04",
        "GENERIC_STATIC_TABLE_FAMILY_INFORMED_SPSA_A": "0.05",
        "GENERIC_STATIC_TABLE_ADAPT_SPSA_C": "0.02",
        "OTHER_ENV": "kept",
    }

    _clear_stale_env(env, {"family": "hh"})

    assert "GENERIC_STATIC_TABLE_STATIC_ROUTE_ID" not in env
    assert "STATIC_ROUTE_ID" not in env
    assert "GENERIC_STATIC_TABLE_SELECTED_LOGICAL_ROUTE" not in env
    assert "GENERIC_STATIC_TABLE_SELECTED_LOGICAL_SOURCE_JSON" not in env
    assert "GENERIC_STATIC_TABLE_SELECTED_LOGICAL_TRANSFER_MODE" not in env
    assert "SELECTED_LOGICAL_ROUTE" not in env
    assert "GENERIC_STATIC_TABLE_PHASE3_POLICY_JSON" not in env
    assert "PHASE3_POLICY_PHASE2_NOVELTY_MODE" not in env
    assert "PHASE3_POLICY_INNER_OPTIMIZER" not in env
    assert "GENERIC_STATIC_TABLE_GENERIC_ADAPT_STOP_POLICY" not in env
    assert "GENERIC_STATIC_TABLE_POWELL_MAXITER_CAP_POLICY" not in env
    assert "GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_MODE" not in env
    assert "GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_SYMMETRY_POLICY" not in env
    assert "GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_MAX_SUBSET_SIZE" not in env
    assert "GENERIC_STATIC_TABLE_SHOTS_PER_PAULI_TERM_PROXY" not in env
    assert "GENERIC_STATIC_TABLE_HEA_SPSA_LEARNING_RATE" not in env
    assert "GENERIC_STATIC_TABLE_FAMILY_INFORMED_SPSA_A" not in env
    assert "GENERIC_STATIC_TABLE_ADAPT_SPSA_C" not in env
    assert env["OTHER_ENV"] == "kept"


def test_main_prefers_positional_records_path_and_out_root(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def _fake_run_record(*, record_id: str, records_path: str | Path, out_root: str | Path, cwd: str | Path = runner.REPO_ROOT) -> int:
        calls.append(
            {
                "record_id": record_id,
                "records_path": Path(records_path),
                "out_root": Path(out_root),
                "cwd": Path(cwd),
            }
        )
        return 0

    custom_records = tmp_path / "h2_records.tsv"
    custom_out = tmp_path / "h2_out"
    custom_records.write_text("record_id\tfamily\tcase_id\talgorithm_id\nh2\tmolecular_vibronic_h2\tcase\talg\n", encoding="utf-8")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_RECORDS_PATH", "chtc/phase3_optuna/input/generic_static_table_records.tsv")
    monkeypatch.setattr(runner, "run_record", _fake_run_record)

    assert runner.main(["h2", str(custom_records), str(custom_out)]) == 0
    assert calls == [
        {
            "record_id": "h2",
            "records_path": custom_records,
            "out_root": custom_out,
            "cwd": runner.REPO_ROOT,
        }
    ]


def test_apptainer_wrapper_forwards_positional_args_without_records_env_override() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "chtc/phase3_optuna/run_generic_static_table_task_apptainer.sh").read_text(encoding="utf-8")

    assert "GENERIC_STATIC_TABLE_RECORDS_PATH_FOR_CONTAINER" not in text
    assert "GENERIC_STATIC_TABLE_OUT_ROOT_FOR_CONTAINER" not in text
    assert 'run_generic_static_table_task.sh "$@"' in text
    assert '-- "$@"' in text


def test_run_command_with_heartbeat_updates_before_child_exit(tmp_path: Path) -> None:
    heartbeat = tmp_path / "heartbeat.json"
    events = tmp_path / "heartbeat_events.jsonl"
    command = [
        sys.executable,
        "-u",
        "-c",
        (
            "import json, time; "
            "print('AI_LOG ' + json.dumps({'event':'hardcoded_adapt_iter_done','depth':4,"
            "'delta_abs_current':0.25,'phase1_shortlist_size':9}), flush=True); "
            "time.sleep(1.0)"
        ),
    ]
    result: dict[str, int] = {}

    def _target() -> None:
        result["returncode"] = run_command_with_heartbeat(
            command,
            cwd=tmp_path,
            env=None,
            stdout_path=tmp_path / "stdout.log",
            stderr_path=tmp_path / "stderr.log",
            heartbeat_path=heartbeat,
            heartbeat_events_path=events,
            metadata={"record_id": "fake"},
            echo_stdout=False,
        )

    thread = threading.Thread(target=_target)
    thread.start()
    observed_live = False
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if heartbeat.exists():
            data = json.loads(heartbeat.read_text(encoding="utf-8"))
            progress = data.get("progress", {})
            if progress.get("depth") == 4 and thread.is_alive():
                observed_live = True
                break
        time.sleep(0.05)
    thread.join(timeout=5.0)

    assert observed_live
    assert result["returncode"] == 0
    final = json.loads(heartbeat.read_text(encoding="utf-8"))
    assert final["status"] == "completed"
    assert final["progress"]["phase1_shortlist_size"] == 9
    assert events.exists()


def test_run_command_with_heartbeat_receives_forwarded_nested_ai_log(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    inner_script = tmp_path / "inner_adapt.py"
    inner_script.write_text(
        "import json, time\n"
        "print('AI_LOG ' + json.dumps({"
        "'event':'hardcoded_adapt_gradient_timing',"
        "'depth':2,'energy':-1.5,'candidate_count':7}), flush=True)\n"
        "time.sleep(0.5)\n",
        encoding="utf-8",
    )
    outer_script = tmp_path / "outer_runner.py"
    outer_script.write_text(
        "import sys, time\n"
        "from pathlib import Path\n"
        "from pipelines.exact_bench.static_benchmark_runtime import _run_subprocess_logged\n"
        f"root = Path({str(tmp_path)!r})\n"
        f"inner = Path({str(inner_script)!r})\n"
        "rc, elapsed = _run_subprocess_logged(\n"
        "    [sys.executable, '-u', str(inner)],\n"
        "    cwd=root,\n"
        "    stdout_path=root / 'inner_stdout.log',\n"
        "    stderr_path=root / 'inner_stderr.log',\n"
        ")\n"
        "time.sleep(0.5)\n"
        "raise SystemExit(rc)\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    heartbeat = tmp_path / "heartbeat.json"
    events = tmp_path / "heartbeat_events.jsonl"
    result: dict[str, int] = {}

    def _target() -> None:
        result["returncode"] = run_command_with_heartbeat(
            [sys.executable, "-u", str(outer_script)],
            cwd=tmp_path,
            env=env,
            stdout_path=tmp_path / "stdout.log",
            stderr_path=tmp_path / "stderr.log",
            heartbeat_path=heartbeat,
            heartbeat_events_path=events,
            metadata={"record_id": "nested-fake"},
            echo_stdout=False,
        )

    thread = threading.Thread(target=_target)
    thread.start()
    observed_live = False
    deadline = time.time() + 8.0
    while time.time() < deadline:
        if heartbeat.exists():
            data = json.loads(heartbeat.read_text(encoding="utf-8"))
            progress = data.get("progress", {})
            if progress.get("depth") == 2 and progress.get("candidate_count") == 7 and thread.is_alive():
                observed_live = True
                break
        time.sleep(0.05)
    thread.join(timeout=8.0)

    assert observed_live
    assert result["returncode"] == 0
    final = json.loads(heartbeat.read_text(encoding="utf-8"))
    assert final["status"] == "completed"
    assert final["progress"]["energy"] == -1.5
    assert "AI_LOG" in (tmp_path / "inner_stdout.log").read_text(encoding="utf-8")
    assert "AI_LOG" in (tmp_path / "stdout.log").read_text(encoding="utf-8")
