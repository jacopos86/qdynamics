from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_pareto_tracking import (
    compute_pareto_frontier,
    extract_staged_hh_pareto_rows,
    write_pareto_tracking,
)


def _physics() -> dict[str, object]:
    return {
        "L": 2,
        "t": 1.0,
        "u": 2.0,
        "dv": 0.0,
        "omega0": 1.0,
        "g_ep": 1.0,
        "n_ph_max": 1,
    }


def test_extract_staged_hh_pareto_rows_accumulates_measurement_and_gate_cost() -> None:
    rows = extract_staged_hh_pareto_rows(
        run_tag="hh_run",
        physics=_physics(),
        warm_payload={
            "ansatz": "hh_hva_ptw",
            "energy": -1.00,
            "exact_filtered_energy": -1.10,
            "num_parameters": 4,
        },
        adapt_payload={
            "continuation_mode": "phase3_v1",
            "pool_type": "paop_full",
            "exact_gs_energy": -1.10,
            "optimal_point": [0.1, 0.2, 0.3, 0.4],
            "history": [
                {
                    "depth": 1,
                    "depth_cumulative": 1,
                    "batch_size": 1,
                    "candidate_family": "paop_full",
                    "selection_mode": "append",
                    "energy_after_opt": -1.04,
                    "delta_abs_current": 0.06,
                    "delta_abs_drop_from_prev": 0.04,
                    "measurement_cache_stats": {
                        "groups_new": 2,
                        "shots_new": 2000.0,
                        "reuse_count_cost": 2.0,
                    },
                    "compile_cost_proxy": {
                        "gate_proxy_total": 5.0,
                        "cx_proxy_total": 3.0,
                        "sq_proxy_total": 4.0,
                        "max_pauli_weight": 2.0,
                    },
                },
                {
                    "depth": 2,
                    "depth_cumulative": 2,
                    "batch_size": 2,
                    "candidate_family": "paop_full",
                    "selection_mode": "batch",
                    "energy_after_opt": -1.08,
                    "delta_abs_current": 0.02,
                    "delta_abs_drop_from_prev": 0.04,
                    "runtime_split_mode": "shortlist_pauli_children_v1",
                    "runtime_split_child_count": 2,
                    "measurement_cache_stats": {
                        "groups_new": 1,
                        "shots_new": 1000.0,
                        "reuse_count_cost": 1.0,
                    },
                    "compile_cost_proxy": {
                        "gate_proxy_total": 7.0,
                        "cx_proxy_total": 4.0,
                        "sq_proxy_total": 6.0,
                        "max_pauli_weight": 3.0,
                    },
                },
            ],
        },
        replay_payload={
            "replay_contract": {"continuation_mode": "phase3_v1"},
            "generator_family": {"resolved": "paop_full"},
            "exact": {"E_exact_sector": -1.10},
            "vqe": {
                "energy": -1.099,
                "abs_delta_e": 0.001,
                "num_parameters": 8,
            },
        },
        recorded_utc="2026-03-20T12:00:00Z",
    )

    assert len(rows) == 4
    warm, adapt_1, adapt_2, replay = rows
    assert warm.stage_kind == "warm_start"
    assert bool(warm.pareto_eligible) is False

    assert adapt_1.stage_kind == "adapt_depth"
    assert adapt_1.num_parameters == 2
    assert adapt_1.measurement_groups_cumulative == 2
    assert adapt_1.compile_gate_proxy_cumulative == 5.0
    assert adapt_1.delta_E_drop_per_new_group == 0.02

    assert adapt_2.num_parameters == 4
    assert adapt_2.measurement_groups_cumulative == 3
    assert adapt_2.measurement_shots_cumulative == 3000.0
    assert adapt_2.compile_gate_proxy_cumulative == 12.0
    assert adapt_2.compile_cx_proxy_cumulative == 7.0
    assert adapt_2.runtime_split_mode == "shortlist_pauli_children_v1"
    assert adapt_2.runtime_split_child_count == 2
    assert bool(adapt_2.pareto_eligible) is True

    assert replay.stage_kind == "conventional_replay"
    assert bool(replay.pareto_eligible) is False


def test_compute_pareto_frontier_filters_dominated_rows() -> None:
    frontier = compute_pareto_frontier(
        [
            {
                "run_tag": "a",
                "stage_depth": 1,
                "delta_E_abs": 0.05,
                "measurement_groups_cumulative": 2,
                "compile_gate_proxy_cumulative": 10,
                "pareto_eligible": True,
            },
            {
                "run_tag": "a",
                "stage_depth": 2,
                "delta_E_abs": 0.04,
                "measurement_groups_cumulative": 3,
                "compile_gate_proxy_cumulative": 9,
                "pareto_eligible": True,
            },
            {
                "run_tag": "a",
                "stage_depth": 3,
                "delta_E_abs": 0.06,
                "measurement_groups_cumulative": 4,
                "compile_gate_proxy_cumulative": 12,
                "pareto_eligible": True,
            },
        ]
    )

    assert [row["stage_depth"] for row in frontier] == [2, 1]


def test_write_pareto_tracking_replaces_existing_run_tag_rows(tmp_path: Path) -> None:
    run_json = tmp_path / "hh_run.json"
    rows = extract_staged_hh_pareto_rows(
        run_tag="hh_run",
        physics=_physics(),
        warm_payload={"energy": -1.0, "exact_filtered_energy": -1.1},
        adapt_payload={
            "continuation_mode": "phase3_v1",
            "pool_type": "paop_full",
            "exact_gs_energy": -1.1,
            "optimal_point": [0.1],
            "history": [
                {
                    "depth": 1,
                    "depth_cumulative": 1,
                    "batch_size": 1,
                    "energy_after_opt": -1.05,
                    "delta_abs_current": 0.05,
                    "delta_abs_drop_from_prev": 0.05,
                    "measurement_cache_stats": {"groups_new": 1, "shots_new": 1000.0, "reuse_count_cost": 1.0},
                    "compile_cost_proxy": {"gate_proxy_total": 4.0, "cx_proxy_total": 2.0, "sq_proxy_total": 4.0},
                }
            ],
        },
        replay_payload={"vqe": {"energy": -1.09, "abs_delta_e": 0.01}, "exact": {"E_exact_sector": -1.1}},
        recorded_utc="2026-03-20T12:00:00Z",
    )

    out1 = write_pareto_tracking(rows=rows, output_json_path=run_json, run_tag="hh_run")
    assert Path(out1["paths"]["run_rows_json"]).exists()
    assert Path(out1["paths"]["rolling_frontier_json"]).exists()

    trimmed_rows = rows[:2]
    out2 = write_pareto_tracking(rows=trimmed_rows, output_json_path=run_json, run_tag="hh_run")
    ledger_path = Path(out2["paths"]["rolling_ledger_jsonl"])
    ledger_lines = [line for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(ledger_lines) == len(trimmed_rows)

    frontier_payload = json.loads(Path(out2["paths"]["run_frontier_json"]).read_text(encoding="utf-8"))
    assert frontier_payload["frontier_count"] == 1
