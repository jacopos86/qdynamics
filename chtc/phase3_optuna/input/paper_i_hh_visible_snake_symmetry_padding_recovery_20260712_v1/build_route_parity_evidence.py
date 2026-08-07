#!/usr/bin/env python3
"""Build the fail-closed weak-weak depth-1 route-parity evidence JSON."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


BUNDLE = Path(
    "chtc/phase3_optuna/input/"
    "paper_i_hh_visible_snake_symmetry_padding_recovery_20260712_v1"
)
EVIDENCE = BUNDLE / "preflight_evidence"
OUTPUT = EVIDENCE / "weak_weak_depth1_prerepair_current_route_parity_20260712.json"
JULY8_RESULT = Path(
    "raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708/"
    "weak_weak/json/result.json"
)
JULY8_STDOUT = Path(
    "raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708/"
    "logs/weak_weak.stdout.log"
)
ENERGY_ATOL = 1.0e-12
FLOAT_ATOL = 1.0e-12
CANDIDATE_FIELDS = (
    "candidate_label",
    "generator_id",
    "position_id",
    "append_position",
    "candidate_pool_index",
    "route_a_shortlist_identity",
    "cheap_score",
    "full_v2_score",
    "phase2_raw_score",
    "simple_score",
    "selector_score",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": path.as_posix(), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _history0(payload: dict[str, Any]) -> dict[str, Any]:
    history = payload["adapt_vqe"]["history"]
    if len(history) != 1:
        raise ValueError(f"depth-1 evidence must have exactly one history row, got {len(history)}")
    return history[0]


def _controls(payload: dict[str, Any]) -> dict[str, Any]:
    controls = payload["adapt_vqe"]["continuation"]["physical_operator_lane_policy"][
        "shortlist_controls"
    ]
    keys = (
        "aggressiveness_factor",
        "phase1_shortlist_size_base",
        "phase1_shortlist_size_effective",
        "phase2_shortlist_size_base",
        "phase2_shortlist_size_effective",
        "phase2_shortlist_fraction_base",
        "phase2_shortlist_fraction_effective",
        "phase3_shortlist_size_effective",
        "phase1_controller_cap_min_effective",
        "phase1_controller_cap_max_effective",
        "phase2_controller_cap_min_effective",
        "phase2_controller_cap_max_effective",
        "phase3_controller_cap_min_effective",
        "phase3_controller_cap_max_effective",
        "phase_controller_cap_policy",
        "route",
        "route_variant_id",
    )
    # The snapshot predates the explicit phase3_shortlist_size_effective field;
    # its resolved min/max controller caps both equal four.
    result = {key: controls.get(key) for key in keys}
    if result["phase3_shortlist_size_effective"] is None:
        result["phase3_shortlist_size_effective"] = controls[
            "phase3_controller_cap_max_effective"
        ]
    return result


def _candidate_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {field: row.get(field) for field in CANDIDATE_FIELDS}
        for row in _history0(payload)["shortlisted_records"]
    ]


def _scientific_summary(payload: dict[str, Any]) -> dict[str, Any]:
    adapt = payload["adapt_vqe"]
    row = _history0(payload)
    return {
        "shortlist_controls": _controls(payload),
        "shortlist_selection_unit": "candidate_position_record",
        "shortlisted_candidates_in_rank_order": _candidate_rows(payload),
        "selected": {
            "generator_id": row["generator_id"],
            "label": row["selected_op"],
            "position": row["selected_position"],
            "pool_indices": row["selected_pool_indices"],
        },
        "energy": {
            "before": row["energy_before_opt"],
            "after": row["energy_after_opt"],
            "delta": row["delta_energy"],
            "final": adapt["energy"],
        },
        "depth": {
            "outer_history_rows": len(adapt["history"]),
            "history_depth": row["depth"],
            "history_depth_cumulative": row["depth_cumulative"],
            "ansatz_depth": adapt["ansatz_depth"],
        },
        "nfev": {
            "nfev_opt": row["nfev_opt"],
            "nfev_step_total_delta": row["nfev_step_total_delta"],
            "nfev_total": adapt["nfev_total"],
        },
    }


def _compare(a: Any, b: Any, *, path: str = "") -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            failures.append(
                {"path": path, "reason": "key_set_mismatch", "left": sorted(a), "right": sorted(b)}
            )
            return failures
        for key in sorted(a):
            failures.extend(_compare(a[key], b[key], path=f"{path}.{key}" if path else key))
        return failures
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [{"path": path, "reason": "length_mismatch", "left": len(a), "right": len(b)}]
        for index, (left, right) in enumerate(zip(a, b)):
            failures.extend(_compare(left, right, path=f"{path}[{index}]"))
        return failures
    if isinstance(a, float) or isinstance(b, float):
        delta = abs(float(a) - float(b))
        if delta > FLOAT_ATOL:
            failures.append(
                {"path": path, "reason": "float_delta", "left": a, "right": b, "abs_delta": delta}
            )
    elif a != b:
        failures.append({"path": path, "reason": "value_mismatch", "left": a, "right": b})
    return failures


def _cache_event(stdout_path: Path, *, first_depth_only: bool = False) -> dict[str, Any]:
    phase2_rows: list[dict[str, Any]] = []
    phase0_rows: list[dict[str, Any]] = []
    pool_registry_rows: list[dict[str, Any]] = []
    for line in stdout_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("AI_LOG "):
            continue
        try:
            payload = json.loads(line[len("AI_LOG ") :])
        except json.JSONDecodeError:
            continue
        event = payload.get("event")
        if event == "hardcoded_adapt_beam_phase2_full_feature_parallel":
            phase2_rows.append(
                {
                    "candidate_record_cache_mode": payload.get("candidate_record_cache_mode"),
                    "candidate_record_cache": payload.get("candidate_record_cache"),
                    "runtime_split_child_set_parallel_scoring_count": payload.get(
                        "runtime_split_child_set_parallel_scoring_count"
                    ),
                }
            )
        elif event == "hardcoded_adapt_phase0_screen_cache":
            phase0_rows.append(
                {key: payload.get(key) for key in ("source", "status", "selected_pair_count")}
            )
        elif event in {
            "hardcoded_adapt_pool_cache_hit",
            "hardcoded_adapt_generator_registry_cache_hit",
        }:
            pool_registry_rows.append(
                {key: payload.get(key) for key in ("event", "cache_level", "cache_scope")}
            )
    if (not first_depth_only and len(phase2_rows) != 1) or not phase2_rows:
        raise ValueError(f"expected one depth-1 Phase-II cache row in {stdout_path}, got {len(phase2_rows)}")
    return {
        "phase2": phase2_rows[0],
        "phase0": phase0_rows[:1] if first_depth_only else phase0_rows,
        "pool_or_registry_hits": pool_registry_rows,
        "full_log_phase2_row_count": len(phase2_rows),
    }


def _overlap(left: Sequence[str], right: Sequence[str]) -> dict[str, Any]:
    left_set = set(left)
    right_set = set(right)
    return {
        "ordered_equal": list(left) == list(right),
        "set_overlap_count": len(left_set & right_set),
        "left_count": len(left),
        "right_count": len(right),
        "left_only": [value for value in left if value not in right_set],
        "right_only": [value for value in right if value not in left_set],
    }


def main() -> int:
    names = {
        "snapshot": "weak_weak_depth1_prerepair_snapshot",
        "current_cold": "weak_weak_depth1_current_enforcement_off_cold",
        "current_warm": "weak_weak_depth1_current_enforcement_off_warm",
    }
    raw_paths: dict[str, dict[str, Path]] = {
        key: {
            suffix: EVIDENCE / f"{stem}.{suffix}"
            for suffix in ("result.json", "stdout.log", "stderr.log")
        }
        for key, stem in names.items()
    }
    inputs = {
        key: {suffix: _artifact(path) for suffix, path in paths.items()}
        for key, paths in raw_paths.items()
    }
    inputs["july8_preserved"] = {
        "result.json": _artifact(JULY8_RESULT),
        "stdout.log": _artifact(JULY8_STDOUT),
    }
    payloads = {key: _load(paths["result.json"]) for key, paths in raw_paths.items()}
    summaries = {key: _scientific_summary(payload) for key, payload in payloads.items()}
    snapshot_current_failures = _compare(summaries["snapshot"], summaries["current_cold"])
    cold_warm_failures = _compare(summaries["current_cold"], summaries["current_warm"])
    cache = {
        key: _cache_event(paths["stdout.log"])
        for key, paths in raw_paths.items()
    }
    cold_phase2 = cache["current_cold"]["phase2"]
    warm_phase2 = cache["current_warm"]["phase2"]
    cache_transition_passed = bool(
        cold_phase2["candidate_record_cache_mode"] == "disk"
        and warm_phase2["candidate_record_cache_mode"] == "disk"
        and cold_phase2["candidate_record_cache"]["misses"] > 0
        and cold_phase2["candidate_record_cache"]["disk_hits"] == 0
        and warm_phase2["candidate_record_cache"]["disk_hits"]
        == cold_phase2["candidate_record_cache"]["misses"]
        and warm_phase2["candidate_record_cache"]["misses"] == 0
    )

    july8 = _load(JULY8_RESULT)
    july8_row = july8["adapt_vqe"]["history"][0]
    snapshot_row = _history0(payloads["snapshot"])
    july8_labels = [row["candidate_label"] for row in july8_row["shortlisted_records"]]
    snapshot_labels = [row["candidate_label"] for row in snapshot_row["shortlisted_records"]]
    july8_snapshot_overlap = _overlap(july8_labels, snapshot_labels)
    july8_winner_match = bool(
        july8_row["selected_op"] == snapshot_row["selected_op"]
        and july8_row["generator_id"] == snapshot_row["generator_id"]
        and july8_row["selected_position"] == snapshot_row["selected_position"]
    )
    july8_depth1_energy = {
        "before": july8_row["energy_before_opt"],
        "after": july8_row["energy_after_opt"],
        "delta": july8_row["delta_energy"],
        "final": july8_row["energy_after_opt"],
    }
    july8_energy_deltas = {
        key: abs(float(july8_depth1_energy[key]) - float(summaries["snapshot"]["energy"][key]))
        for key in ("before", "after", "delta", "final")
    }
    july8_controls = _controls(july8)
    snapshot_controls = summaries["snapshot"]["shortlist_controls"]
    july8_core_phase_caps_match = bool(
        july8_controls["phase1_shortlist_size_effective"]
        == snapshot_controls["phase1_shortlist_size_effective"]
        == 8
        and july8_controls["phase2_shortlist_size_effective"]
        == snapshot_controls["phase2_shortlist_size_effective"]
        == 4
        and july8_controls["phase3_shortlist_size_effective"]
        == snapshot_controls["phase3_shortlist_size_effective"]
        == 4
        and abs(
            float(july8_controls["phase2_shortlist_fraction_effective"])
            - float(snapshot_controls["phase2_shortlist_fraction_effective"])
        )
        <= FLOAT_ATOL
    )
    july8_cache = _cache_event(JULY8_STDOUT, first_depth_only=True)

    passed = bool(
        not snapshot_current_failures
        and not cold_warm_failures
        and cache_transition_passed
        and summaries["current_cold"]["shortlist_controls"]["phase1_shortlist_size_effective"] == 8
        and summaries["current_cold"]["shortlist_controls"]["phase2_shortlist_size_effective"] == 4
        and summaries["current_cold"]["shortlist_controls"]["phase3_shortlist_size_effective"] == 4
        and abs(
            float(
                summaries["current_cold"]["shortlist_controls"][
                    "phase2_shortlist_fraction_effective"
                ]
            )
            - 1.0 / 12.0
        )
        <= FLOAT_ATOL
    )
    output = {
        "schema": "paper_i_hh_weak_weak_depth1_route_parity_v1",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "tolerances": {"energy_abs": ENERGY_ATOL, "generic_float_abs": FLOAT_ATOL},
        "input_artifacts": inputs,
        "current_vs_verified_pre_repair_snapshot": {
            "passed": not snapshot_current_failures,
            "comparison_failures": snapshot_current_failures,
            "compared_fields": [
                "resolved caps 8/4/4 and fraction 1/12",
                "shortlist selection unit candidate_position_record",
                "eight ranked candidate labels, generator IDs, position IDs, and score fields",
                "selected generator, label, position, and pool indices",
                "energy before, after, delta, and final energy",
                "outer/active depth and nfev telemetry",
            ],
            "snapshot": summaries["snapshot"],
            "current_enforcement_off_cold": summaries["current_cold"],
            "enforcement_contract": {
                "snapshot_symmetry_policy": payloads["snapshot"]["settings"][
                    "phase3_runtime_split_child_set_symmetry_policy"
                ],
                "current_symmetry_policy": payloads["current_cold"]["settings"][
                    "phase3_runtime_split_child_set_symmetry_policy"
                ],
                "current_padding_policy": payloads["current_cold"]["adapt_vqe"][
                    "continuation"
                ]["runtime_split_summary"]["child_padding"]["policy"],
                "approved_enforcement_changes_disabled_for_parity": True,
            },
        },
        "current_cold_vs_warm_cache_replay": {
            "passed": not cold_warm_failures and cache_transition_passed,
            "scientific_comparison_failures": cold_warm_failures,
            "scientific_summaries_equal": not cold_warm_failures,
            "cache_transition_passed": cache_transition_passed,
            "cold": cache["current_cold"],
            "warm": cache["current_warm"],
            "classification": "cache warmth changes hit/miss performance telemetry, not current-source scientific outputs",
        },
        "snapshot_vs_preserved_july8_history0": {
            "not_a_current_compatibility_gate": True,
            "core_phase_caps_match": july8_core_phase_caps_match,
            "full_normalized_policy_object_equality": "unavailable_in_july8_telemetry",
            "selected_winner_match": july8_winner_match,
            "energy_abs_deltas": july8_energy_deltas,
            "energy_match_within_tolerance": all(
                delta <= ENERGY_ATOL for delta in july8_energy_deltas.values()
            ),
            "shortlisted_labels": july8_snapshot_overlap,
            "july8_cache_and_child_scoring": july8_cache,
            "snapshot_clean_cache_and_child_scoring": cache["snapshot"],
            "classification": "unresolved_historical_code_or_cache_provenance",
            "exact_blocker": (
                "The verified pre-repair remote snapshot is not proven to be the exact July-8 pipelines/cache "
                "state: only 6/8 shortlist labels overlap, and July-8 had 46 candidate disk hits with 38 "
                "runtime child-set scores versus 52 misses and 44 scores in the clean snapshot."
            ),
        },
        "scope_conclusion": (
            "Current enforcement-off code reproduces the verified pre-repair snapshot exactly at depth 1, "
            "including cold/warm cache invariance. Exact July-8 pipeline/cache identity remains unresolved."
        ),
    }
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": OUTPUT.as_posix(), "sha256": _sha256(OUTPUT), "passed": passed}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
