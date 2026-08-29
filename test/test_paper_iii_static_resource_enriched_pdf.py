from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.reporting import build_paper_iii_static_resource_enriched_pdf as report


HH_REGIMES = (
    "weak_weak",
    "intermediate_weak",
    "strong_weak_u8",
    "weak_strong",
    "intermediate_strong",
    "strong_strong_u8",
)
BLOCK_CASE_IDS = tuple(
    [f"hh::{regime}::nph{nph}" for regime in HH_REGIMES for nph in (1, 3)]
    + [
        "peierls_weak_weak",
        "peierls_weak_strong",
        "peierls_u8_weak",
        "siam_nb4_pilot",
    ]
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _block_payload() -> dict[str, object]:
    cases: dict[str, object] = {}
    for position, case_id in enumerate(BLOCK_CASE_IDS):
        reached = position < 11
        numerical = {
            "k": 12,
            "retained_rank": 8,
            "K": 20,
            "max_root_abs_error": 1.0e-8 if reached else 1.0e-1,
            "qiskit_compiled": {"n2q": 30, "d2q": 15},
            "estimator": {"queries": 40},
        }
        weights = [1.0] * 7
        missing: list[int] = []
        if not reached:
            weights[0] = 0.0
            missing = [1]
        cases[case_id] = {
            "method_id": report.METHOD_BLOCK_KRYLOV,
            "policy_id": "block_seeded_realtime_krylov_lowest_diagonal_b6_dt0p5_v1",
            "target_root_count": 6,
            "threshold_crossings": {
                "1e-06": {
                    "status": "REACHED" if reached else "NOT_REACHED_WITHIN_CAP",
                    **(numerical if reached else {}),
                }
            },
            "terminal": numerical,
            "post_run_source_coverage_audit": {
                "post_run_only": True,
                "used_by_online_route": False,
                "target_root_indices_below_zero_weight_floor": missing,
                "source_block_overlap_weight_by_root": weights,
                "zero_weight_floor": 1.0e-20,
            },
        }
    return {
        "schema_version": report.EXPECTED_BLOCK_KRYLOV_SCHEMA,
        "route_contract": {
            "method_id": report.METHOD_BLOCK_KRYLOV,
            "policy_id": "block_seeded_realtime_krylov_lowest_diagonal_b6_dt0p5_v1",
            "target_root_count": 6,
            "block_size": 6,
            "root_matching": "minimum_total_abs_energy_hungarian_one_to_one",
            "online_exact_reference_use": False,
            "same_policy_across_model_families": True,
        },
        "cases": cases,
    }


def _seal_block_fixture(tmp_path: Path, payload: dict[str, object]) -> Path:
    matrix_path = tmp_path / "matrix.json"
    _write_json(matrix_path, payload)
    _write_json(
        tmp_path / "receipt.json",
        {
            "schema_version": report.EXPECTED_BLOCK_KRYLOV_RECEIPT_SCHEMA,
            "sha256": report._sha256(matrix_path),
            "case_count": 16,
        },
    )
    return matrix_path


def test_block_krylov_validator_locks_source_coverage(tmp_path: Path) -> None:
    payload = _block_payload()
    matrix_path = _seal_block_fixture(tmp_path, payload)
    assert report._validate_block_krylov_matrix(
        payload, input_path=matrix_path
    ) == tmp_path / "receipt.json"

    invalid = deepcopy(payload)
    failed_case = BLOCK_CASE_IDS[11]
    invalid["cases"][failed_case]["post_run_source_coverage_audit"][
        "target_root_indices_below_zero_weight_floor"
    ] = [999]
    with pytest.raises(report.ReportBuildError, match="invalid target root"):
        report._validate_block_krylov_matrix(invalid, input_path=matrix_path)


def _pool_payload() -> dict[str, object]:
    pool_ids = (
        "hh_full_meta_macro_v1",
        "hh_full_meta_plus_encoded_boson_pauli_completion_v1",
    )
    rows = []
    for regime in HH_REGIMES:
        for nph in (1, 3):
            case_key = f"{regime}_nph{nph}"
            rows.append(
                {
                    "case": {"case_key": case_key},
                    "selected_qse_by_pool_policy": {
                        pool_id: {
                            "method_id": report.METHOD_SELECTED,
                            "pool_policy_id": pool_id,
                            "crossings": [],
                        }
                        for pool_id in pool_ids
                    },
                    "direct_davidson": {"method_id": report.METHOD_DAVIDSON},
                }
            )
    return {
        "schema_version": report.EXPECTED_POOL_ABLATION_SCHEMA,
        "selector_policy_id": "full_pencil_ky_fan_cost_ratio_novelty_floor_no_reward_v1",
        "pool_policy_ids": list(pool_ids),
        "rows": rows,
        "controller_boundary": {
            "implicit_fallback_enabled": False,
            "exact_audit_feeds_online_decisions": False,
        },
        "lock_gate": {
            "profiles": {
                pool_ids[0]: {"target_reached_case_count": 10},
                pool_ids[1]: {
                    "target_reached_case_count": 12,
                    "coverage_qualified": True,
                },
            }
        },
    }


def test_pool_validator_requires_twelve_distinct_typed_cases(tmp_path: Path) -> None:
    payload = _pool_payload()
    input_path = tmp_path / "pool_profile_ablation.json"
    _write_json(input_path, payload)
    _write_json(
        tmp_path / "status.json",
        {
            "schema_version": "paper_iii_hh_pool_profile_ablation_status_v1",
            "execution_status": "COMPLETE",
            "case_count": 12,
        },
    )
    assert report._validate_pool_ablation(
        payload, input_path=input_path
    ) == tmp_path / "status.json"

    invalid = deepcopy(payload)
    invalid["rows"][-1]["case"]["case_key"] = invalid["rows"][0]["case"][
        "case_key"
    ]
    with pytest.raises(report.ReportBuildError, match="case set changed"):
        report._validate_pool_ablation(invalid, input_path=input_path)


def test_first_crossing_uses_declared_sequence_not_minimum_cost() -> None:
    route = {
        "online_receipt": {
            "rows": [
                {"k": 8, "retained_rank": 6, "n2q": 50},
                {"k": 10, "retained_rank": 6, "n2q": 20},
                {"k": 12, "retained_rank": 6, "n2q": 10},
            ]
        },
        "post_run_exact_audit": {
            "rungs": [
                {"k": 12, "max_root_abs_error": 1.0e-8},
                {
                    "k": 10,
                    "max_root_abs_error": 8.0e-7,
                    "per_root_abs_error": [8.0e-7] * 6,
                },
                {"k": 8, "max_root_abs_error": 2.0e-5},
            ]
        },
    }

    derived, receipt = report._first_reached_online_rung(route, target=1.0e-6)

    assert derived["k"] == 10
    assert derived["n2q"] == 20
    assert derived["max_root_abs_error"] == 8.0e-7
    assert receipt["sequence_index"] == 1
    assert receipt["audit"]["k"] == 10
    assert "max_root_abs_error" not in route["online_receipt"]["rows"][1]
