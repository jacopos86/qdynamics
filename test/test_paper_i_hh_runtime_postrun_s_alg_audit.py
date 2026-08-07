from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from pipelines.reporting.audit_paper_i_hh_runtime_postrun_s_alg import (
    DEFAULT_ROUTE_ID,
    build_audit,
)
from pipelines.exact_bench.paper_i_s_alg_accounting import (
    PAPER_I_S_ALG_ACCOUNTING_SCHEMA,
    PAPER_I_S_ALG_CONTRACT,
    SNAKE_REPRESENTATION_PROJECTED_SINGLETON,
)
from pipelines.static_adapt.estimator_call_ledger import (
    summarize_estimator_occurrence_prefix,
)


COMPONENTS = {
    "N_H_outer": 1,
    "N_H_refit": 1,
    "N_grad": 0,
    "N_metric": 1,
}
RAW_COMPONENTS = {
    "N_H_outer": 1,
    "N_H_refit": 1,
    "N_grad": 0,
    "N_metric": 2,
}
OCCURRENCES = [
    {
        "sequence": 1,
        "primitive_id": "energy",
        "component": "N_H_outer",
        "consumer_scope": "outer",
        "branch_id": None,
        "charged": True,
    },
    {
        "sequence": 2,
        "primitive_id": "metric",
        "component": "N_metric",
        "consumer_scope": "phase3",
        "branch_id": None,
        "charged": True,
    },
    {
        "sequence": 3,
        "primitive_id": "metric",
        "component": "N_metric",
        "consumer_scope": "accepted_refit",
        "branch_id": None,
        "charged": False,
    },
    {
        "sequence": 4,
        "primitive_id": "refit",
        "component": "N_H_refit",
        "consumer_scope": "energy:depth_opt",
        "branch_id": None,
        "charged": True,
    },
]
AUDIT_COMPONENTS = {
    "N_H_outer": 1,
    "N_H_refit": 1,
    "N_grad": 2,
    "N_metric": 4,
}
CLEAN_AUDIT_COMPONENTS = {
    "N_H_outer": 1,
    "N_H_refit": 1,
    "N_grad": 2,
    "N_metric": 3,
}
AUDIT_OCCURRENCES = [
    {
        "sequence": sequence,
        "primitive_id": f"{component}_{sequence}",
        "component": component,
        "consumer_scope": f"audit:{component}",
        "branch_id": None,
        "charged": True,
    }
    for sequence, component in enumerate(
        [
            "N_H_outer",
            "N_H_refit",
            "N_grad",
            "N_grad",
            "N_metric",
            "N_metric",
            "N_metric",
            "N_metric",
        ],
        start=1,
    )
]


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()


def _add_json(archive: tarfile.TarFile, name: str, payload: object) -> None:
    encoded = _json_bytes(payload)
    info = tarfile.TarInfo(name)
    info.size = len(encoded)
    archive.addfile(info, io.BytesIO(encoded))


def test_occurrence_prefix_reconstruction_deduplicates_cross_scope_reuse():
    summary = summarize_estimator_occurrence_prefix(
        iter(OCCURRENCES),
        occurrence_sequence_end_inclusive=4,
    )
    assert summary["cumulative_raw_occurrences"] == {
        "components": RAW_COMPONENTS,
        "total": 4,
    }
    assert summary["cumulative_executed_queries"] == {
        "components": RAW_COMPONENTS,
        "S_alg": 4,
        "unit": "executed_logical_scalar_estimator_invocation",
    }
    assert summary["cumulative_unique_primitives"] == {
        "components": COMPONENTS,
        "S_unique": 3,
    }


def test_occurrence_prefix_reconstruction_fails_on_charged_flag_drift():
    occurrences = [dict(row) for row in OCCURRENCES]
    occurrences[2]["charged"] = True
    with pytest.raises(ValueError, match="charged flag"):
        summarize_estimator_occurrence_prefix(
            iter(occurrences),
            occurrence_sequence_end_inclusive=4,
        )


def test_six_row_runtime_postrun_audit_closes_synthetic_archives(tmp_path: Path):
    regimes = (
        "weak_weak",
        "intermediate_weak",
        "strong_weak_u8",
        "weak_strong",
        "intermediate_strong",
        "strong_strong_u8",
    )
    controller = {
        "by_phase": {
            "phase1": {
                "method_input_candidate_count_total": 1,
                "actual_operator_probe_count_total": 1,
                "method_shortlist_candidate_count_total": 1,
            },
            "phase2": {"events_count": 1},
            "phase3": {"method_input_candidate_count_total": 1},
        },
        "by_scope": {
            "test|phase=phase2|event=phase2_rerank_records|depth=1": {
                "method_input_candidate_count_total": 1,
                "actual_evaluated_candidate_count_total": 1,
                "candidate_count_total": 1,
                "pre_shortlist_count_total": 1,
                "records_evaluated": 1,
                "shortlist_size_total": 1,
                "retained_count_total": 1,
            },
            (
                "test|phase=phase3|"
                "event=phase3_reduced_geometry_rerank|depth=1"
            ): {
                "method_input_candidate_count_total": 1,
                "actual_evaluated_candidate_count_total": 1,
                "candidate_count_total": 1,
                "pre_shortlist_count_total": 1,
                "records_evaluated": 1,
            },
        },
    }
    synthetic_history_row = {
        "phase3_active_logical_coordinate_count": 0,
        "nfev_opt": 1,
        "controller_measurement_work_proxy": controller,
        "scored_surface_size": 1,
        "scored_surface_records": [
            {
                "runtime_split_mode": "off",
                "runtime_split_chosen_representation": "parent",
                "runtime_split_child_count": None,
            }
        ],
        "projected_phase3_population_receipt": {
            "schema": "paper_i_projected_phase3_population_receipt_v2",
            "phase_order": (
                "phase1_parent_shortlist_then_split_then_"
                "phase2_children_then_phase3"
            ),
            "phase1_retained_parent_count": 1,
            "phase2_input_child_count": 1,
            "phase2_retained_child_count": 1,
            "split_parent_count": 0,
            "split_child_count": 0,
            "split_children_by_parent": [],
            "unsplit_singleton_count": 1,
            "phase3_evaluated_candidate_count": 1,
            "child_primitive_reuse_count": 0,
            "cross_outer_iteration_reuse_count": 0,
        },
    }
    clean_receipt = {
        "schema": PAPER_I_S_ALG_ACCOUNTING_SCHEMA,
        "contract": PAPER_I_S_ALG_CONTRACT,
        "method": "SNAKE",
        "representation": SNAKE_REPRESENTATION_PROJECTED_SINGLETON,
        "accepted_prefix_length": 1,
        "scope": (
            "all required estimator invocations through the displayed "
            "post-admission prefix; post-prefix diagnostics excluded"
        ),
        "unit": "logical_scalar_estimator_invocation",
        "components": dict(CLEAN_AUDIT_COMPONENTS),
        "S_alg": sum(CLEAN_AUDIT_COMPONENTS.values()),
        "normalization": {
            "fixture": "manually specified independent formula oracle"
        },
        "round_cardinalities": [
            {
                "history_index": 0,
                "n_active": 0,
                "R1": 1,
                "R2": 1,
                "phase1_retained_parent_count": 1,
                "R3": 1,
            }
        ],
    }
    archive_path = tmp_path / "six.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for regime in regimes:
            root = f"raw_outputs/test/{regime}/json"
            receipt = {
                "schema": "paper_i_active_prefix_estimator_ledger_receipt_v1",
                "status": "complete",
                "canonical_same_state_deduplication_active": True,
                "raw_occurrences_preserved": True,
                "outer_iteration": 1,
                "occurrence_sequence_end_inclusive": 8,
                "cumulative_raw_occurrences": {
                    "components": AUDIT_COMPONENTS,
                    "total": 8,
                },
                "cumulative_unique_primitives": {
                    "components": AUDIT_COMPONENTS,
                    "S_alg": 8,
                },
            }
            _add_json(
                archive,
                f"{root}/result.json",
                {
                    "history": [
                        {
                            **synthetic_history_row,
                            "active_prefix_checkpoint": {
                                "estimator_ledger_receipt": receipt
                            }
                        }
                    ]
                },
            )
            _add_json(
                archive,
                f"{root}/estimator_call_ledger.json",
                {"ledger": {"occurrences": AUDIT_OCCURRENCES}},
            )

    archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    rows = []
    for regime in regimes:
        root = f"raw_outputs/test/{regime}/json"
        rows.append(
            {
                "route_id": DEFAULT_ROUTE_ID,
                "regime": regime,
                "history_position": 1,
                "k_target": 1,
                "outer_iteration": 1,
                "S_alg": clean_receipt["S_alg"],
                "S_alg_components": clean_receipt["components"],
                "S_alg_receipt": clean_receipt,
                "source": {
                    "path": str(archive_path),
                    "sha256": archive_sha256,
                    "result_member": f"{root}/result.json",
                },
            }
        )
    target_json = tmp_path / "targets.json"
    target_json.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    output_json = tmp_path / "audit.json"

    result = build_audit(
        target_json=target_json,
        output_json=output_json,
        route_id=DEFAULT_ROUTE_ID,
    )

    assert result["status"] == "pass"
    assert result["summary"]["row_count"] == 6
    assert all(row["S_alg"] == 7 for row in result["rows"])
    assert all(
        row["runtime_unique_primitive_count"] == 8 for row in result["rows"]
    )
    assert json.loads(output_json.read_text())["status"] == "pass"
