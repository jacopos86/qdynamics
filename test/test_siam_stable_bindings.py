"""Stable regular-file bindings for the live SIAM Nb4 evidence chain."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import pipelines.time_dynamics.siam_stable_bindings as stable_bindings
from pipelines.time_dynamics.siam_stable_bindings import (
    DEFAULT_REPORT_CONTRACT_NAME,
    DEFAULT_SCIENTIFIC_STATE_NAME,
    SIAM_STABLE_REPORT_CONTRACT_V1,
    SIAM_STABLE_SCIENTIFIC_STATE_V1,
    SIAMStableBindingError,
    _promotion_slot_from_completion_arm,
    _validate_ladder_and_frontier,
    _validated_result_evidence,
    build_siam_stable_binding_documents,
    validate_siam_results_pdf_binding,
    write_siam_stable_binding_documents,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_EPOCH_RECEIPT = Path(
    "output/siam_screening/nb4_pilot_v1/source_epochs/"
    "siam-nb4-source-ba50339c15ffe1aa6507/planner_receipt.json"
)
LADDER_PLAN = Path(
    "output/siam_screening/nb4_pilot_v1/plans/"
    "source_ba50339c_frontier_9713113/ladder_plan.json"
)
PREPARED_FRONTIER_RECEIPT = Path(
    "output/siam_screening/nb4_pilot_v1/plans/"
    "source_ba50339c_frontier_9713113/prepared_frontier_receipt.json"
)
RESULT_RECEIPT = Path(
    "output/siam_screening/nb4_pilot_v1/attempts/cluster_9713113/"
    "increments/proc0/evidence/increment_receipt.json"
)
PROMOTION = Path(
    "output/siam_screening/nb4_pilot_v1/results/cluster_9713113/promotion.json"
)
REPORT_RECEIPT = Path(
    "output/pdf/siam_nb4_screening_benchmark/"
    "siam_nb4_screening_results_cluster9713113_receipt.json"
)
REPORT_PDF = Path(
    "output/pdf/siam_nb4_screening_benchmark/"
    "siam_nb4_screening_results_cluster9713113.pdf"
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json_binding(root: Path, name: str, value: object) -> dict:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(content)
    binding = {
        "path": path.relative_to(root).as_posix(),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    if isinstance(value, dict) and isinstance(value.get("schema"), str):
        binding["schema"] = value["schema"]
    return binding


def _write_bytes_binding(root: Path, name: str, content: bytes) -> dict:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _build() -> tuple[dict, dict]:
    return build_siam_stable_binding_documents(
        repository_root=REPOSITORY_ROOT,
        source_epoch_receipt=SOURCE_EPOCH_RECEIPT,
        ladder_plan=LADDER_PLAN,
        prepared_frontier_receipt=PREPARED_FRONTIER_RECEIPT,
        result_receipt=RESULT_RECEIPT,
    )


@pytest.mark.parametrize("status", ("IDLE", "CLOSED"))
def test_zero_action_frontier_accepts_only_digest_bound_idle_or_closed(
    tmp_path: Path,
    status: str,
) -> None:
    epoch = {
        "schema": "siam_source_epoch_contract_v1",
        "source_epoch_id": "siam-nb4-source-" + "a" * 64,
        "source_revision": "b" * 40,
        "source_tree_files_sha256": "c" * 64,
    }
    ladder = {
        "schema": "siam_nb4_comparator_first_ladder_plan_v1",
        "source_epoch_id": epoch["source_epoch_id"],
        "source_revision": epoch["source_revision"],
        "next_actions": [],
    }
    ladder_binding = _write_json_binding(tmp_path, "ladder.json", ladder)
    frontier = {
        "schema": "siam_nb4_prepared_frontier_v1",
        "source_epoch_id": epoch["source_epoch_id"],
        "source_revision": epoch["source_revision"],
        "plan": ladder_binding["path"],
        "plan_sha256": ladder_binding["sha256"],
        "qiskit_compilation_requested": False,
        "status": status,
        "prepared_actions": [],
        "submissions": [],
    }
    _validate_ladder_and_frontier(
        repository_root=tmp_path,
        epoch=epoch,
        ladder=ladder,
        ladder_binding=ladder_binding,
        frontier=frontier,
    )

    with pytest.raises(SIAMStableBindingError, match="not bound to the ladder"):
        _validate_ladder_and_frontier(
            repository_root=tmp_path,
            epoch=epoch,
            ladder=ladder,
            ladder_binding=ladder_binding,
            frontier={**frontier, "status": "SUBMITTED"},
        )


def test_promotion_slot_is_selected_from_canonical_completion_arm() -> None:
    assert _promotion_slot_from_completion_arm("avqds") == "avqds"
    assert _promotion_slot_from_completion_arm("exchange") == "full_apm"
    with pytest.raises(SIAMStableBindingError, match="unsupported SIAM completion"):
        _promotion_slot_from_completion_arm("full_apm")


def test_continuation_binds_raw_and_stitched_evidence_separately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    epoch = {
        "schema": "siam_source_epoch_contract_v1",
        "source_epoch_id": "siam-nb4-source-" + "d" * 64,
        "source_revision": "e" * 40,
        "source_tree_files_sha256": "f" * 64,
    }
    action_id = "siam-nb4-avqds-continue-r00-cut1em03"
    seed_sha256 = "1" * 64
    raw_completion = {
        "schema": stable_bindings.SIAM_NB4_CHTC_RECEIPT_SCHEMA_V4,
        "source_epoch": epoch,
        "source_revision": epoch["source_revision"],
        "source_tree_files_sha256": epoch["source_tree_files_sha256"],
        "seed_sha256": seed_sha256,
        "action_id": action_id,
        "arm": "avqds",
    }
    screen_completion = {
        "source_epoch": epoch,
        "source_revision": epoch["source_revision"],
        "source_tree_files_sha256": epoch["source_tree_files_sha256"],
        "seed_sha256": seed_sha256,
        "action_id": "siam-nb4-avqds-screen-r00-cut1em03",
        "arm": "avqds",
    }
    raw_completion_binding = _write_json_binding(
        tmp_path, "raw_completion.json", raw_completion
    )
    raw_trajectory_binding = _write_bytes_binding(
        tmp_path, "raw_continuation.json", b'{"raw":true}\n'
    )
    stitch_completion = {
        "schema": stable_bindings.SIAM_NB4_STITCH_COMPLETION_V1,
        "arm": "avqds",
    }
    stitch_completion_binding = _write_json_binding(
        tmp_path, "stitch_completion.json", stitch_completion
    )
    stitched_trajectory_binding = _write_bytes_binding(
        tmp_path, "stitched.json", b'{"stitched":true}\n'
    )
    score_binding = _write_json_binding(tmp_path, "score.json", {"score": True})

    def validate_raw(receipt: dict, **expected: object) -> dict:
        assert receipt == raw_completion
        assert expected["expected_trajectory_sha256"] == raw_trajectory_binding[
            "sha256"
        ]
        assert expected["expected_segment"] == "continue_t1_t5_401"
        return {
            "action_id": action_id,
            "arm": "avqds",
            "seed_sha256": seed_sha256,
        }

    def validate_stitch(receipt: dict, **expected: object) -> dict:
        assert receipt == stitch_completion
        assert expected["expected_trajectory_sha256"] == stitched_trajectory_binding[
            "sha256"
        ]
        return {
            "source_segments": [
                {
                    "segment": "screen_t0_t1_101",
                    "source_revision": epoch["source_revision"],
                    "completion_receipt": screen_completion,
                },
                {
                    "segment": "continue_t1_t5_401",
                    "source_revision": epoch["source_revision"],
                    "completion_receipt": raw_completion,
                },
            ]
        }

    monkeypatch.setattr(
        stable_bindings, "validate_siam_nb4_completion_receipt", validate_raw
    )
    monkeypatch.setattr(
        stable_bindings,
        "validate_siam_nb4_stitch_completion_receipt",
        validate_stitch,
    )
    monkeypatch.setattr(
        stable_bindings,
        "_verified_score",
        lambda _score: {
            "method": {
                "code_revision": epoch["source_revision"],
                "structural_policy": "avqds",
            },
            "source": {
                "seed_sha256": seed_sha256,
                "run_sha256": stitched_trajectory_binding["sha256"],
            },
        },
    )
    evidence = _validated_result_evidence(
        repository_root=tmp_path,
        epoch=epoch,
        result={
            "action_id": action_id,
            "source_completion_receipt": raw_completion_binding,
            "raw_continuation": raw_trajectory_binding,
            "stitch": {
                "completion_receipt": stitch_completion_binding,
                "trajectory": stitched_trajectory_binding,
            },
            "score": score_binding,
        },
    )

    assert evidence["mode"] == "continuation"
    assert evidence["promotion_slot"] == "avqds"
    assert evidence["raw_completion_binding"] == raw_completion_binding
    assert evidence["raw_trajectory_binding"] == raw_trajectory_binding
    assert evidence["promotion_completion_binding"] == stitch_completion_binding
    assert evidence["scored_trajectory_binding"] == stitched_trajectory_binding
    assert evidence["extra_bindings"] == {
        "stitched_completion_receipt": stitch_completion_binding,
        "stitched_trajectory": stitched_trajectory_binding,
    }


def test_current_epoch_stable_documents_are_deterministic_and_self_bound() -> None:
    first_scientific, first_report = _build()
    second_scientific, second_report = _build()
    assert (first_scientific, first_report) == (second_scientific, second_report)

    scientific_digest = first_scientific.pop("scientific_state_sha256")
    report_digest = first_report.pop("report_contract_sha256")
    assert first_scientific["schema"] == SIAM_STABLE_SCIENTIFIC_STATE_V1
    assert first_report["schema"] == SIAM_STABLE_REPORT_CONTRACT_V1
    assert scientific_digest == _canonical_sha256(first_scientific)
    assert report_digest == _canonical_sha256(first_report)
    assert first_scientific["source_epoch"] == first_report["source_epoch"]
    assert first_scientific["bindings"] == first_report["bindings"]

    bindings = first_scientific["bindings"]
    for binding in bindings.values():
        path = REPOSITORY_ROOT / binding["path"]
        content = path.read_bytes()
        assert path.is_file()
        assert binding["bytes"] == len(content)
        assert binding["sha256"] == hashlib.sha256(content).hexdigest()


def test_current_epoch_excludes_only_the_two_stale_result_backlinks() -> None:
    scientific, report = _build()
    exclusions = scientific["provenance_policy"][
        "historical_mutable_backlink_exclusions"
    ]
    assert exclusions == report["provenance_policy"][
        "historical_mutable_backlink_exclusions"
    ]
    assert len(exclusions) == 2
    assert {
        Path(row["artifact"]["path"]).name for row in exclusions
    } == {
        "incremental_refresh_blocked_supersession_receipt.json",
        "siam_nb4_screen_gate_pending_cluster9713113_supersession_receipt.json",
    }
    current = scientific["bindings"]["result_receipt"]["sha256"]
    for row in exclusions:
        assert row["json_pointer"] == "/increment_receipt/sha256"
        assert row["declared_sha256"] != current
        assert row["current_sha256"] == current


def test_stable_current_files_are_idempotent_regular_files(tmp_path: Path) -> None:
    scientific, report = _build()
    scientific_path, report_path = write_siam_stable_binding_documents(
        output_directory=tmp_path,
        scientific_state=scientific,
        report_contract=report,
    )
    first_bytes = (scientific_path.read_bytes(), report_path.read_bytes())
    first_mtimes = (
        scientific_path.stat().st_mtime_ns,
        report_path.stat().st_mtime_ns,
    )
    repeated_paths = write_siam_stable_binding_documents(
        output_directory=tmp_path,
        scientific_state=scientific,
        report_contract=report,
    )
    assert repeated_paths == (scientific_path, report_path)
    assert first_bytes == (scientific_path.read_bytes(), report_path.read_bytes())
    assert first_mtimes == (
        scientific_path.stat().st_mtime_ns,
        report_path.stat().st_mtime_ns,
    )
    assert scientific_path.name == DEFAULT_SCIENTIFIC_STATE_NAME
    assert report_path.name == DEFAULT_REPORT_CONTRACT_NAME
    assert scientific_path.is_file() and report_path.is_file()


def test_report_chain_fails_closed_on_pdf_or_receipt_tampering() -> None:
    promotion = json.loads((REPOSITORY_ROOT / PROMOTION).read_bytes())
    receipt = json.loads((REPOSITORY_ROOT / REPORT_RECEIPT).read_bytes())
    pdf_bytes = (REPOSITORY_ROOT / REPORT_PDF).read_bytes()
    validate_siam_results_pdf_binding(promotion, receipt, pdf_bytes)

    changed_pdf = pdf_bytes[:-1] + bytes([pdf_bytes[-1] ^ 1])
    with pytest.raises(SIAMStableBindingError, match="chain mismatch"):
        validate_siam_results_pdf_binding(promotion, receipt, changed_pdf)

    changed_receipt = dict(receipt)
    changed_receipt["pdf"] = {**receipt["pdf"], "bytes": len(pdf_bytes) + 1}
    with pytest.raises(SIAMStableBindingError, match="self-digest mismatch"):
        validate_siam_results_pdf_binding(
            promotion, changed_receipt, pdf_bytes
        )
