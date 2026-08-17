from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.reporting import build_paper_i_ra_phase3_cost_defect_affected_map as builder


@pytest.fixture(scope="module")
def affected_map() -> dict[str, object]:
    return builder.build_map()


def _records(payload: dict[str, object]) -> list[dict[str, object]]:
    return [
        record
        for group in payload["groups"]  # type: ignore[index]
        for record in group["records"]
    ]


def test_inventory_exhausts_zero_centered_protocols_and_required_groups(
    affected_map: dict[str, object],
) -> None:
    records = _records(affected_map)
    mapped_paths = {record["protocol"]["path"] for record in records}  # type: ignore[index]

    independently_discovered: set[str] = set()
    campaign_root = builder.REPO_ROOT / builder.CAMPAIGN_ROOT_RELATIVE
    for container_id in builder.AFFECTED_CONTAINER_IDS:
        for path in (campaign_root / container_id).rglob("*.json"):
            if (
                path.parent.name != "protocols"
                or path.stat().st_size > 8 * 1024 * 1024
            ):
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            mode = (
                payload.get("route_contract", {})
                .get("execution_settings", {})
                .get("phase3_hardware_cost_normalization_mode")
            )
            if mode == builder.SIGNED_MODE:
                independently_discovered.add(
                    path.resolve().relative_to(builder.REPO_ROOT.resolve()).as_posix()
                )

    assert mapped_paths == independently_discovered
    assert len(records) == affected_map["summary"]["protocol_artifact_count"]  # type: ignore[index]
    assert len(records) == 138
    assert affected_map["summary"]["distinct_protocol_digest_count"] == 132  # type: ignore[index]
    assert affected_map["summary"]["container_count"] == len(  # type: ignore[index]
        builder.AFFECTED_CONTAINER_IDS
    )

    groups = {group["group_id"]: group for group in affected_map["groups"]}  # type: ignore[index]
    assert set(groups) == builder.REQUIRED_GROUPS
    assert set(groups["page12_global_singleton"]["policy_counts"]) >= {
        "plateau",
        "always_open",
        "ra_append_only",
    }
    assert set(groups["page16_intact_macro"]["policy_counts"]) >= {
        "plateau",
        "always_open",
        "ra_append_only",
    }
    beam_records = groups["page16_beam3x2_metric_prune"]["records"]
    assert beam_records
    assert {record["route_axes"]["beam_kind"] for record in beam_records} == {
        "fork_local"
    }
    assert {record["route_axes"]["pruning_kind"] for record in beam_records} == {
        "metric"
    }


def test_source_and_execution_proof_are_fail_closed_and_trajectory_is_separate(
    affected_map: dict[str, object],
) -> None:
    records = _records(affected_map)
    executed = [
        record
        for record in records
        if record["actually_executed_defective_phase3_consumer"]
    ]
    submitted_only = [
        record
        for record in records
        if record["evidence_status"] == "submitted_affected_no_execution_proof"
    ]
    assert executed
    assert submitted_only

    for record in records:
        configuration = record["defect_configuration"]
        assert configuration["configured_normalization_mode"] == builder.SIGNED_MODE
        assert configuration["normalized_feature_policy"] == builder.SIGNED_MODE
        assert configuration["configured_and_normalized_policy_match"] is True
        scoring_member = configuration["source_lock"]["scoring_member"]
        assert scoring_member["sha256"] in builder.KNOWN_DEFECTIVE_SCORING_SHA256S
        assert scoring_member["known_defective_consumer"] is True
        assert record["accepted_trajectory_change"] == {
            "proven": False,
            "evidence": [],
            "reason": (
                "No authenticated corrected counterfactual replay or rescoring "
                "receipt proves a changed winner or accepted prefix for this protocol."
            ),
        }

    for record in executed:
        assert record["evidence_status"] == "authenticated_executed_defective_consumer"
        assert record["execution_evidence"]
        assert all(
            evidence["protocol_sha256"]
            == record["protocol"]["canonical_sha256"]
            for evidence in record["execution_evidence"]
        )
    for record in submitted_only:
        assert record["actually_executed_defective_phase3_consumer"] is False
        assert record["execution_evidence"] == []
        assert record["submission_evidence"]

    summary = affected_map["summary"]
    assert summary["authenticated_executed_protocol_digest_count"] == 34
    assert summary["accepted_trajectory_change_proven_count"] == 0
    observation = affected_map["confirmed_score_factor_mismatch_observations"][0]
    assert observation["phase_ii_signed_factor"] == 0.8645492142678116
    assert observation["recorded_phase_iii_factor"] == 1.0
    assert observation["score_factor_mismatch_proven"] is True
    assert observation["accepted_winner_or_prefix_change_proven"] is False


def test_output_is_deterministic_self_digested_and_inputs_are_not_mutated(
    tmp_path: Path,
) -> None:
    inputs = builder.source_input_paths()
    before = {path: builder.sha256_file(path) for path in inputs}

    first = builder.build_map()
    second = builder.build_map()
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    builder.write_map(first_path, first)
    builder.write_map(second_path, second)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert first["sha256"] == builder.canonical_sha256(first)
    assert json.loads(first_path.read_text(encoding="utf-8"))["sha256"] == first["sha256"]
    assert {path: builder.sha256_file(path) for path in inputs} == before
    assert set(tmp_path.iterdir()) == {first_path, second_path}
