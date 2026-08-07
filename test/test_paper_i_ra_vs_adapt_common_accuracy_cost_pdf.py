from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile

import pytest

from pipelines.reporting import (
    build_paper_i_ra_vs_adapt_common_accuracy_cost_pdf as report,
)


def _trace(errors: list[float]) -> list[dict[str, float | int]]:
    assert len(errors) == 50
    return [
        {
            "controller_round": controller_round,
            "absolute_energy_error": error,
        }
        for controller_round, error in enumerate(errors, start=1)
    ]


def _write_result_archive(tmp_path: Path, result: dict) -> Path:
    archive_path = tmp_path / "attempt.tar.gz"
    raw = json.dumps(result, sort_keys=True).encode("utf-8")
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo("worker_outputs/result.json")
        member.size = len(raw)
        archive.addfile(member, io.BytesIO(raw))
    return archive_path


def test_streamed_projection_selects_only_requested_ra_round(
    tmp_path: Path,
) -> None:
    result = {
        "run": {
            "accepted_trajectory": [
                {"controller_round": index, "value": f"state-{index}"}
                for index in range(1, 51)
            ],
            "scientific_replay": [
                {"controller_round": index, "value": f"replay-{index}"}
                for index in range(1, 51)
            ],
            "canonical_reporting": {
                "accepted_prefix_work": [
                    {"controller_round": index, "s_alg": index * 10}
                    for index in range(1, 51)
                ],
                "reference_state": {"source_label": "hf"},
            },
            "route": {"profile": "ra"},
            "problem": {"problem_request_sha256": "a" * 64},
        }
    }
    archive = _write_result_archive(tmp_path, result)

    projection = report._capture_result_objects(
        {
            "method_family": "ra",
            "execution_id": "ra-test",
            "attempt_path": str(archive),
            "result_member": "worker_outputs/result.json",
        },
        controller_round=17,
    )

    assert projection["run.accepted_trajectory.item"]["value"] == "state-17"
    assert projection["run.scientific_replay.item"]["value"] == "replay-17"
    assert (
        projection[
            "run.canonical_reporting.accepted_prefix_work.item"
        ]["s_alg"]
        == 170
    )
    assert projection["run.route"] == {"profile": "ra"}


def test_streamed_projection_selects_only_requested_append_round(
    tmp_path: Path,
) -> None:
    result = {
        "result_payload": {
            "controller_replay_evidence": {
                "signed_controller_round_prefixes": [
                    {
                        "controller_round": index,
                        "active_prefix_checkpoint": {"round": index},
                    }
                    for index in range(1, 51)
                ]
            }
        }
    }
    archive = _write_result_archive(tmp_path, result)

    projection = report._capture_result_objects(
        {
            "method_family": "append",
            "execution_id": "append-test",
            "attempt_path": str(archive),
            "result_member": "worker_outputs/result.json",
        },
        controller_round=29,
    )

    key = (
        "result_payload.controller_replay_evidence."
        "signed_controller_round_prefixes.item"
    )
    assert projection[key]["controller_round"] == 29


def test_common_target_uses_worse_horizon_minimum_and_earliest_crossings() -> None:
    ra_errors = [1.0] * 50
    adapt_errors = [1.0] * 50
    ra_errors[6] = 0.2
    ra_errors[29] = 0.1
    adapt_errors[3] = 0.2
    adapt_errors[39] = 0.05

    selected = report.select_full_horizon_common_accuracy(
        _trace(ra_errors),
        _trace(adapt_errors),
    )

    assert selected["common_target_absolute_error"] == pytest.approx(0.1)
    assert selected["limiting_method"] == "ra"
    assert selected["ra_crossing_controller_round"] == 30
    assert selected["adapt_crossing_controller_round"] == 40
    assert selected["ra_crossing_absolute_error"] == pytest.approx(0.1)
    assert selected["adapt_crossing_absolute_error"] == pytest.approx(0.05)


def test_common_target_rejects_short_or_noncontiguous_traces() -> None:
    with pytest.raises(
        report.CommonAccuracyInputError,
        match="does not span the 50-round horizon",
    ):
        report.select_full_horizon_common_accuracy(
            _trace([1.0] * 50)[:-1],
            _trace([1.0] * 50),
        )

    drifted = _trace([1.0] * 50)
    drifted[8]["controller_round"] = 10
    with pytest.raises(
        report.CommonAccuracyInputError,
        match="is not contiguous",
    ):
        report.select_full_horizon_common_accuracy(
            drifted,
            _trace([1.0] * 50),
        )


def test_cost_ratios_are_adapt_over_ra_and_dominance_is_explicit() -> None:
    ra = {"N2q": 10, "D2q": 5, "Dc": 20, "W1q": 30, "S_alg": 100}
    adapt = {"N2q": 20, "D2q": 10, "Dc": 40, "W1q": 60, "S_alg": 200}

    classified = report.classify_costs(ra, adapt)

    assert classified["ratios"] == {
        "N2q": 2.0,
        "D2q": 2.0,
        "Dc": 2.0,
        "W1q": 2.0,
        "S_alg": 2.0,
    }
    assert classified["circuit_verdict"] == "RA"
    assert classified["s_alg_verdict"] == "RA"
    assert classified["overall_verdict"] == "RA"


def test_cross_coordinate_tradeoff_is_mixed_not_forced_to_a_winner() -> None:
    ra = {"N2q": 10, "D2q": 20, "Dc": 30, "W1q": 40, "S_alg": 100}
    adapt = {"N2q": 9, "D2q": 21, "Dc": 30, "W1q": 40, "S_alg": 99}

    classified = report.classify_costs(ra, adapt)

    assert classified["circuit_verdict"] == "mixed"
    assert classified["s_alg_verdict"] == "ADAPT"
    assert classified["overall_verdict"] == "mixed"


def test_s_alg_display_uses_requested_scientific_notation() -> None:
    assert report._format_s_alg(0) == "0.0e0"
    assert report._format_s_alg(123456) == "1.2e5"
    assert report._format_s_alg(999) == "1.0e3"
