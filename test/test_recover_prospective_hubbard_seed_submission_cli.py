"""CLI coverage for canonical Hubbard-L4 submission recovery."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from pipelines.time_dynamics.prospective_hubbard_seed_recovery import (
    HUBBARD_LANE_ID,
)
from pipelines.time_dynamics.prospective_submission_recovery import (
    SubmissionRecoveryError,
)
from pipelines.time_dynamics.runners.recover_prospective_hubbard_seed_submission import (
    main,
)


def _load_recovery_fixtures():
    source = Path(__file__).with_name("test_prospective_hubbard_seed_recovery.py")
    spec = importlib.util.spec_from_file_location(
        "_paper_ii_hubbard_recovery_test_fixtures", source
    )
    if spec is None or spec.loader is None:  # pragma: no cover - importlib guard
        raise RuntimeError("could not load Hubbard recovery test fixtures")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_FIXTURES = _load_recovery_fixtures()
_campaign = _FIXTURES._campaign
_reseal_snapshot = _FIXTURES._reseal_snapshot


def _arguments(root: Path, campaign: dict, *extra: str) -> list[str]:
    return [
        "--repository-root",
        str(root),
        "--intent",
        str(campaign["intent_path"]),
        "--provisional-result",
        str(campaign["result_path"]),
        "--later-combined-snapshot",
        str(campaign["later_path"]),
        *extra,
    ]


def _canonical_receipt(root: Path, campaign: dict) -> Path:
    return (
        root
        / "output"
        / HUBBARD_LANE_ID
        / "acquisition_supervisor/submission_receipts"
        / f"{campaign['unit_id']}.json"
    )


def test_cli_recovers_and_replays_with_identical_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    campaign = _campaign(tmp_path)
    receipt = _canonical_receipt(tmp_path, campaign)
    arguments = _arguments(
        tmp_path,
        campaign,
        "--output",
        receipt.relative_to(tmp_path).as_posix(),
    )

    assert main(arguments) == 0
    first = capsys.readouterr().out
    assert main(arguments) == 0
    second = capsys.readouterr().out

    summary = json.loads(first)
    assert second == first
    assert summary == {
        "cluster_id": 9714040,
        "lane_id": HUBBARD_LANE_ID,
        "mode": "recover",
        "process_ids": [0],
        "receipt": receipt.relative_to(tmp_path).as_posix(),
        "receipt_file_sha256": summary["receipt_file_sha256"],
        "remote_raw_evidence_policy": "preserve",
        "scheduler_confirmation": True,
        "snapshot_sha256": campaign["later"]["snapshot_sha256"],
        "status": "canonical_submission_receipt_recovered",
        "submission_authorized": False,
        "unit_id": campaign["unit_id"],
    }
    assert len(summary["receipt_file_sha256"]) == 64
    assert receipt.is_file()


def test_validate_only_requires_existing_receipt_then_replays_read_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    campaign = _campaign(tmp_path)
    arguments = _arguments(tmp_path, campaign)

    with pytest.raises(SubmissionRecoveryError, match="requires an existing"):
        main([*arguments, "--validate-only"])

    assert main(arguments) == 0
    capsys.readouterr()
    receipt = _canonical_receipt(tmp_path, campaign)
    before = receipt.read_bytes()

    assert main([*arguments, "--validate-only"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["mode"] == "validate"
    assert summary["status"] == "canonical_submission_receipt_validated"
    assert receipt.read_bytes() == before


def test_cli_rejects_noncanonical_output_without_materializing(
    tmp_path: Path,
) -> None:
    campaign = _campaign(tmp_path)

    with pytest.raises(SubmissionRecoveryError, match="canonical Hubbard receipt"):
        main(
            _arguments(
                tmp_path,
                campaign,
                "--output",
                "output/not-the-canonical-receipt.json",
            )
        )

    assert not _canonical_receipt(tmp_path, campaign).exists()


def test_cli_rejects_first_cycle_indeterminate_snapshot(
    tmp_path: Path,
) -> None:
    campaign = _campaign(tmp_path)

    def make_indeterminate(value: dict) -> None:
        value["queries"]["queue"]["rows"] = []
        value["reconciliation"] = {
            campaign["batch"]: {
                "status": "indeterminate",
                "match_count": 0,
                "matches": [],
            }
        }

    _reseal_snapshot(campaign["later_path"], make_indeterminate)
    with pytest.raises(SubmissionRecoveryError, match="exactly one Hubbard"):
        main(_arguments(tmp_path, campaign))

    assert not _canonical_receipt(tmp_path, campaign).exists()


@pytest.mark.parametrize("failure", ["stale", "wrong_identity"])
def test_cli_rejects_stale_or_wrong_identity_snapshot(
    tmp_path: Path, failure: str
) -> None:
    campaign = _campaign(tmp_path)
    if failure == "stale":
        _reseal_snapshot(
            campaign["later_path"],
            lambda value: value.__setitem__(
                "captured_at", "2026-08-29T01:06:00-05:00"
            ),
        )
        expected = "must be later"
    else:
        _reseal_snapshot(
            campaign["later_path"],
            lambda value: value["queries"]["queue"]["rows"][0].__setitem__(
                "PaperIIExecutionRequestSha256", "0" * 64
            ),
        )
        expected = "batch/unit/lane/action/request/source"

    with pytest.raises(SubmissionRecoveryError, match=expected):
        main(_arguments(tmp_path, campaign))

    assert not _canonical_receipt(tmp_path, campaign).exists()
