from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from pipelines.reporting import build_paper_i_hh_sr_no_prune_no_beam_tracking_pdf as tracker_builder
from pipelines.reporting.build_paper_i_hh_comparator_tracking_summary import (
    SCHEMA,
    build_tracking_summary,
)


def _add_json(archive: tarfile.TarFile, name: str, payload: dict) -> None:
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(raw)
    archive.addfile(info, io.BytesIO(raw))


def _synthetic_archive(path: Path, *, job_id: str) -> None:
    history = [
        {
            "outer_iteration": index,
            "abs_delta_e_same_cutoff_after": 1.0 / index,
        }
        for index in range(1, 51)
    ]
    result = {
        "status": "completed",
        "method_id": "static_full_meta_append_adapt_vqe",
        "n_ph_work": 7,
        "n_ph_reference": 7,
        "same_cutoff_reference": True,
        "result": {
            "abs_delta_e_same_cutoff": 0.02,
            "adapt_depth_reached": 41,
            "S_alg": 12345,
            "adapt_history": history,
        },
    }
    receipt = {
        "schema": "paper_i_hh_append_completion_validation_receipt_v1",
        "status": "pass",
        "job_id": job_id,
        "variant": "macro",
        "adapt_iterations": 50,
        "ledger_closure": "pass",
        "sector_leak_flag": False,
        "boson_truncation_leak_flag": False,
        "active_depth": 41,
        "same_cutoff_abs_error": 0.02,
        "S_alg": 12345,
        "compiled_count_2q_total": 101,
        "compiled_depth_2q_total": 77,
        "compiled_depth_total": 222,
    }
    with tarfile.open(path, "w:gz") as archive:
        _add_json(archive, f"payload/{job_id}/result.json", result)
        _add_json(
            archive,
            f"payload/{job_id}/validation_receipt.json",
            receipt,
        )


def test_bounded_summary_records_exact_archive_and_member_identity(tmp_path) -> None:
    job_id = "append_macro__strong_strong__r50"
    archive_path = tmp_path / f"{job_id}_transfer.tar.gz"
    output_path = tmp_path / f"{job_id}_tracking_summary.json"
    _synthetic_archive(archive_path, job_id=job_id)

    summary = build_tracking_summary(
        archive_path=archive_path,
        job_id=job_id,
        output_json=output_path,
    )

    assert summary["schema"] == SCHEMA
    assert summary["status"] == "pass"
    assert summary["archive"]["sha256"]
    assert summary["archive"]["size_bytes"] == archive_path.stat().st_size
    assert summary["result_member"]["name"].endswith(f"{job_id}/result.json")
    assert summary["validation_receipt_member"]["name"].endswith(
        f"{job_id}/validation_receipt.json"
    )
    assert summary["projection"]["bounded_memory"] is True
    assert len(summary["result"]["trajectory"]) == 50
    assert summary["result"]["trajectory"][-1] == {
        "round": 50,
        "error": pytest.approx(0.02),
    }
    assert summary["qiskit"] == {"N2q": 101, "D2q": 77, "Dc": 222}
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "pass"


def test_tracker_uses_compact_summary_and_fails_closed_on_archive_sha(
    tmp_path, monkeypatch
) -> None:
    job_id = "append_macro__strong_strong__r50"
    archive_path = tmp_path / f"{job_id}_transfer.tar.gz"
    summary_path = tmp_path / f"{job_id}_tracking_summary.json"
    _synthetic_archive(archive_path, job_id=job_id)
    build_tracking_summary(
        archive_path=archive_path,
        job_id=job_id,
        output_json=summary_path,
    )
    monkeypatch.setattr(tracker_builder, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        tracker_builder,
        "_tar_json_members",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("compact-summary route must not materialize result.json")
        ),
    )
    archives = {
        regime: (
            archive_path
            if regime == "strong_strong_u8"
            else tmp_path / f"missing_{regime}.tar.gz"
        )
        for regime in tracker_builder.REGIMES
    }

    route = tracker_builder._build_comparator_archive_route(
        route_id="append_adapt_macro_nph3_7",
        label="Append macro",
        subtitle="test",
        policy="test",
        archives=archives,
        expected_receipt_schema=(
            "paper_i_hh_append_completion_validation_receipt_v1"
        ),
        expected_variant="append_macro",
        expected_method_id="static_full_meta_append_adapt_vqe",
        sources=[],
        tracking_summaries={"strong_strong_u8": summary_path},
    )

    result = route["results"]["strong_strong_u8"]
    assert result["status"] == "complete"
    assert result["trajectory"][-1]["error"] == pytest.approx(0.02)
    assert route["costs"]["strong_strong_u8"] == {
        "N2q": 101,
        "D2q": 77,
        "Dc": 222,
    }

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["archive"]["sha256"] = "0" * 64
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(RuntimeError, match="archive SHA-256 drift"):
        tracker_builder._build_comparator_archive_route(
            route_id="append_adapt_macro_nph3_7",
            label="Append macro",
            subtitle="test",
            policy="test",
            archives=archives,
            expected_receipt_schema=(
                "paper_i_hh_append_completion_validation_receipt_v1"
            ),
            expected_variant="append_macro",
            expected_method_id="static_full_meta_append_adapt_vqe",
            sources=[],
            tracking_summaries={"strong_strong_u8": summary_path},
        )


def test_tracker_fails_closed_when_late_summary_is_missing(tmp_path) -> None:
    archive_path = tmp_path / "late_transfer.tar.gz"
    archive_path.write_bytes(b"present archive")

    with pytest.raises(RuntimeError, match="compact tracking summary is required"):
        tracker_builder._comparator_tracking_summary(
            archive_path=archive_path,
            summary_path=tmp_path / "missing_tracking_summary.json",
            regime="strong_strong_u8",
            expected_receipt_schema=(
                "paper_i_hh_append_completion_validation_receipt_v1"
            ),
            expected_variant="append_macro",
            expected_method_id="static_full_meta_append_adapt_vqe",
        )
