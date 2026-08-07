from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
from typing import Any

import pytest

from pipelines.reporting import (
    add_paper_i_historical_mean_global_singleton_live_page7 as subject,
)


EXACT = {
    "weak_weak": -1.0,
    "intermediate_weak": -0.9,
    "strong_weak_u8": -0.8,
    "weak_strong": -0.7,
    "intermediate_strong": -0.6,
    "strong_strong_u8": -0.5,
}
PROTOCOL = "f" * 64


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _costs(seed: int) -> dict[str, int]:
    return {
        "N2q": seed + 1,
        "D2q": seed + 2,
        "Dc": seed + 3,
        "W1q": seed + 4,
        "S_alg": seed + 5,
    }


def _points(exact: float, horizon: int) -> list[dict[str, Any]]:
    return [
        {
            "round": round_index,
            "energy": exact + 1.0 / (round_index + 1),
            "delta_e": 1.0 / (round_index + 1),
        }
        for round_index in range(horizon + 1)
    ]


def _base_cell(regime: str) -> dict[str, Any]:
    exact = EXACT[regime]
    append_horizon = subject.completed.APPEND_TERMINAL_ROUND_BY_REGIME[regime]
    append_points = _points(exact, append_horizon)
    append = {
        "points": append_points,
        "exact_same_cutoff_energy": exact,
        "display_terminal_round": append_horizon,
        "effective_plateau": subject.completed._effective_plateau(
            append_points, label=f"fixture {regime} Append"
        ),
        "terminal": {
            "round": append_horizon,
            "energy": append_points[-1]["energy"],
            "delta_e": append_points[-1]["delta_e"],
            "costs": _costs(100 + append_horizon),
        },
    }
    cell: dict[str, Any] = {
        "regime_id": regime,
        "display_name": subject.completed.REGIME_LABELS[regime],
        "nph": subject.completed.NPH_BY_REGIME[regime],
        "append": append,
    }
    if regime in subject.completed.NPH3_REGIMES:
        ra_points = _points(exact, 50)
        cell.update(
            {
                "status": "complete",
                "ra": {
                    "points": ra_points,
                    "effective_plateau": subject.completed._effective_plateau(
                        ra_points, label=f"fixture {regime} RA"
                    ),
                    "terminal": {
                        "round": 50,
                        "energy": ra_points[-1]["energy"],
                        "delta_e": ra_points[-1]["delta_e"],
                        "costs": _costs(250),
                    },
                    "source": {"fixture": True},
                },
                "common_accuracy": {
                    "target_delta_e": 0.1,
                    "ra": {"round": 9, "costs": _costs(20)},
                    "append": {"round": 9, "costs": _costs(30)},
                },
            }
        )
    else:
        cell.update({"status": "pending", "ra": None, "common_accuracy": None})
    return cell


def _install_base_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "base_adapter.json"
    base = subject.digested(
        {
            "schema": "fixture_completed_adapter_v1",
            "completed_regimes": list(subject.completed.REGIME_ORDER[:3]),
            "pending_regimes": list(subject.completed.REGIME_ORDER[3:]),
            "append_adapter": {"canonical_sha256": "a" * 64},
            "same_cutoff_reference": {"sha256": "b" * 64},
            "route_description": "fixture global-singleton route",
            "marker_policy": "effective_plateau",
            "cost_policy": {"complete": "compiled"},
            "cells": [
                _base_cell(regime) for regime in subject.completed.REGIME_ORDER
            ],
        }
    )
    _write_json(path, base)

    def fake_validate(candidate: Path) -> dict[str, Any]:
        raw = json.loads(Path(candidate).read_text(encoding="utf-8"))
        unsigned = copy.deepcopy(raw)
        observed = unsigned.pop("sha256")
        assert observed == hashlib.sha256(
            subject.canonical_json_bytes(unsigned)
        ).hexdigest()
        result = copy.deepcopy(raw)
        result["file_binding"] = subject.file_binding(Path(candidate))
        return result

    monkeypatch.setattr(subject.completed, "validate_adapter", fake_validate)
    monkeypatch.setattr(subject, "_loaderfix_protocol_sha256", lambda regime: PROTOCOL)

    def fake_authority(
        regime: str, *, execution_id: str, protocol_sha256: str
    ) -> dict[str, Any]:
        assert execution_id == subject._expected_execution_id(regime)
        assert protocol_sha256 == PROTOCOL
        return {
            "exact_same_cutoff_energy": EXACT[regime],
            "resume_controller_round": subject.RESUME_ROUNDS[regime],
            "execution_id": execution_id,
            "regime_id": regime,
            "cluster_id": subject.CLUSTER_ID,
            "proc_id": subject.PROC_IDS[regime],
            "scientific_protocol_sha256": protocol_sha256,
            "route_contract_sha256": subject.completed.ROUTE_CONTRACT_SHA256,
            "route_profile": subject.ROUTE_PROFILE,
            "package_manifest": {"sha256": "1" * 64},
            "activation_manifest": {"sha256": "2" * 64},
            "submission_receipt": {"sha256": "3" * 64},
            "job": {"sha256": "4" * 64},
            "authorization": {"sha256": "5" * 64},
            "source_job": {"sha256": "6" * 64},
            "source_checkpoint_sha256": {
                "weak_strong": "7" * 64,
                "intermediate_strong": "6" * 64,
                "strong_strong_u8": "5" * 64,
            }[regime],
        }

    monkeypatch.setattr(subject, "_validate_loaderfix_authority", fake_authority)
    return path


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _snapshot_fixture(
    tmp_path: Path,
    *,
    regime: str = "weak_strong",
    live_round: int | None = None,
    timestamp: str = "20260803T155149Z",
) -> tuple[Path, Path]:
    exact = EXACT[regime]
    live_round = (
        subject.RESUME_ROUNDS[regime] if live_round is None else int(live_round)
    )
    components = ["N_H_outer", "N_grad", "N_H_outer", "N_metric"]
    component_counts = {field: components.count(field) for field in subject.STREAM_COMPONENTS}
    occurrences = [
        {
            "sequence": index,
            "component": component,
            "primitive_id": f"p{index}",
            "charged": True,
        }
        for index, component in enumerate(components, start=1)
    ]
    ledger = {
        "schema": "paper_i_estimator_call_ledger_checkpoint_sidecar_v2",
        "checkpoint": {
            "depth": live_round,
            "current_round_finalized": True,
        },
        "S_alg": len(occurrences),
        "raw_occurrence_count": len(occurrences),
        "ledger": {
            "schema": "estimator_call_ledger_v1",
            "occurrences": occurrences,
            "occurrence_summary": {
                "total_call_occurrences": len(occurrences),
                "components": component_counts,
                "component_occurrence_counts": component_counts,
                **component_counts,
            },
        },
    }
    ledger_bytes = _json_bytes(ledger)
    ledger_sha = hashlib.sha256(ledger_bytes).hexdigest()
    ledger_name = (
        f"checkpoint.estimator_call_ledger_checkpoint.{ledger_sha[:16]}.json"
    )
    source_projection_sha = "d" * 64
    resume = {
        "schema": "static_adapt_signed_active_prefix_resume_sidecar_v2",
        "source_result_sha256": source_projection_sha,
        "controller_state": {"controller_round": live_round},
        "selection_state": {"controller_round": live_round},
    }
    resume_bytes = _json_bytes(resume)
    resume_sha = hashlib.sha256(resume_bytes).hexdigest()
    resume_name = f"checkpoint.verified_singleton_resume.{resume_sha[:16]}.json"
    history = []
    previous = exact + 1.0
    for round_index in range(1, live_round + 1):
        after = exact + 1.0 / (round_index + 1)
        history.append(
            {"energy_before_opt": previous, "energy_after_opt": after}
        )
        previous = after
    checkpoint = {
        "adapt_vqe": {
            "route_profile": subject.ROUTE_PROFILE,
            "sr_route_profile_contract_sha256": (
                subject.completed.ROUTE_CONTRACT_SHA256
            ),
            "accepted_state_resume": {
                "schema": "paper_i_canonical_accepted_state_resume_v1",
                "source_sha256": {
                    "weak_strong": "7" * 64,
                    "intermediate_strong": "6" * 64,
                    "strong_strong_u8": "5" * 64,
                }[regime],
                "source_controller_round": subject.RESUME_ROUNDS[regime],
                "strict_numerical_replay_passed": True,
                "route_and_problem_binding_passed": True,
            },
            "history_count": live_round,
            "ansatz_depth": live_round,
            "terminal_active_prefix_checkpoint": {
                "active_ansatz_depth": live_round
            },
            "history": history,
            "estimator_call_accounting": {
                "complete": True,
                "S_alg": len(occurrences),
                "components": component_counts,
            },
            "S_alg": len(occurrences),
            "S_alg_components": component_counts,
            "estimator_call_ledger_checkpoint": {
                "path": ledger_name,
                "sha256": ledger_sha,
                "S_alg": len(occurrences),
                "raw_occurrence_count": len(occurrences),
                "checkpoint_depth": live_round,
                "status": "complete",
                "current_round_finalized": True,
            },
            "verified_singleton_resume_sidecar": {
                "path": resume_name,
                "sha256": resume_sha,
                "status": "complete",
                "source_projection_sha256": source_projection_sha,
            },
        }
    }
    checkpoint_bytes = _json_bytes(checkpoint)
    members = {
        "checkpoint.json": checkpoint_bytes,
        ledger_name: ledger_bytes,
        resume_name: resume_bytes,
    }
    archive_path = tmp_path / (
        f"{subject.CLUSTER_ID}.{subject.PROC_IDS[regime]}__"
        f"{subject.SNAPSHOT_REGIME_TOKENS[regime]}__{timestamp}.tar.gz"
    )
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))
    member_bindings = {
        name: {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        for name, payload in members.items()
    }
    validation = {
        "schema": subject.SNAPSHOT_RECEIPT_SCHEMA,
        "archive": str(archive_path.resolve()),
        "archive_sha256": subject.sha256_file(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "checkpoint_depth": live_round,
        "members": member_bindings,
        "pointers": {
            "ledger": {"path": ledger_name, "sha256": ledger_sha},
            "resume": {"path": resume_name, "sha256": resume_sha},
        },
        "validation": "passed",
    }
    validation_path = archive_path.with_name(
        archive_path.name[: -len(".tar.gz")] + ".validation.json"
    )
    _write_json(validation_path, validation)
    return archive_path, validation_path


def _build_projection(
    tmp_path: Path,
    *,
    base: Path,
    regime: str,
    live_round: int,
    timestamp: str,
) -> Path:
    archive, validation = _snapshot_fixture(
        tmp_path,
        regime=regime,
        live_round=live_round,
        timestamp=timestamp,
    )
    output = tmp_path / f"{archive.name[: -len('.tar.gz')]}.live_projection.json"
    subject.build_live_projection_from_snapshot(
        base_adapter_path=base,
        regime=regime,
        archive_path=archive,
        validation_path=validation,
        output=output,
    )
    return output


def _built_projection_set(
    tmp_path: Path,
    *,
    base: Path,
    tag: int,
    ws_round: int = 36,
) -> dict[str, Path]:
    rounds = {
        "weak_strong": ws_round,
        "intermediate_strong": 32,
        "strong_strong_u8": 18,
    }
    return {
        regime: _build_projection(
            tmp_path,
            base=base,
            regime=regime,
            live_round=round_index,
            timestamp=f"20260803T{tag + index:06d}Z",
        )
        for index, (regime, round_index) in enumerate(rounds.items())
    }


def _pdf(path: Path, labels: list[str]) -> None:
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject

    writer = PdfWriter()
    for label in labels:
        page = writer.add_blank_page(width=612, height=792)
        stream = DecodedStreamObject()
        stream.set_data(f"% {label}\n".encode("ascii"))
        page.replace_contents(stream)
    with path.open("wb") as output:
        writer.write(output)


def test_snapshot_projection_streams_triplet_without_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _install_base_adapter(tmp_path, monkeypatch)
    archive, validation = _snapshot_fixture(tmp_path)
    output = tmp_path / "weak_strong_live.json"

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("snapshot members must never be extracted")

    monkeypatch.setattr(tarfile.TarFile, "extract", forbidden)
    monkeypatch.setattr(tarfile.TarFile, "extractall", forbidden)
    projection = subject.build_live_projection_from_snapshot(
        base_adapter_path=base,
        regime="weak_strong",
        archive_path=archive,
        validation_path=validation,
        output=output,
    )

    assert projection["live_controller_round"] == subject.RESUME_ROUNDS["weak_strong"]
    assert [point["round"] for point in projection["points"]] == list(
        range(projection["live_controller_round"] + 1)
    )
    assert projection["points"][0]["energy"] == pytest.approx(
        EXACT["weak_strong"] + 1.0
    )
    assert projection["algorithmic_work"] == {
        "components": {
            "N_H_outer": 2,
            "N_H_refit": 0,
            "N_grad": 1,
            "N_metric": 1,
        },
        "S_alg": 4,
    }
    assert projection["qiskit_costs"] == subject.QISKIT_PENDING
    assert subject._verify_self_digest(projection, label="projection") == projection["sha256"]
    base_adapter = subject.completed.validate_adapter(base)
    append = next(
        cell["append"]
        for cell in base_adapter["cells"]
        if cell["regime_id"] == "weak_strong"
    )
    validated = subject.validate_live_projection(
        output, regime="weak_strong", append_cell=append
    )
    assert validated["live_controller_round"] == subject.RESUME_ROUNDS["weak_strong"]


def test_projection_validator_rejects_redigested_snapshot_forgery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _install_base_adapter(tmp_path, monkeypatch)
    archive, validation = _snapshot_fixture(tmp_path)
    output = tmp_path / "valid_live.json"
    subject.build_live_projection_from_snapshot(
        base_adapter_path=base,
        regime="weak_strong",
        archive_path=archive,
        validation_path=validation,
        output=output,
    )
    base_adapter = subject.completed.validate_adapter(base)
    append = next(
        cell["append"]
        for cell in base_adapter["cells"]
        if cell["regime_id"] == "weak_strong"
    )

    forged_trajectory = json.loads(output.read_text(encoding="utf-8"))
    forged_trajectory.pop("sha256")
    forged_trajectory["points"][-1]["energy"] -= 1.0e-4
    forged_trajectory["points"][-1]["delta_e"] = abs(
        forged_trajectory["points"][-1]["energy"] - EXACT["weak_strong"]
    )
    forged_trajectory = subject.digested(forged_trajectory)
    trajectory_path = tmp_path / "forged_trajectory.json"
    _write_json(trajectory_path, forged_trajectory)
    with pytest.raises(
        subject.LivePage7InputError,
        match="compact projection does not equal authenticated snapshot",
    ):
        subject.validate_live_projection(
            trajectory_path, regime="weak_strong", append_cell=append
        )

    forged_identity = json.loads(output.read_text(encoding="utf-8"))
    forged_identity.pop("sha256")
    forged_identity["snapshot_execution_binding"][
        "source_checkpoint_sha256"
    ] = "0" * 64
    forged_identity = subject.digested(forged_identity)
    identity_path = tmp_path / "forged_identity.json"
    _write_json(identity_path, forged_identity)
    with pytest.raises(
        subject.LivePage7InputError, match="declared snapshot execution drifted"
    ):
        subject.validate_live_projection(
            identity_path, regime="weak_strong", append_cell=append
        )


def test_snapshot_projection_rejects_rehashed_ledger_component_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _install_base_adapter(tmp_path, monkeypatch)
    archive, validation = _snapshot_fixture(tmp_path)
    receipt = json.loads(validation.read_text(encoding="utf-8"))
    ledger_name = receipt["pointers"]["ledger"]["path"]
    receipt["members"][ledger_name]["sha256"] = "0" * 64
    receipt["pointers"]["ledger"]["sha256"] = "0" * 64
    _write_json(validation, receipt)

    with pytest.raises(subject.LivePage7InputError, match="pointer hash drifted"):
        subject.build_live_projection_from_snapshot(
            base_adapter_path=base,
            regime="weak_strong",
            archive_path=archive,
            validation_path=validation,
            output=tmp_path / "rejected.json",
        )


def test_live_adapter_preserves_complete_cells_and_append_and_is_monotone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_path = _install_base_adapter(tmp_path, monkeypatch)
    base = subject.completed.validate_adapter(base_path)
    output = tmp_path / "live_adapter.json"
    first_projections = _built_projection_set(
        tmp_path, base=base_path, tag=155100
    )
    adapter = subject.build_live_adapter(
        base_adapter_path=base_path,
        live_projections=first_projections,
        output=output,
    )
    base_cells = {cell["regime_id"]: cell for cell in base["cells"]}
    live_cells = {cell["regime_id"]: cell for cell in adapter["cells"]}
    assert adapter["layout"] == {"panel_count": 6, "grid": "2x3", "page_count": 1}
    for regime in subject.completed.REGIME_ORDER:
        assert subject.canonical_json_bytes(live_cells[regime]["append"]) == subject.canonical_json_bytes(
            base_cells[regime]["append"]
        )
    for regime in subject.completed.NPH3_REGIMES:
        assert subject.canonical_json_bytes(live_cells[regime]) == subject.canonical_json_bytes(
            base_cells[regime]
        )
    for regime in subject.completed.NPH7_REGIMES:
        assert live_cells[regime]["status"] == "live_partial"
        assert live_cells[regime]["ra"]["terminal"]["costs"]["N2q"] is None
        assert live_cells[regime]["ra"]["qiskit_status"] == subject.LIVE_QISKIT_STATUS

    advanced_projections = dict(first_projections)
    advanced_projections["weak_strong"] = _build_projection(
        tmp_path,
        base=base_path,
        regime="weak_strong",
        live_round=37,
        timestamp="20260803T155200Z",
    )
    advanced = subject.build_live_adapter(
        base_adapter_path=base_path,
        live_projections=advanced_projections,
        output=output,
    )
    assert next(
        cell for cell in advanced["cells"] if cell["regime_id"] == "weak_strong"
    )["ra"]["live_controller_round"] == 37
    regressed_projections = dict(first_projections)
    regressed_projections["weak_strong"] = _build_projection(
        tmp_path,
        base=base_path,
        regime="weak_strong",
        live_round=36,
        timestamp="20260803T155300Z",
    )
    with pytest.raises(subject.LivePage7InputError, match="regressed"):
        subject.build_live_adapter(
            base_adapter_path=base_path,
            live_projections=regressed_projections,
            output=output,
        )

    png = tmp_path / "live.png"
    plot_pdf = tmp_path / "live_plot.pdf"
    tex = tmp_path / "live.tex"
    subject.render_plot(advanced, png_path=png, pdf_path=plot_pdf)
    subject.write_page_tex(advanced, plot_pdf=plot_pdf, tex_path=tex)
    assert png.is_file() and plot_pdf.is_file()
    assert tex.read_text(encoding="utf-8").count(r"\text{Qiskit pending}") >= 3


def test_update_replaces_only_page_seven_and_records_pending_qiskit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_path = _install_base_adapter(tmp_path, monkeypatch)
    adapter_path = tmp_path / "live_adapter.json"
    adapter = subject.build_live_adapter(
        base_adapter_path=base_path,
        live_projections=_built_projection_set(
            tmp_path, base=base_path, tag=155400
        ),
        output=adapter_path,
    )
    target_pdf = tmp_path / "report.pdf"
    _pdf(target_pdf, [f"old-page-{index}" for index in range(1, 8)])
    before = subject.completed.legacy_page._page_content_hashes(target_pdf)
    provenance_path = tmp_path / "report_provenance.json"
    provenance = {
        "outputs": {"partial_progress_pdf": subject.file_binding(target_pdf)},
        "layout": {
            "page_count": 7,
            "page_6": subject.completed.EXPECTED_BASE_PAGE_6,
            "page_7": subject.completed.PAGE_ID,
        },
        subject.completed.REPORT_KEY: {
            "adapter": {"canonical_sha256": "0" * 64},
            "completed_cell_sha256": copy.deepcopy(adapter["completed_cell_sha256"]),
            "append_cell_sha256": copy.deepcopy(adapter["append_cell_sha256"]),
        },
        "limitations": [subject.completed.LIMITATION],
    }
    _write_json(provenance_path, provenance)
    asset_dir = tmp_path / "assets"

    def fake_assets(
        value: Any, *, asset_dir: Path, asset_stem: str
    ) -> dict[str, Path]:
        asset_dir.mkdir(parents=True, exist_ok=True)
        assets = {
            "plot_png": asset_dir / f"{asset_stem}_plot.png",
            "plot_pdf": asset_dir / f"{asset_stem}_plot.pdf",
            "page_tex": asset_dir / f"{asset_stem}.tex",
            "page_pdf": asset_dir / f"{asset_stem}.pdf",
        }
        assets["plot_png"].write_bytes(b"png")
        assets["plot_pdf"].write_bytes(b"pdf")
        assets["page_tex"].write_text("Qiskit pending", encoding="utf-8")
        _pdf(assets["page_pdf"], ["new-live-page-seven"])
        return assets

    monkeypatch.setattr(subject, "build_assets", fake_assets)
    result = subject.update_page7(
        target_pdf=target_pdf,
        target_provenance=provenance_path,
        adapter_path=adapter_path,
        asset_dir=asset_dir,
        asset_stem="live_page7",
    )
    after = subject.completed.legacy_page._page_content_hashes(target_pdf)
    assert result["preserved_pages_1_6"] is True
    assert before[:6] == after[:6]
    assert before[6] != after[6]
    updated = json.loads(provenance_path.read_text(encoding="utf-8"))
    report = updated[subject.completed.REPORT_KEY]
    assert report["live_horizons"] == {
        "weak_strong": 36,
        "intermediate_strong": 32,
        "strong_strong_u8": 18,
    }
    live_rows = [row for row in report["cells"] if row["status"] == "live_partial"]
    assert len(live_rows) == 3
    assert all(row["qiskit_status"] == subject.LIVE_QISKIT_STATUS for row in live_rows)

    rollback_pdf = tmp_path / "rollback_report.pdf"
    _pdf(rollback_pdf, [f"rollback-page-{index}" for index in range(1, 8)])
    rollback_provenance_path = tmp_path / "rollback_report_provenance.json"
    rollback_provenance = copy.deepcopy(provenance)
    rollback_provenance["outputs"]["partial_progress_pdf"] = subject.file_binding(
        rollback_pdf
    )
    _write_json(rollback_provenance_path, rollback_provenance)
    original_pdf_bytes = rollback_pdf.read_bytes()
    original_provenance_bytes = rollback_provenance_path.read_bytes()
    real_replace = subject._publication_replace

    def fail_second_replace(source: Path, target: Path) -> None:
        if Path(target) == rollback_provenance_path:
            raise OSError("injected provenance replacement failure")
        real_replace(source, target)

    monkeypatch.setattr(subject, "_publication_replace", fail_second_replace)
    with pytest.raises(
        subject.LivePage7InputError, match="publication failed and was rolled back"
    ):
        subject.update_page7(
            target_pdf=rollback_pdf,
            target_provenance=rollback_provenance_path,
            adapter_path=adapter_path,
            asset_dir=asset_dir,
            asset_stem="rollback_live_page7",
        )
    assert rollback_pdf.read_bytes() == original_pdf_bytes
    assert rollback_provenance_path.read_bytes() == original_provenance_bytes
    assert not list(tmp_path.glob(".*.live-page7.*"))
