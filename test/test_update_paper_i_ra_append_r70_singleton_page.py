from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from pipelines.reporting import (
    update_paper_i_ra_append_r70_singleton_page as page6,
)


def _attempt_archive(
    root: Path, *, regime: str, drift_worker_binding: bool = False
) -> Path:
    execution_id = page6.execution_id(regime)
    proc_id = page6.PROC_BY_REGIME[regime]
    attempt_identity = (
        f"{execution_id}\t{page6.RA_CLUSTER_ID}\t{proc_id}\t1\n"
    ).encode("ascii")
    worker_status = b"2\n"
    worker_files = (
        ("attempt_identity.tsv", attempt_identity),
        ("worker_exit_status.txt", worker_status),
    )
    authority = {
        "job.json": b'{"kind":"job"}\n',
        "execution_authorization.json": b'{"kind":"authorization"}\n',
        "activation_manifest.json": b'{"kind":"activation"}\n',
    }
    receipt = {
        "schema": page6.RA_ATTEMPT_SCHEMA,
        "execution_id": execution_id,
        "cluster_id": page6.RA_CLUSTER_ID,
        "proc_id": proc_id,
        "attempt_ordinal": 1,
        "worker_exit_status": 2,
        "job_file_sha256": hashlib.sha256(authority["job.json"]).hexdigest(),
        "authorization_file_sha256": hashlib.sha256(
            authority["execution_authorization.json"]
        ).hexdigest(),
        "activation_manifest_file_sha256": hashlib.sha256(
            authority["activation_manifest.json"]
        ).hexdigest(),
        "source_archive_sha256": "1" * 64,
        "image_sha256": "2" * 64,
        "worker_files": [
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            for relative, payload in worker_files
        ],
    }
    if drift_worker_binding:
        receipt["worker_files"][0]["sha256"] = "f" * 64
    receipt = page6.digested(receipt)
    members = {
        **{
            f"worker_outputs/{relative}": payload
            for relative, payload in worker_files
        },
        **{f"authority/{relative}": payload for relative, payload in authority.items()},
        "worker_attempt_receipt.json": page6.canonical_json_bytes(receipt) + b"\n",
    }
    output = root / (
        f"{execution_id}__cluster_{page6.RA_CLUSTER_ID}__proc_{proc_id}.tar.gz"
    )
    with tarfile.open(output, mode="w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return output


def test_parse_ra_stdout_recovers_rounds_zero_through_sixty_nine(
    tmp_path: Path,
) -> None:
    stdout = tmp_path / "worker.out"
    lines = [
        "AI_LOG "
        + json.dumps(
            {
                "event": "hardcoded_adapt_iter",
                "depth": depth,
                "energy": -(depth - 1) / 100.0,
                "best_op": f"generator-{depth}",
                "selected_position": depth - 1,
            }
        )
        for depth in range(1, 71)
    ]
    lines.append(
        json.dumps(
            {
                "status": "passed",
                "output_archive": "transfer/example.tar.gz",
                "output_archive_sha256": "a" * 64,
                "output_archive_size_bytes": 123,
                "worker_attempt_receipt_sha256": "b" * 64,
            }
        )
    )
    stdout.write_text("\n".join(lines) + "\n", encoding="utf-8")

    points, decisions, source = page6.parse_ra_stdout(
        stdout,
        regime="weak_weak",
        exact_energy=-1.0,
    )

    assert [point["round"] for point in points] == list(range(70))
    assert points[0] == {"round": 0, "energy": 0.0, "delta_e": 1.0}
    assert points[-1]["round"] == 69
    assert points[-1]["energy"] == pytest.approx(-0.69)
    assert decisions[-1]["accepted_round"] == 70
    assert source["attempt_packaging_result"]["status"] == "passed"


def test_attempt_archive_closes_worker_and_authority_bindings(
    tmp_path: Path,
) -> None:
    archive = _attempt_archive(tmp_path, regime="weak_weak")

    binding = page6.validate_attempt_archive(archive, regime="weak_weak")

    assert binding["sha256"] == page6.sha256_file(archive)
    assert binding["attempt_ordinal"] == 1


def test_attempt_archive_rejects_rehashed_receipt_with_false_worker_binding(
    tmp_path: Path,
) -> None:
    archive = _attempt_archive(
        tmp_path,
        regime="weak_weak",
        drift_worker_binding=True,
    )

    with pytest.raises(page6.Page6InputError, match="worker binding bytes drifted"):
        page6.validate_attempt_archive(archive, regime="weak_weak")


def test_cost_tuple_uses_compact_s_alg_notation() -> None:
    assert page6.format_costs(
        {"N2q": 12, "D2q": 8, "Dc": 31, "W1q": 17, "S_alg": 313_231}
    ) == r"$(12,8,31,17,3.1\mathrm{e}5)$"
    assert page6.format_costs(
        {"N2q": 12, "D2q": 8, "Dc": 31, "W1q": 17, "S_alg": None}
    ) == r"$(12,8,31,17,\text{--})$"


def test_build_assets_rejects_path_like_asset_stem(tmp_path: Path) -> None:
    with pytest.raises(page6.Page6InputError, match="safe filename component"):
        page6.build_assets(
            {}, cost_adapter={}, asset_dir=tmp_path, asset_stem="../escape"
        )


def _current_cost_inputs() -> tuple[Path, Path, dict[str, object]]:
    output = page6.REPO_ROOT / (
        "output/pdf/"
        "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving"
    )
    combined_path = output / (
        "paper_i_ra_adapt_stationary_core_full48_r50_20260728_"
        "evolving_ra_append_singleton_r70_page6_adapter.json"
    )
    cost_path = output / "paper_i_ra_append_singleton_r70_prefix_costs_v1.json"
    combined = page6.load_json(combined_path, label="combined fixture")
    return combined_path, cost_path, combined


def test_current_cost_adapter_closes_matched_selection_and_append_s_alg() -> None:
    combined_path, cost_path, combined = _current_cost_inputs()

    validated = page6.validate_cost_adapter(
        cost_path,
        combined=combined,
        combined_path=combined_path,
        combined_canonical_sha256=page6.verify_self_digest(
            combined, label="combined fixture"
        ),
    )

    cells = {cell["regime_id"]: cell for cell in validated["cells"]}
    assert cells["weak_weak"]["common_accuracy"]["ra"]["round"] == 37
    assert cells["weak_weak"]["common_accuracy"]["append"]["round"] == 27
    assert (
        cells["weak_weak"]["common_accuracy"]["append"]["costs"]["S_alg"]
        == 131_568
    )
    assert cells["strong_strong_u8"]["ra_round_69"]["costs"]["S_alg"] is None


def test_cost_adapter_rejects_fabricated_ra_s_alg(tmp_path: Path) -> None:
    combined_path, cost_path, combined = _current_cost_inputs()
    payload = page6.load_json(cost_path, label="cost fixture")
    payload.pop("sha256")
    payload["cells"][0]["ra_round_69"]["costs"]["S_alg"] = 1
    tampered = page6.digested(payload)
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_bytes(page6.canonical_json_bytes(tampered) + b"\n")

    with pytest.raises(page6.Page6InputError, match="RA round 69 must preserve"):
        page6.validate_cost_adapter(
            tampered_path,
            combined=combined,
            combined_path=combined_path,
            combined_canonical_sha256=page6.verify_self_digest(
                combined, label="combined fixture"
            ),
        )


def test_replace_page6_preserves_later_diagnostic_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pypdf import PdfReader, PdfWriter

    target_pdf = tmp_path / "report.pdf"
    writer = PdfWriter()
    for index in range(9):
        writer.add_blank_page(width=100 + index, height=200)
    with target_pdf.open("wb") as stream:
        writer.write(stream)

    page_pdf = tmp_path / "replacement.pdf"
    replacement = PdfWriter()
    replacement.add_blank_page(width=999, height=200)
    with page_pdf.open("wb") as stream:
        replacement.write(stream)
    plot_png = tmp_path / "plot.png"
    plot_pdf = tmp_path / "plot.pdf"
    page_tex = tmp_path / "page.tex"
    plot_png.write_bytes(b"plot")
    plot_pdf.write_bytes(b"pdf")
    page_tex.write_text("page", encoding="utf-8")
    monkeypatch.setattr(
        page6,
        "build_assets",
        lambda *_args, **_kwargs: {
            "plot_png": plot_png,
            "plot_pdf": plot_pdf,
            "page_tex": page_tex,
            "page_pdf": page_pdf,
        },
    )
    cost_path = tmp_path / "costs.json"
    cost_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        page6,
        "validate_cost_adapter",
        lambda *_args, **_kwargs: {
            "sha256": "c" * 64,
            "file_binding": page6.file_binding(cost_path),
            "cells": [],
        },
    )

    combined = page6.digested(
        {
            "schema": page6.COMBINED_ADAPTER_SCHEMA,
            "append_adapter": {"path": "append.json"},
            "limitations": [],
            "cells": [
                {
                    "regime_id": regime,
                    "append": {"endpoints": {"round_70": {"round": 70}}},
                    "ra_historical_average_plateau": {
                        "points": [{"round": 69}],
                        "source": {"kind": "stdout"},
                    },
                }
                for regime in page6.REGIME_ORDER
            ],
        }
    )
    combined_path = tmp_path / "combined.json"
    combined_path.write_bytes(page6.canonical_json_bytes(combined) + b"\n")
    provenance_path = tmp_path / "provenance.json"
    provenance = {
        "layout": {
            "page_count": 9,
            "page_6": page6.legacy_page.PAGE_ID,
        },
        "outputs": {"partial_progress_pdf": page6.file_binding(target_pdf)},
        "append_singleton_r70_progress": {"status": "current"},
        "limitations": [page6.OLD_PAGE_LIMITATION],
    }
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    result = page6.replace_page6(
        target_pdf=target_pdf,
        target_provenance=provenance_path,
        combined_adapter_path=combined_path,
        cost_adapter_path=cost_path,
        asset_dir=tmp_path,
        asset_stem="replacement",
    )

    pages = PdfReader(str(target_pdf), strict=False).pages
    assert result["pages"] == 9
    assert len(pages) == 9
    assert [float(page.mediabox.width) for page in pages] == [
        100,
        101,
        102,
        103,
        104,
        999,
        106,
        107,
        108,
    ]
    updated = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert updated["layout"]["page_count"] == 9
    assert updated["layout"]["page_6"] == page6.PAGE_ID
    assert len(
        updated["ra_append_singleton_r70_comparison"]
        ["structural_validation"]
        ["preserved_pages_7_onward_content_sha256"]
    ) == 3
