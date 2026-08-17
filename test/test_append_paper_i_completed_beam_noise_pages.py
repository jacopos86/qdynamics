from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipelines.reporting import append_paper_i_completed_beam_noise_pages as pages


def _write_pdf(path: Path, payloads: list[bytes]) -> None:
    pypdf = pytest.importorskip("pypdf")
    from pypdf.generic import DecodedStreamObject, NameObject

    writer = pypdf.PdfWriter()
    for index, payload in enumerate(payloads, 1):
        page = writer.add_blank_page(width=600 + index, height=800)
        stream = DecodedStreamObject()
        stream.set_data(payload)
        page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as output:
        writer.write(output)


def _content_hashes(path: Path) -> list[str]:
    pypdf = pytest.importorskip("pypdf")
    result = []
    for page in pypdf.PdfReader(str(path), strict=False).pages:
        contents = page.get_contents()
        payload = b"" if contents is None else contents.get_data()
        result.append(hashlib.sha256(payload).hexdigest())
    return result


def _noise_adapter_fixture() -> dict[str, object]:
    return {
        "status": "completed_6_of_6_mixed_horizon",
        "sha256": "b" * 64,
        "source_packages": {"low_high_r20": {}, "extreme_r50": {}},
        "terminal_horizon_by_noise_level": {
            "low": 20,
            "high": 20,
            "extreme": 50,
        },
        "pending_low_high_extension": {
            "status": "canceled_r30_not_required_after_completed_r20_review",
            "cluster_id": 9644468,
            "scheduler_disposition": "removed_20260812",
        },
        "cells": [{"result": {}}],
        "limitations": [],
    }


def test_corrected_beam_allowlist_closes_all_six_regimes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_specs = {
        "weak_weak": (
            0,
            290_332_669,
            "8d8e88d1942cb4ceb43f75f1705dd5d7171c0bc8bf991b5773c1f0426e2e4770",
        ),
        "intermediate_weak": (
            1,
            339_489_014,
            "2766ba8cd6a878d7a9fb77504d6133754a18d2f92c108c59426aafb7b83fb727",
        ),
        "strong_weak_u8": (
            2,
            265_280_978,
            "8dd2d2298240c0dd05083e5f81bc9516fa6dedf1cc065da405cc14b40abbfba2",
        ),
        "weak_strong": (
            3,
            303_811_642,
            "70cdb2b96b36c5fa8d617d58b7d43f588dbdcbd21efb7d3aa37e81f4ee572882",
        ),
        "intermediate_strong": (
            4,
            234_589_358,
            "69b76b4cfc25db24746a9bbbc58d4dc451b8104b2791f2a224654229b9a829b0",
        ),
        "strong_strong_u8": (
            5,
            275_161_207,
            "8d8e069987ec657e2f413223f2d44a83b931738c5aedb79f83a59154f9e64fa3",
        ),
    }
    assert pages.BEAM_CLUSTER_ID == 9638417
    assert pages.BEAM_PACKAGE_SHA256 == (
        "125f4857de21ce348c6ad8e437cd8b1a728bc6b4c1be780e749fc10361380658"
    )
    assert {
        regime: (
            int(spec["proc_id"]),
            int(spec["size_bytes"]),
            str(spec["sha256"]),
        )
        for regime, spec in pages.BEAM_ARCHIVES.items()
    } == expected_specs
    observed: list[tuple[Path, dict[str, object], int]] = []

    def fake_archive_result(
        *,
        path: Path,
        expected: dict[str, object],
        cluster_id: int,
        job_path: Path,
        job: dict[str, object],
    ) -> dict[str, object]:
        del job_path, job
        observed.append((path, expected, cluster_id))
        return {
            "terminal": {"k": 20, "error": 1.0e-6},
            "costs": {
                "N2q": 1,
                "D2q": 1,
                "Dc": 1,
                "W1q": 1,
                "S_alg": 1,
            },
            "compile": {"qiskit_basis_work_status": "ok"},
        }

    with monkeypatch.context() as scoped:
        scoped.setattr(pages, "_archive_result", fake_archive_result)
        beam = pages.build_beam_adapter()

    assert beam["status"] == "completed_6_of_6"
    assert beam["completed_regime_count"] == 6
    assert beam["pending_regime_count"] == 0
    assert beam["cluster_id"] == 9638417
    assert beam["supersedes"] == {
        "package_id": pages.SUPERSEDED_BEAM_PACKAGE_ID,
        "package_manifest_canonical_sha256": (
            pages.SUPERSEDED_BEAM_PACKAGE_SHA256
        ),
        "cluster_id": 9631689,
        "reason": "beam_pool_contraction_defect_corrected_in_v4",
        "prior_completed_archives_preserved": True,
        "prior_page_evidence_state": "superseded_defective_v3",
    }
    assert [row[2] for row in observed] == [9638417] * 6
    assert [row[0] for row in observed] == [
        pages.BEAM_RETRIEVED / pages.BEAM_ARCHIVES[regime]["filename"]
        for regime in pages.REGIME_ORDER
    ]
    assert [
        (
            int(row[1]["proc_id"]),
            int(row[1]["size_bytes"]),
            str(row[1]["sha256"]),
        )
        for row in observed
    ] == [
        (
            int(pages.BEAM_ARCHIVES[regime]["proc_id"]),
            int(pages.BEAM_ARCHIVES[regime]["size_bytes"]),
            str(pages.BEAM_ARCHIVES[regime]["sha256"]),
        )
        for regime in pages.REGIME_ORDER
    ]
    completed_beam = [
        cell["beam_metric_route"]
        for cell in beam["cells"]
        if cell["beam_metric_route"] is not None
    ]
    assert [row["terminal"]["k"] for row in completed_beam] == [20] * 6
    assert all(
        set(row["costs"]) == {"N2q", "D2q", "Dc", "W1q", "S_alg"}
        and row["compile"]["qiskit_basis_work_status"] == "ok"
        for row in completed_beam
    )


def test_completed_mixed_horizon_noise_archives_close_adapter() -> None:
    expected_specs = {
        ("u1p5", "low"): (
            0,
            20,
            48_822_613,
            "bda143927e4eab8e37f396b5ca9343831ad500c8d80e1ad5a8bb20b7a8bf721f",
        ),
        ("u1p5", "high"): (
            1,
            20,
            44_276_774,
            "c083cbfedea4ebe9a49acec25ca1b48f63c203f761a281b3211612466622c759",
        ),
        ("u1p5", "extreme"): (
            2,
            50,
            205_692_040,
            "b7e2d35e33e9c195f1ab7eb3282f10b873d217e2b4cb4d5345e3429b5f38b078",
        ),
        ("u8", "low"): (
            2,
            20,
            49_470_933,
            "8e204d137817e2295e1b67568f8b38399e687c631cccd36ab105f5a759606a6c",
        ),
        ("u8", "high"): (
            3,
            20,
            49_639_473,
            "c027d590bf146016b9db0373c15ede2aa4160abc592e49f77cc14e9cd28c44db",
        ),
        ("u8", "extreme"): (
            5,
            50,
            216_361_821,
            "e845fdafc56eefaf155923dc5de8c78921fcde46bbb8234c8f39619700ff6fba",
        ),
    }
    assert pages.NOISE_R20_CLUSTER_ID == 9636601
    assert pages.NOISE_R20_PACKAGE_SHA256 == (
        "494971753470c7a83093c849d4d35d7dba48424a3fe2ca61b9fbd8b136cd3a8b"
    )
    assert pages.PENDING_NOISE_R30_CLUSTER_ID == 9644468
    assert pages.PENDING_NOISE_R30_PACKAGE_SHA256 == (
        "0b30d0314caa44047cff1af850bc84b065c3a68d628346ffbad9cd2959214dce"
    )
    assert {
        key: (
            int(spec["proc_id"]),
            int(spec["target_horizon"]),
            int(spec["size_bytes"]),
            str(spec["sha256"]),
        )
        for key, spec in pages.NOISE_ARCHIVES.items()
    } == expected_specs
    for (u_key, level), spec in pages.NOISE_ARCHIVES.items():
        cluster_id = 9634547 if level == "extreme" else 9636601
        assert spec["filename"] == (
            f"pure_hubbard_page12_fullnoise__{u_key}__{level}__"
            f"{cluster_id}__{spec['proc_id']}.tar.gz"
        )

    noise = pages.build_noise_adapter()
    completed_noise = [
        cell["result"] for cell in noise["cells"] if cell["result"] is not None
    ]
    assert noise["status"] == "completed_6_of_6_mixed_horizon"
    assert [row["terminal"]["k"] for row in completed_noise] == [
        20,
        20,
        50,
        20,
        20,
        50,
    ]
    assert [
        completed_noise[index]["terminal"]["error"] for index in (2, 5)
    ] == pytest.approx([0.12781876434916817, 0.3902761877166218])
    assert [completed_noise[index]["costs"]["S_alg"] for index in (2, 5)] == [
        93_135,
        96_518,
    ]
    assert all(
        set(row["costs"]) == {"N2q", "D2q", "Dc", "W1q", "S_alg"}
        and row["compile"]["qiskit_basis_work_status"] == "ok"
        for row in completed_noise
    )
    assert noise["terminal_horizon_by_noise_level"] == {
        "low": 20,
        "high": 20,
        "extreme": 50,
    }
    assert noise["source_packages"]["low_high_r20"][
        "package_manifest_canonical_sha256"
    ] == pages.NOISE_R20_PACKAGE_SHA256
    assert noise["source_packages"]["extreme_r50"][
        "package_manifest_canonical_sha256"
    ] == pages.NOISE_R50_PACKAGE_SHA256
    assert noise["pending_low_high_extension"]["cluster_id"] == 9644468
    assert noise["pending_low_high_extension"]["status"] == (
        "canceled_r30_not_required_after_completed_r20_review"
    )
    assert noise["pending_low_high_extension"]["scheduler_disposition"] == (
        "removed_20260812"
    )
    assert noise["pending_low_high_extension"][
        "package_manifest_canonical_sha256"
    ] == pages.PENDING_NOISE_R30_PACKAGE_SHA256


def test_append_pages_preserves_first_thirteen_content_streams(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pypdf = pytest.importorskip("pypdf")
    target_pdf = tmp_path / "report.pdf"
    target_provenance = tmp_path / "report-provenance.json"
    page14_pdf = tmp_path / "page14.pdf"
    page15_pdf = tmp_path / "page15.pdf"
    page14_png = tmp_path / "page14.png"
    page15_png = tmp_path / "page15.png"
    page14_adapter = tmp_path / "page14-adapter.json"
    page15_adapter = tmp_path / "page15-adapter.json"
    _write_pdf(
        target_pdf,
        [f"q {index} 0 1 1 re f Q\n".encode() for index in range(1, 14)],
    )
    _write_pdf(page14_pdf, [b"q 14 0 1 1 re f Q\n"])
    _write_pdf(page15_pdf, [b"q 15 0 1 1 re f Q\n"])
    page14_png.write_bytes(b"page14")
    page15_png.write_bytes(b"page15")
    page14_adapter.write_text("{}\n", encoding="utf-8")
    page15_adapter.write_text("{}\n", encoding="utf-8")
    layout = {
        "page_count": 13,
        "page_13": (
            "macro_gradient_phase0_macro_phase123_proxy_no_lanes_partial_v1"
        ),
    }
    provenance = {
        "schema": "fixture",
        "layout": layout,
        "outputs": {"partial_progress_pdf": pages.binding(target_pdf)},
    }
    target_provenance.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name, value in (
        ("TARGET_PDF", target_pdf),
        ("TARGET_PROVENANCE", target_provenance),
        ("PAGE14_PDF", page14_pdf),
        ("PAGE15_PDF", page15_pdf),
        ("PAGE14_PNG", page14_png),
        ("PAGE15_PNG", page15_png),
        ("PAGE14_ADAPTER", page14_adapter),
        ("PAGE15_ADAPTER", page15_adapter),
    ):
        monkeypatch.setattr(pages, name, value)
    before = _content_hashes(target_pdf)
    beam = {
        "status": "partial",
        "sha256": "a" * 64,
        "cells": [{"beam_metric_route": {}}],
        "limitations": [],
        "supersedes": {"cluster_id": 9631689},
    }
    noise = _noise_adapter_fixture()

    result = pages.append_or_replace_pages(beam, noise, provenance)

    updated = json.loads(target_provenance.read_text(encoding="utf-8"))
    assert result["page_count"] == 15
    assert len(pypdf.PdfReader(str(target_pdf), strict=False).pages) == 15
    assert _content_hashes(target_pdf)[:13] == before
    assert updated["layout"]["page_14"] == pages.PAGE14_ID
    assert updated["layout"]["page_15"] == pages.PAGE15_ID
    assert updated["layout"]["page_count"] == 15
    assert updated["outputs"]["partial_progress_pdf"] == pages.binding(target_pdf)


def test_replace_pages_14_and_15_in_sixteen_page_report_preserves_page16(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pypdf = pytest.importorskip("pypdf")
    target_pdf = tmp_path / "report.pdf"
    target_provenance = tmp_path / "report-provenance.json"
    page14_pdf = tmp_path / "page14.pdf"
    page15_pdf = tmp_path / "page15.pdf"
    page14_png = tmp_path / "page14.png"
    page15_png = tmp_path / "page15.png"
    page14_adapter = tmp_path / "page14-adapter.json"
    page15_adapter = tmp_path / "page15-adapter.json"
    _write_pdf(
        target_pdf,
        [f"q {index} 0 1 1 re f Q\n".encode() for index in range(1, 17)],
    )
    _write_pdf(page14_pdf, [b"q 114 0 2 2 re f Q\n"])
    _write_pdf(page15_pdf, [b"q 115 0 2 2 re f Q\n"])
    page14_png.write_bytes(b"corrected page14")
    page15_png.write_bytes(b"mixed-horizon page15")
    page14_adapter.write_text("{}\n", encoding="utf-8")
    page15_adapter.write_text("{}\n", encoding="utf-8")
    prior_page15 = {"status": "unchanged-page15", "cells": []}
    provenance = {
        "schema": "fixture",
        "layout": {
            "page_count": 16,
            "page_13": (
                "macro_gradient_phase0_macro_phase123_proxy_no_lanes_"
                "partial_v1"
            ),
            "page_14": pages.PAGE14_ID,
            "page_15": pages.PAGE15_ID,
            "page_16": (
                "macro_gradient_phase0_macro_phase123_qiskit_phase23_"
                "no_lanes_partial_v1"
            ),
        },
        "macro_phase0_beam_metric_progress": {
            "status": "superseded-defective-v3"
        },
        "pure_hubbard_page12_noise_progress": prior_page15,
        "outputs": {"partial_progress_pdf": pages.binding(target_pdf)},
    }
    target_provenance.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name, value in (
        ("TARGET_PDF", target_pdf),
        ("TARGET_PROVENANCE", target_provenance),
        ("PAGE14_PDF", page14_pdf),
        ("PAGE15_PDF", page15_pdf),
        ("PAGE14_PNG", page14_png),
        ("PAGE15_PNG", page15_png),
        ("PAGE14_ADAPTER", page14_adapter),
        ("PAGE15_ADAPTER", page15_adapter),
    ):
        monkeypatch.setattr(pages, name, value)
    before = _content_hashes(target_pdf)
    beam = {
        "status": "completed_6_of_6",
        "sha256": "a" * 64,
        "cells": [{"beam_metric_route": {}} for _ in range(6)],
        "limitations": [],
        "supersedes": {
            "cluster_id": 9631689,
            "prior_page_evidence_state": "superseded_defective_v3",
        },
    }
    noise = _noise_adapter_fixture()

    result = pages.append_or_replace_pages(beam, noise, provenance)

    after = _content_hashes(target_pdf)
    updated = json.loads(target_provenance.read_text(encoding="utf-8"))
    assert result["page_count"] == 16
    assert len(pypdf.PdfReader(str(target_pdf), strict=False).pages) == 16
    assert after[:13] == before[:13]
    assert after[13] != before[13]
    assert after[14] != before[14]
    assert after[15:] == before[15:]
    assert updated["layout"] == provenance["layout"]
    assert updated["pure_hubbard_page12_noise_progress"]["status"] == (
        "completed_6_of_6_mixed_horizon"
    )
    assert updated["pure_hubbard_page12_noise_progress"][
        "terminal_horizon_by_noise_level"
    ] == noise["terminal_horizon_by_noise_level"]
    assert updated["pure_hubbard_page12_noise_progress"] != prior_page15
    assert updated["macro_phase0_beam_metric_progress"]["supersedes"] == (
        beam["supersedes"]
    )
    assert updated["outputs"]["partial_progress_pdf"] == pages.binding(
        target_pdf
    )
