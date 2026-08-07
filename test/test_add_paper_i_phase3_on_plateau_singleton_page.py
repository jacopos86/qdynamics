from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile
from typing import Any, Mapping

import pytest

from pipelines.reporting import (
    add_paper_i_phase3_on_plateau_singleton_page as page8,
)


ACTIVATION_ROOT = page8.REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727"
)
CANARY = ACTIVATION_ROOT / (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_"
    "r50_20260803_v3_chtc_activation_canary_weak_strong_v1"
)
REMAINING = ACTIVATION_ROOT / (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_"
    "r50_20260803_v3_chtc_activation_remaining5_v1"
)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return page8.canonical_json_bytes(value) + b"\n"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _binding(raw: bytes) -> dict[str, Any]:
    return {"sha256": _sha(raw), "size_bytes": len(raw)}


def _authority_paths(regime: str) -> tuple[Path, Path, Path]:
    execution_id = page8.expected_execution_id(regime)
    activation = CANARY if regime == "weak_strong" else REMAINING
    return (
        page8.PACKAGE_DIR / "jobs" / f"{execution_id}.json",
        activation / "authorizations" / f"{execution_id}.json",
        activation / "activation_manifest.json",
    )


def _summary(job: Mapping[str, Any], *, qiskit: Mapping[str, int]) -> dict[str, Any]:
    exact = float(job["exact_same_cutoff_energy"])
    checkpoint = "a" * 64
    work = {
        "components": {
            "n_h_outer": 50,
            "n_h_refit": 75,
            "n_grad": 125,
            "n_metric": 250,
        },
        "s_alg": 500,
    }
    trace = []
    for round_index in range(1, 51):
        error = 1.0 / (round_index + 2.0)
        trace.append(
            {
                "controller_round": round_index,
                "active_ansatz_depth": 1,
                "accepted_energy": exact + error,
                "exact_same_cutoff_energy": exact,
                "absolute_energy_error": error,
                "projective_state_fingerprint": "projective_state_v1:test",
                "checkpoint_sha256": checkpoint,
            }
        )
    prefix = {
        "source_method": "sr_snake",
        "controller_round": 50,
        "active_ansatz_depth": 1,
        "ordered_operator_labels": ["guarded_singleton::y"],
        "operators": [
            {
                "candidate_label": "guarded_singleton::y",
                "logical_index": 0,
                "runtime_start": 0,
                "runtime_count": 1,
                "execution_mode": "termwise_product",
                "runtime_terms": [
                    {
                        "pauli_exyz": "y",
                        "coefficient_real": 1.0,
                        "coefficient_imaginary": 0.0,
                        "qubit_count": 1,
                    }
                ],
            }
        ],
        "logical_parameters": [0.25],
        "runtime_parameters": [0.25],
        "reference_state": {
            "amplitudes_real": [1.0, 0.0],
            "amplitudes_imaginary": [0.0, 0.0],
            "qubit_count": 1,
            "source_label": "synthetic",
            "state_fingerprint": "projective_state_v1:test",
        },
        "checkpoint_sha256": checkpoint,
        "projective_state_fingerprint": "projective_state_v1:test-prefix",
        "problem_request_sha256": "b" * 64,
        "route_profile": page8.ROUTE_PROFILE,
        "route_contract_sha256": page8.ROUTE_CONTRACT_SHA256,
        "algorithmic_work": work,
    }
    return {
        "schema": page8.SUMMARY_SCHEMA,
        "available_controller_rounds": 50,
        "horizon_scope": "deliberately_stopped_prefix",
        "accepted_error_trace": trace,
        "canonical_all_work": work,
        "effective_plateau": {
            "policy": "paper_i_effective_plateau_v1",
            "controller_round": 50,
            "active_ansatz_depth": 1,
            "absolute_energy_error": trace[-1]["absolute_energy_error"],
            "best_observed_error": trace[-1]["absolute_energy_error"],
            "available_horizon_controller_rounds": 50,
            "horizon_scope": "deliberately_stopped_prefix",
            "algorithmic_work": work,
            "prefix": prefix,
            "failure": None,
            "status": "available",
        },
        "requested_rounds": [
            {
                "purpose": "requested_controller_round",
                "status": "available",
                "controller_round": 50,
                "active_ansatz_depth": 1,
                "absolute_energy_error": trace[-1]["absolute_energy_error"],
                "algorithmic_work": work,
                "prefix": prefix,
                "resources": {
                    "compile_convention": page8.COMPILE_CONVENTION,
                    "compiled_two_qubit_count": qiskit["N2q"],
                    "compiled_two_qubit_depth": qiskit["D2q"],
                    "compiled_total_depth": qiskit["Dc"],
                },
                "failure": None,
            }
        ],
        "append_matched": {"status": "unavailable"},
        "provenance": {
            "candidate_representation": "single_pauli_word_v1",
            "exact_same_cutoff_energy": exact,
            "qiskit_compile_convention": page8.COMPILE_CONVENTION,
            "optimizer": "POWELL",
            "optimizer_maxiter": 200,
            "route_contract_sha256": page8.ROUTE_CONTRACT_SHA256,
            "route_profile": page8.ROUTE_PROFILE,
            "seed": 7,
        },
    }


def _build_attempt(
    root: Path,
    *,
    regime: str,
    proc_id: int,
    omit_member: str | None = None,
) -> Path:
    execution_id = page8.expected_execution_id(regime)
    job_path, authorization_path, activation_path = _authority_paths(regime)
    job_raw = job_path.read_bytes()
    authorization_raw = authorization_path.read_bytes()
    activation_raw = activation_path.read_bytes()
    job = json.loads(job_raw)
    authorization = json.loads(authorization_raw)
    qiskit = {"N2q": 10 + proc_id, "D2q": 20 + proc_id, "Dc": 30 + proc_id}
    summary_raw = _json_bytes(_summary(job, qiskit=qiskit))
    exact = float(job["exact_same_cutoff_energy"])
    result_raw = _json_bytes(
        {
            "run": {
                "accepted_transitions": [
                    {"energy_before": exact + 2.0, "controller_round": 1}
                ]
            },
            "schema": page8.RESULT_SCHEMA,
        }
    )
    estimator_sidecar_raw = _json_bytes(
        {"schema": "paper_i_estimator_call_ledger_checkpoint_sidecar_v2"}
    )
    estimator_sidecar = (
        "checkpoint.estimator_call_ledger_checkpoint."
        f"{_sha(estimator_sidecar_raw)[:16]}.json"
    )
    resume_sidecar_raw = _json_bytes(
        {"schema": "static_adapt_signed_active_prefix_resume_sidecar_v2"}
    )
    resume_sidecar = (
        "checkpoint.verified_singleton_resume."
        f"{_sha(resume_sidecar_raw)[:16]}.json"
    )
    checkpoint_raw = _json_bytes(
        {
            "sidecars": [
                {"path": estimator_sidecar, "sha256": _sha(estimator_sidecar_raw)},
                {"path": resume_sidecar, "sha256": _sha(resume_sidecar_raw)},
            ]
        }
    )
    ledger_raw = _json_bytes(
        {
            "schema": "paper_i_estimator_call_ledger_sidecar_v2",
            "adapt_success": True,
            "adapt_error": None,
            "accounting": {
                "complete": True,
                "exact_blockers": [],
                "components": {
                    "N_H_outer": 50,
                    "N_H_refit": 75,
                    "N_grad": 125,
                    "N_metric": 250,
                },
                "S_alg": 500,
            },
        }
    )
    artifacts = {
        "checkpoint.json": checkpoint_raw,
        "estimator_ledger.json": ledger_raw,
        "paper_i_summary.json": summary_raw,
        "result.json": result_raw,
        estimator_sidecar: estimator_sidecar_raw,
        resume_sidecar: resume_sidecar_raw,
    }
    execution_manifest = page8.digested(
        {
            "schema": page8.EXECUTION_MANIFEST_SCHEMA,
            "status": "passed",
            "package_id": page8.PACKAGE_ID,
            "campaign_id": page8.CAMPAIGN_ID,
            "execution_id": execution_id,
            "job_spec_sha256": job["sha256"],
            "authorization_sha256": authorization["sha256"],
            "protocol_sha256": job["protocol_sha256"],
            "target_horizon": 50,
            "controller_rounds_completed": 50,
            "fresh_start": True,
            "source_checkpoint_consumed": False,
            "worker_owned_live_progress": True,
            "same_filesystem_atomic_success_publication": True,
            "output_payloads": {
                name: _binding(raw) for name, raw in sorted(artifacts.items())
            },
        }
    )
    artifacts["execution_manifest.json"] = _json_bytes(execution_manifest)
    worker_receipt = page8.digested(
        {
            "schema": page8.WORKER_RECEIPT_SCHEMA,
            "status": "passed",
            "package_id": page8.PACKAGE_ID,
            "campaign_id": page8.CAMPAIGN_ID,
            "execution_id": execution_id,
            "job_spec_sha256": job["sha256"],
            "authorization_sha256": authorization["sha256"],
            "execution_manifest_sha256": execution_manifest["sha256"],
            "controller_rounds_completed": 50,
            "fresh_start": True,
            "artifacts": [
                {"path": name, **_binding(raw)}
                for name, raw in sorted(artifacts.items())
            ],
        }
    )
    worker_files = {
        **{f"artifacts/{name}": raw for name, raw in artifacts.items()},
        "worker_receipt.json": _json_bytes(worker_receipt),
    }
    attempt_receipt = page8.digested(
        {
            "schema": page8.ATTEMPT_SCHEMA,
            "execution_id": execution_id,
            "cluster_id": 9463745 if regime == "weak_strong" else 9463747,
            "proc_id": proc_id,
            "attempt_ordinal": 1,
            "worker_exit_status": 0,
            "job_file_sha256": _sha(job_raw),
            "authorization_file_sha256": _sha(authorization_raw),
            "activation_manifest_file_sha256": _sha(activation_raw),
            "source_archive_sha256": authorization["source_archive_sha256"],
            "image_sha256": authorization["remote_image_sha256"],
            "science_evidence_state": "success_payload_closed_v2",
            "worker_files": [
                {"path": name, **_binding(raw)}
                for name, raw in sorted(worker_files.items())
            ],
        }
    )
    members = {
        **{f"worker_outputs/{name}": raw for name, raw in worker_files.items()},
        "authority/job.json": job_raw,
        "authority/execution_authorization.json": authorization_raw,
        "authority/activation_manifest.json": activation_raw,
        "worker_attempt_receipt.json": _json_bytes(attempt_receipt),
    }
    if omit_member is not None:
        members.pop(omit_member)
    output = root / f"{execution_id}.tar.gz"
    with tarfile.open(output, "w:gz") as archive:
        for name, raw in sorted(members.items()):
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            info.mode = 0o644
            info.mtime = 0
            archive.addfile(info, io.BytesIO(raw))
    return output


def _compiler(prefix: Mapping[str, Any]) -> Mapping[str, Any]:
    assert prefix["controller_round"] == 50
    # The synthetic summaries use the same three values in every call site.
    # The per-archive values are recovered from the requested row by the
    # closure check, so tests build one matching compiler per invocation.
    return {
        "compile_convention": page8.COMPILE_CONVENTION,
        "compiled_count_2q_total": 10,
        "compiled_depth_2q_total": 20,
        "compiled_depth_total": 30,
        "qiskit_pretranspile_pauli_1q_work_total": 40,
        "qiskit_pretranspile_basis_change_1q_total": 15,
        "qiskit_basis_work_status": "ok",
        "qiskit_basis_work_schema": "qiskit_pretranspile_pauli_basis_work_v1",
    }


def _compiler_for_proc(proc_id: int):
    def compile_prefix(prefix: Mapping[str, Any]) -> Mapping[str, Any]:
        assert prefix["route_contract_sha256"] == page8.ROUTE_CONTRACT_SHA256
        return {
            "compile_convention": page8.COMPILE_CONVENTION,
            "compiled_count_2q_total": 10 + proc_id,
            "compiled_depth_2q_total": 20 + proc_id,
            "compiled_depth_total": 30 + proc_id,
            "qiskit_pretranspile_pauli_1q_work_total": 40 + proc_id,
            "qiskit_pretranspile_basis_change_1q_total": 15 + proc_id,
            "qiskit_basis_work_status": "ok",
            "qiskit_basis_work_schema": "qiskit_pretranspile_pauli_basis_work_v1",
        }

    return compile_prefix


def _build_append_adapter(path: Path) -> Path:
    cells = []
    for index, regime in enumerate(page8.REGIME_ORDER):
        job_path, _, _ = _authority_paths(regime)
        job = json.loads(job_path.read_text(encoding="utf-8"))
        exact = float(job["exact_same_cutoff_energy"])
        points = [
            {
                "round": round_index,
                "energy": exact + 1.0 / (round_index + 2.0),
                "delta_e": 1.0 / (round_index + 2.0),
            }
            for round_index in range(71)
        ]

        def endpoint(round_index: int) -> dict[str, Any]:
            point = points[round_index]
            return {
                "round": round_index,
                "energy": point["energy"],
                "delta_e": point["delta_e"],
                "costs": {
                    "N2q": 100 + index + round_index,
                    "D2q": 200 + index + round_index,
                    "Dc": 300 + index + round_index,
                    "W1q": 400 + index + round_index,
                    "S_alg": 500_000 + index + round_index,
                },
                "compile": {
                    "compile_convention": page8.COMPILE_CONVENTION,
                },
            }

        cells.append(
            {
                "regime_id": regime,
                "display_name": page8.REGIME_LABELS[regime],
                "nph": page8.NPH_BY_REGIME[regime],
                "execution_id": f"append__{regime}",
                "exact_same_cutoff_energy": exact,
                "points": points,
                "endpoints": {
                    "round_50": endpoint(50),
                    "round_70": endpoint(70),
                },
                "source": {"synthetic": True},
            }
        )
    adapter = page8.digested(
        {
            "schema": page8.APPEND_ADAPTER_SCHEMA,
            "status": "passed",
            "classification": "diagnostic_not_paper_evidence",
            "package_id": page8.APPEND_PACKAGE_ID,
            "regime_order": list(page8.REGIME_ORDER),
            "completed_regimes": list(page8.REGIME_ORDER),
            "pending_regimes": [],
            "source_authentication_summary": {"synthetic": True},
            "limitations": [],
            "cells": cells,
        }
    )
    page8._atomic_write_json(path, adapter)
    return path


def test_validates_exact_v3_attempt_and_projects_trajectory_cost_and_sources(
    tmp_path: Path,
) -> None:
    archive = _build_attempt(tmp_path, regime="weak_weak", proc_id=0)
    cell = page8.validate_attempt_archive(
        archive,
        regime="weak_weak",
        compiler=_compiler_for_proc(0),
    )
    assert cell["execution_id"] == page8.expected_execution_id("weak_weak")
    assert len(cell["points"]) == 51
    assert cell["points"][0] == {"k": 0, "error": 2.0}
    assert cell["points"][-1]["k"] == 50
    assert cell["terminal"] == {
        "k": 50,
        "error": pytest.approx(1.0 / 52.0),
        "N2q": 10,
        "D2q": 20,
        "Dc": 30,
        "W1q": 40,
        "B1q": 15,
        "compile_convention": page8.COMPILE_CONVENTION,
        "qiskit_basis_work_status": "ok",
        "qiskit_basis_work_schema": "qiskit_pretranspile_pauli_basis_work_v1",
        "qiskit_version": None,
        "generator_coefficients_sha256": None,
        "S_alg": 500,
        "status": "complete",
    }
    assert cell["source_bindings"]["archive"]["sha256"] == page8.sha256_file(
        archive
    )
    assert cell["source_bindings"]["worker_attempt_receipt"][
        "canonical_sha256"
    ]
    assert cell["source_bindings"]["execution_manifest"]["canonical_sha256"]


def test_rejects_archive_missing_a_receipted_member(tmp_path: Path) -> None:
    archive = _build_attempt(
        tmp_path,
        regime="weak_weak",
        proc_id=0,
        omit_member="worker_outputs/artifacts/paper_i_summary.json",
    )
    with pytest.raises(page8.Page8InputError, match="member closure"):
        page8.validate_attempt_archive(
            archive,
            regime="weak_weak",
            compiler=_compiler_for_proc(0),
        )


def _blank_pdf(path: Path, pages: int) -> None:
    from pypdf import PdfWriter

    writer = PdfWriter()
    for index in range(pages):
        page = writer.add_blank_page(width=612, height=792)
        # Give each synthetic page a stable distinct metadata-independent box.
        page.mediabox.upper_right = (612 + index, 792)
    with path.open("wb") as stream:
        writer.write(stream)


def test_appends_one_page_and_updates_provenance_without_changing_prior_pages(
    tmp_path: Path,
) -> None:
    attempts = {
        regime: _build_attempt(tmp_path, regime=regime, proc_id=index)
        for index, regime in enumerate(page8.REGIME_ORDER)
    }

    def compiler(prefix: Mapping[str, Any]) -> Mapping[str, Any]:
        # All prefixes are structurally equivalent. Match the serialized value
        # by compiling each archive separately below through an indexed shim.
        raise AssertionError("per-cell compiler shim must be installed")

    package = page8._load_package_authority(page8.PACKAGE_DIR)
    cells = [
        page8.validate_attempt_archive(
            attempts[regime],
            regime=regime,
            package=package,
            compiler=_compiler_for_proc(index),
        )
        for index, regime in enumerate(page8.REGIME_ORDER)
    ]
    adapter = page8.digested(
        {
            "schema": page8.BASE_ADAPTER_SCHEMA,
            "status": "passed_six_completed_cells",
            "classification": "supplemental_candidate_diagnostic_not_adopted_evidence",
            "paper_evidence_adopted": False,
            "page_id": page8.BASE_PAGE_ID,
            "package_id": page8.PACKAGE_ID,
            "campaign_id": page8.CAMPAIGN_ID,
            "package_manifest": package["manifest_binding"],
            "regime_order": list(page8.REGIME_ORDER),
            "completed_regimes": list(page8.REGIME_ORDER),
            "candidate_representation": "single_pauli_word_v1",
            "active_gradient_policy": "stationary_source_response_v1",
            "resource_weighting_scope": "late_resource_weighting_v1",
            "insertion_policy": "plateau_commutation",
            "phase3_population_activation": "same_round_authenticated_insertion_plateau_domain_open_v1",
            "plateau_prior_mean_decrease_ratio_threshold": page8.PLATEAU_RATIO,
            "plateau_threshold_comparison": page8.PLATEAU_COMPARISON,
            "plateau_trigger_source": page8.PLATEAU_TRIGGER,
            "route_contract_sha256": page8.ROUTE_CONTRACT_SHA256,
            "route_profile": page8.ROUTE_PROFILE,
            "target_controller_rounds": 50,
            "error_metric": "same_cutoff_absolute_energy_error",
            "cost_tuple": ["N2q", "D2q", "Dc", "W1q", "S_alg"],
            "cost_round": 50,
            "compile_convention": page8.COMPILE_CONVENTION,
            "cells": cells,
        }
    )
    adapter_path = tmp_path / "adapter.json"
    page8._atomic_write_json(adapter_path, adapter)
    page8.attach_append_comparator(
        ra_adapter_path=adapter_path,
        append_adapter_path=_build_append_adapter(tmp_path / "append.json"),
        output=adapter_path,
    )
    target = tmp_path / "report.pdf"
    _blank_pdf(target, 7)
    before_hashes = page8._page_content_hashes(target)
    provenance_path = tmp_path / "report_provenance.json"
    provenance = {
        "schema": "synthetic_existing_report_v1",
        "layout": {"page_count": 7, "page_7": "existing_page_7"},
        "outputs": {"partial_progress_pdf": page8.file_binding(target)},
    }
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    page_pdf = tmp_path / "page8.pdf"
    _blank_pdf(page_pdf, 1)
    plot_pdf = tmp_path / "plot.pdf"
    _blank_pdf(plot_pdf, 1)
    plot_png = tmp_path / "plot.png"
    plot_png.write_bytes(b"synthetic png")
    page_tex = tmp_path / "page8.tex"
    page_tex.write_text("synthetic tex", encoding="utf-8")
    assets = {
        "plot_png": plot_png,
        "plot_pdf": plot_pdf,
        "page_tex": page_tex,
        "page_pdf": page_pdf,
    }
    result = page8.append_page8(
        target_pdf=target,
        target_provenance=provenance_path,
        adapter_path=adapter_path,
        assets=assets,
    )
    assert result["status"] == "appended_page_8"
    assert page8._page_content_hashes(target)[:7] == before_hashes
    assert len(page8._page_content_hashes(target)) == 8
    updated = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert updated["layout"]["page_count"] == 8
    assert updated["layout"]["page_8"] == page8.PAGE_ID
    report = updated[page8.REPORT_KEY]
    assert len(report["cells"]) == 6
    assert report["structural_validation"]["preserved_page_content_sha256"] == before_hashes
    assert report["cells"][0]["terminal"]["S_alg"] == 500
    assert report["cells"][0]["append_adapt"]["terminal"]["S_alg"] == 500_050
    assert len(report["cells"][0]["append_adapt"]["points"]) == 51
    assert report["cells"][0]["source_bindings"]["archive"]["sha256"]


def test_append_comparator_is_cropped_to_round_50_with_common_cost_tuple(
    tmp_path: Path,
) -> None:
    attempts = {
        regime: _build_attempt(tmp_path, regime=regime, proc_id=index)
        for index, regime in enumerate(page8.REGIME_ORDER)
    }
    package = page8._load_package_authority(page8.PACKAGE_DIR)
    cells = [
        page8.validate_attempt_archive(
            attempts[regime],
            regime=regime,
            package=package,
            compiler=_compiler_for_proc(index),
        )
        for index, regime in enumerate(page8.REGIME_ORDER)
    ]
    base = page8.digested(
        {
            "schema": page8.BASE_ADAPTER_SCHEMA,
            "status": "passed_six_completed_cells",
            "classification": "supplemental_candidate_diagnostic_not_adopted_evidence",
            "paper_evidence_adopted": False,
            "page_id": page8.BASE_PAGE_ID,
            "package_id": page8.PACKAGE_ID,
            "campaign_id": page8.CAMPAIGN_ID,
            "package_manifest": package["manifest_binding"],
            "regime_order": list(page8.REGIME_ORDER),
            "completed_regimes": list(page8.REGIME_ORDER),
            "candidate_representation": "single_pauli_word_v1",
            "active_gradient_policy": "stationary_source_response_v1",
            "resource_weighting_scope": "late_resource_weighting_v1",
            "insertion_policy": "plateau_commutation",
            "phase3_population_activation": "same_round_authenticated_insertion_plateau_domain_open_v1",
            "plateau_prior_mean_decrease_ratio_threshold": page8.PLATEAU_RATIO,
            "plateau_threshold_comparison": page8.PLATEAU_COMPARISON,
            "plateau_trigger_source": page8.PLATEAU_TRIGGER,
            "route_contract_sha256": page8.ROUTE_CONTRACT_SHA256,
            "route_profile": page8.ROUTE_PROFILE,
            "target_controller_rounds": 50,
            "error_metric": "same_cutoff_absolute_energy_error",
            "cost_tuple": ["N2q", "D2q", "Dc", "W1q", "S_alg"],
            "cost_round": 50,
            "compile_convention": page8.COMPILE_CONVENTION,
            "cells": cells,
        }
    )
    base_path = tmp_path / "base.json"
    page8._atomic_write_json(base_path, base)
    comparison = page8.attach_append_comparator(
        ra_adapter_path=base_path,
        append_adapter_path=_build_append_adapter(tmp_path / "append.json"),
        output=tmp_path / "comparison.json",
    )
    first = comparison["cells"][0]["append_adapt"]
    assert [point["k"] for point in first["points"]] == list(range(51))
    assert first["terminal"] == {
        "k": 50,
        "error": pytest.approx(1.0 / 52.0),
        "N2q": 150,
        "D2q": 250,
        "Dc": 350,
        "W1q": 450,
        "S_alg": 500_050,
        "compile_convention": page8.COMPILE_CONVENTION,
    }
    assert comparison["comparison_round"] == 50


def test_build_adapter_requires_all_six_regimes(tmp_path: Path) -> None:
    attempts = {
        "weak_weak": _build_attempt(tmp_path, regime="weak_weak", proc_id=0)
    }
    with pytest.raises(page8.Page8InputError, match="every regime"):
        page8.build_adapter(
            attempts,
            output=tmp_path / "adapter.json",
            compiler=_compiler,
        )


def test_result_stream_accepts_schema_before_large_trailing_payload() -> None:
    payload = page8.canonical_json_bytes(
        {
            "run": {
                "accepted_transitions": [
                    {"energy_before": 1.25},
                ]
            },
            "schema": page8.RESULT_SCHEMA,
            "scientific_receipts": {"padding": "x" * 16_384},
        }
    )
    reader = page8._DigestingReader(io.BytesIO(payload))

    assert page8._result_initial_energy(reader) == pytest.approx(1.25)
    assert reader.size == len(payload)


def test_formats_five_tuple_with_compact_s_alg() -> None:
    assert page8._format_cost(
        {"N2q": 101, "D2q": 202, "Dc": 303, "W1q": 404, "S_alg": 273_870}
    ) == "(101, 202, 303, 404, 2.7e5)"
