from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile
from typing import Any

import pytest

from pipelines.reporting import ingest_paper_i_phase0_completed_archive as ingest
from pipelines.static_adapt.estimator_call_ledger import (
    projective_state_fingerprint,
)


def _bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    package = tmp_path / "package"
    jobs = package / "jobs"
    jobs.mkdir(parents=True)
    execution_id = "phase0__weak_weak__nph3__fixture"
    job = ingest.digested(
        {
            "package_id": "phase0-package",
            "campaign_id": "phase0-campaign",
            "execution_id": execution_id,
            "target_horizon": 50,
            "candidate_representation": "single_pauli_word_v1",
            "route_contract_sha256": ingest.ROUTE_CONTRACT_SHA256,
            "regime_id": "weak_weak",
            "nph": 3,
        }
    )
    job_path = jobs / f"{execution_id}.json"
    job_path.write_bytes(_bytes(job) + b"\n")
    package.joinpath("queue.tsv").write_text(
        "\t".join(
            (
                execution_id,
                f"jobs/{execution_id}.json",
                f"protocols/{execution_id}.json",
                hashlib.sha256(job_path.read_bytes()).hexdigest(),
                "4",
                "1024",
                "2048",
                "3600",
            )
        )
        + "\n"
    )

    exact = -1.0
    trace = []
    for controller_round in range(1, 51):
        energy = exact + 1.0 / (controller_round + 1)
        trace.append(
            {
                "controller_round": controller_round,
                "accepted_energy": energy,
                "absolute_energy_error": abs(energy - exact),
                "exact_same_cutoff_energy": exact,
            }
        )
    work = {
        "components": {
            "n_h_outer": 50,
            "n_h_refit": 20,
            "n_grad": 300,
            "n_metric": 100,
        },
        "s_alg": 470,
    }
    reference = (1.0 + 0.0j, 0.0 + 0.0j)
    prefix = {
        "source_method": "ra_adapt",
        "controller_round": 50,
        "active_ansatz_depth": 1,
        "ordered_operator_labels": ["x"],
        "operators": [
            {
                "candidate_label": "x",
                "logical_index": 0,
                "runtime_start": 0,
                "runtime_count": 1,
                "execution_mode": "termwise_product",
                "runtime_terms": [
                    {
                        "pauli_exyz": "x",
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
            "source_label": "fixture",
            "state_fingerprint": projective_state_fingerprint(reference),
        },
        "checkpoint_sha256": "1" * 64,
        "projective_state_fingerprint": "fixture-projective-state",
        "problem_request_sha256": "2" * 64,
        "route_profile": "fixture-route",
        "route_contract_sha256": ingest.ROUTE_CONTRACT_SHA256,
        "algorithmic_work": work,
    }
    summary = {
        "schema": "paper_i_run_summary_v1",
        "available_controller_rounds": 50,
        "accepted_error_trace": trace,
        "canonical_all_work": work,
        "provenance": {
            "route_contract_sha256": ingest.ROUTE_CONTRACT_SHA256,
            "candidate_representation": "single_pauli_word_v1",
            "qiskit_compile_convention": ingest.COMPILE_CONVENTION,
            "exact_same_cutoff_energy": exact,
        },
        "requested_rounds": [
            {
                "controller_round": 50,
                "absolute_energy_error": trace[-1]["absolute_energy_error"],
                "algorithmic_work": work,
                "status": "available",
                "failure": None,
                "resources": {
                    "compile_convention": ingest.COMPILE_CONVENTION,
                    "compiled_two_qubit_count": 12,
                    "compiled_two_qubit_depth": 8,
                    "compiled_total_depth": 31,
                },
                "prefix": prefix,
            }
        ],
    }
    summary_payload = _bytes(summary)
    summary_relative = f"runs/{execution_id}/summary/summary.json"
    manifest = ingest.digested(
        {
            "schema": (
                "paper_i_ra_adapt_gradient_phase0_phase23_qiskit_"
                "execution_manifest_v1"
            ),
            "status": "passed",
            "package_id": job["package_id"],
            "campaign_id": job["campaign_id"],
            "execution_id": execution_id,
            "job_spec_sha256": job["sha256"],
            "route_contract_sha256": ingest.ROUTE_CONTRACT_SHA256,
            "target_horizon": 50,
            "controller_rounds_completed": 50,
            "output_payloads": {
                "summary": {
                    "path": summary_relative,
                    "sha256": hashlib.sha256(summary_payload).hexdigest(),
                    "size_bytes": len(summary_payload),
                }
            },
        }
    )
    manifest_payload = _bytes(manifest)
    manifest_relative = f"runs/{execution_id}/execution_manifest.json"
    worker = ingest.digested(
        {
            "schema": (
                "paper_i_ra_adapt_gradient_phase0_phase23_qiskit_"
                "worker_receipt_v1"
            ),
            "status": "passed",
            "package_id": job["package_id"],
            "campaign_id": job["campaign_id"],
            "execution_id": execution_id,
            "job_spec_sha256": job["sha256"],
            "controller_rounds_completed": 50,
            "execution_manifest_sha256": manifest["sha256"],
            "artifacts": [
                {
                    "path": manifest_relative,
                    "sha256": hashlib.sha256(manifest_payload).hexdigest(),
                    "size_bytes": len(manifest_payload),
                },
                {
                    "path": summary_relative,
                    "sha256": hashlib.sha256(summary_payload).hexdigest(),
                    "size_bytes": len(summary_payload),
                },
            ],
        }
    )
    archive = tmp_path / "9605157.0_full.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name, payload in (
            ("./worker_receipt.json", _bytes(worker)),
            (f"./{manifest_relative}", manifest_payload),
            (f"./{summary_relative}", summary_payload),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))
    remote = {
        "path": f"/staging/jsstrobel/{archive.name}",
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "size_bytes": archive.stat().st_size,
    }
    return archive, package, remote


def _compiler(_prefix: Any) -> dict[str, Any]:
    return {
        "compile_convention": ingest.COMPILE_CONVENTION,
        "compiled_count_2q_total": 12,
        "compiled_depth_2q_total": 8,
        "compiled_depth_total": 31,
        "qiskit_pretranspile_pauli_1q_work_total": 17,
        "qiskit_pretranspile_basis_change_1q_total": 16,
        "qiskit_basis_work_status": "ok",
        "qiskit_transpile_optimization_level": 0,
        "qiskit_transpile_seed": 7,
        "qiskit_version": "fixture",
    }


def test_authenticated_archive_emits_page12_adapter_and_retrieval_receipt(
    tmp_path: Path,
) -> None:
    archive, package, remote = _fixture(tmp_path)

    adapter, receipt = ingest.build_outputs(
        archive_path=archive,
        cluster_id=9605157,
        proc_id=0,
        remote_archive=remote,
        package_dir=package,
        retrieved_utc="2026-08-09T20:00:00Z",
        compiler=_compiler,
    )

    ingest.verify_self_digest(adapter, label="adapter")
    ingest.verify_self_digest(receipt, label="retrieval receipt")
    assert [row["k"] for row in adapter["points"]] == list(range(1, 51))
    assert adapter["terminal"]["costs"] == {
        "N2q": 12,
        "D2q": 8,
        "Dc": 31,
        "W1q": 17,
        "S_alg": 470,
    }
    assert adapter["source"]["full_archive"] == remote
    assert receipt["byte_identity_passed"] is True
    assert receipt["local_archive"]["full_tar_inventory_scan_passed"] is True


def test_authenticated_archive_rejects_remote_local_byte_drift(
    tmp_path: Path,
) -> None:
    archive, package, remote = _fixture(tmp_path)
    remote["sha256"] = "0" * 64

    with pytest.raises(ingest.IngestError, match="SHA-256 differs"):
        ingest.build_outputs(
            archive_path=archive,
            cluster_id=9605157,
            proc_id=0,
            remote_archive=remote,
            package_dir=package,
            compiler=_compiler,
        )


def test_authenticated_archive_rejects_compiler_summary_disagreement(
    tmp_path: Path,
) -> None:
    archive, package, remote = _fixture(tmp_path)

    def drifted(prefix: Any) -> dict[str, Any]:
        payload = _compiler(prefix)
        payload["compiled_count_2q_total"] = 13
        return payload

    with pytest.raises(ingest.IngestError, match="serialized Qiskit triplet"):
        ingest.build_outputs(
            archive_path=archive,
            cluster_id=9605157,
            proc_id=0,
            remote_archive=remote,
            package_dir=package,
            compiler=drifted,
        )
