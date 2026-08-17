from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile
from typing import Any

import pytest

from pipelines.reporting import append_paper_i_phase0_route_pages as report
from pipelines.reporting import (
    ingest_paper_i_page12_r70_continuation_archive as ingest,
)


def _bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _fixture(
    tmp_path: Path, *, drift_first_50: bool = False
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    package = ingest.DEFAULT_PACKAGE_DIR
    queue = package.joinpath("queue.tsv").read_text().splitlines()[0].split("\t")
    execution_id, job_relative, _protocol, authorization_relative, *_ = queue
    job = json.loads(package.joinpath(job_relative).read_text())
    authorization = json.loads(package.joinpath(authorization_relative).read_text())
    base_path = ingest.BASE_COMPLETED_DIR / (
        ingest.BASE_COMPLETED_ADAPTERS["weak_strong"]
    )
    base = json.loads(base_path.read_text())
    exact = float(base["exact_same_cutoff_energy"])
    trace = []
    for row in base["points"]:
        energy = float(row["energy"])
        if drift_first_50 and int(row["k"]) == 25:
            energy += 1.0e-5
        trace.append(
            {
                "controller_round": int(row["k"]),
                "accepted_energy": energy,
                "absolute_energy_error": abs(energy - exact),
                "exact_same_cutoff_energy": exact,
            }
        )
    terminal_50 = float(base["points"][-1]["energy"])
    for controller_round in range(51, 71):
        energy = terminal_50 - (controller_round - 50) * 1.0e-7
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
            "n_h_outer": 100,
            "n_h_refit": 200,
            "n_grad": 300,
            "n_metric": 400,
        },
        "s_alg": 1000,
    }
    summary = {
        "schema": ingest.SUMMARY_SCHEMA,
        "available_controller_rounds": 70,
        "accepted_error_trace": trace,
        "canonical_all_work": work,
        "requested_rounds": [],
        "provenance": {
            "route_contract_sha256": ingest.ROUTE_CONTRACT_SHA256,
            "candidate_representation": ingest.CANDIDATE_REPRESENTATION,
            "qiskit_compile_convention": ingest.COMPILE_CONVENTION,
            "exact_same_cutoff_energy": exact,
        },
    }
    expected = job["expected_artifacts"]
    payloads = {
        expected["checkpoint"]: b"checkpoint fixture",
        (
            f"runs/{execution_id}/checkpoints/"
            "current.verified_singleton_resume.fixture.json"
        ): b"authenticated checkpoint sidecar fixture",
        expected["estimator_ledger"]: b"ledger fixture",
        expected["result"]: b"result fixture",
        expected["summary"]: _bytes(summary),
    }
    manifest = ingest.digested(
        {
            "schema": ingest.EXECUTION_MANIFEST_SCHEMA,
            "status": "passed",
            "package_id": ingest.PACKAGE_ID,
            "campaign_id": ingest.CAMPAIGN_ID,
            "execution_id": execution_id,
            "job_spec_sha256": job["sha256"],
            "authorization_sha256": authorization["sha256"],
            "protocol_sha256": job["protocol_sha256"],
            "route_contract_sha256": ingest.ROUTE_CONTRACT_SHA256,
            "resume_round": 50,
            "target_horizon": 70,
            "controller_rounds_completed": 70,
            "source_checkpoint_sha256": job["checkpoint_sha256"],
            "accepted_state_resume": True,
            "accepted_energy_roundoff_overlay": (
                "accepted_energy_roundoff_only_128ulp_v1"
            ),
            "operational_source_overlays": [
                "accepted_energy_roundoff_only_128ulp_v1",
                "phase0_gradient_screen_resume_closure_v1",
            ],
            "accepted_prefix_preservation": {
                "status": "passed",
                "source_round": 50,
                "source_checkpoint_sha256": job["checkpoint_sha256"],
                "terminal_energy": trace[49]["accepted_energy"],
                "terminal_state_fingerprint": "fixture",
            },
            "output_payloads": {
                role: {
                    "path": path,
                    "sha256": hashlib.sha256(payloads[path]).hexdigest(),
                    "size_bytes": len(payloads[path]),
                }
                for role, path in expected.items()
                if role != "execution_manifest"
            },
        }
    )
    manifest_payload = _bytes(manifest)
    payloads[expected["execution_manifest"]] = manifest_payload
    worker = ingest.digested(
        {
            "schema": ingest.WORKER_SCHEMA,
            "status": "passed",
            "package_id": ingest.PACKAGE_ID,
            "campaign_id": ingest.CAMPAIGN_ID,
            "execution_id": execution_id,
            "job_spec_sha256": job["sha256"],
            "authorization_sha256": authorization["sha256"],
            "execution_manifest_sha256": manifest["sha256"],
            "resume_round": 50,
            "controller_rounds_completed": 70,
            "accepted_state_resume": True,
            "operational_source_overlays": [
                "accepted_energy_roundoff_only_128ulp_v1",
                "phase0_gradient_screen_resume_closure_v1",
            ],
            "artifacts": [
                {
                    "path": path,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
                for path, payload in sorted(payloads.items())
            ],
        }
    )
    archive = tmp_path / f"{execution_id}__9629628__0.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name, payload in (
            ("worker_receipt.json", _bytes(worker)),
            *sorted(payloads.items()),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))
    remote = {
        "path": f"/staging/jsstrobel/page12/{archive.name}",
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "size_bytes": archive.stat().st_size,
    }
    return archive, remote, base


def test_page12_r70_archive_closes_and_preserves_fixed_round50_costs(
    tmp_path: Path,
) -> None:
    archive, remote, base = _fixture(tmp_path)

    adapter, receipt = ingest.build_outputs(
        archive_path=archive,
        cluster_id=9629628,
        proc_id=0,
        remote_archive=remote,
        retrieved_utc="2026-08-12T12:00:00Z",
    )

    ingest.verify_self_digest(adapter, label="continuation adapter")
    ingest.verify_self_digest(receipt, label="retrieval receipt")
    assert [row["k"] for row in adapter["merged_points"]] == list(range(1, 71))
    assert [row["k"] for row in adapter["continuation_points"]] == list(
        range(51, 71)
    )
    assert adapter["fixed_round_50_reporting"]["costs"] == base["terminal"][
        "costs"
    ]
    assert adapter["latest"]["k"] == 70
    assert receipt["package_job_worker_manifest_summary_closure_passed"] is True


def test_page12_r70_archive_rejects_hydrated_prefix_drift(tmp_path: Path) -> None:
    archive, remote, _base = _fixture(tmp_path, drift_first_50=True)

    with pytest.raises(
        ingest.ContinuationIngestError,
        match="authenticated first-50 trajectory drifted at round 25",
    ):
        ingest.build_outputs(
            archive_path=archive,
            cluster_id=9629628,
            proc_id=0,
            remote_archive=remote,
        )


def test_report_merge_extends_only_trajectory_and_keeps_round50_tuple(
    tmp_path: Path,
) -> None:
    archive, remote, base = _fixture(tmp_path)
    continuation, _receipt = ingest.build_outputs(
        archive_path=archive,
        cluster_id=9629628,
        proc_id=0,
        remote_archive=remote,
    )
    terminal = base["terminal"]
    current = {
        "status": "completed_authenticated_remote_summary",
        "points": base["points"],
        "latest": {
            "k": 50,
            "energy": terminal["energy"],
            "error": terminal["error"],
        },
        "costs": terminal["costs"],
        "compile": terminal["compile"],
        "work_components": terminal["work_components"],
        "source": {},
    }
    base_binding = continuation["source"]["base_completed_adapter"]

    merged = report.merge_page12_r70_continuation(
        current,
        continuation,
        regime="weak_strong",
        completed_adapter_binding=base_binding,
        continuation_adapter_binding={"path": "fixture", "sha256": "a" * 64},
        continuation_archive_binding={"status": "passed"},
    )

    assert merged["trajectory_controller_round"] == 70
    assert merged["fixed_resource_controller_round"] == 50
    assert merged["latest"]["k"] == 70
    assert merged["costs"] == current["costs"]
    assert merged["compile"] == current["compile"]
    assert merged["work_components"] == current["work_components"]


def test_report_merge_rejects_round50_cost_substitution(tmp_path: Path) -> None:
    archive, remote, base = _fixture(tmp_path)
    continuation, _receipt = ingest.build_outputs(
        archive_path=archive,
        cluster_id=9629628,
        proc_id=0,
        remote_archive=remote,
    )
    continuation["fixed_round_50_reporting"]["costs"]["N2q"] += 1
    terminal = base["terminal"]
    current = {
        "status": "completed_authenticated_remote_summary",
        "points": base["points"],
        "latest": terminal,
        "costs": terminal["costs"],
        "compile": terminal["compile"],
        "work_components": terminal["work_components"],
        "source": {},
    }

    with pytest.raises(report.UpdateError, match="continuation identity drifted"):
        report.merge_page12_r70_continuation(
            current,
            continuation,
            regime="weak_strong",
            completed_adapter_binding=continuation["source"][
                "base_completed_adapter"
            ],
            continuation_adapter_binding={},
            continuation_archive_binding={},
        )
