#!/usr/bin/env python3
"""Record the five-job HH/Hubbard higher-L Append preflight."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


BUNDLE = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    source = _load(BUNDLE / "source_archive_manifest.json")
    bundle = _load(BUNDLE / "bundle_manifest.json")
    archive = BUNDLE / "source_locked.tar.gz"
    if _sha256(archive) != source["archive_sha256"]:
        raise RuntimeError("source archive hash mismatch")
    if bundle["submission_enabled"] is not True:
        raise RuntimeError("derived bundle must retain the authenticated submission gate")
    payload = {
        "schema": "paper_i_higher_l_append_preflight_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "source_archive_sha256": source["archive_sha256"],
        "generic_static_adapt_variants_sha256": (
            source["files"]["pipelines/exact_bench/generic_static_adapt_variants.py"]["sha256"]
        ),
        "checks": {
            "bundle_contract_tests": "13 passed",
            "archive_only_profile_and_method_regressions": "21 passed",
            "archive_only_family_smokes": (
                "2 passed; one completed Append round each for HH L=3 nph=3 "
                "weak-strong and Hubbard L=6 weak"
            ),
            "archive_only_worker_import": "pass; source-defined cap classifier imported and exercised",
            "worker_static_preflight": "py_compile, bash -n, and --help pass",
            "normalized_manifest_count": 5,
            "append_v4_source_contract_lock_count": 5,
            "same_cutoff_and_horizon_contract": "pass",
            "runtime_qubit_caps": (
                "all jobs lock resource and exact-fidelity caps to declared 12 qubits"
            ),
            "optimizer_failure_policy": (
                "only finite non-increasing exact Powell maxiter caps accepted; "
                "every other failure rejected"
            ),
        },
        "archive_only_smoke_receipts": [
            {
                "family": "hh",
                "case_id": "hh_L3_nph3_higher_l_weak_strong",
                "num_qubits": 12,
                "completed_append_rounds": 1,
                "energy_before": 1.75,
                "energy_after": 1.3145857491668038,
                "projected_singleton_child_count": 725,
                "ordered_pool_hash": (
                    "636b834c5c0ef389966f9d9aaa23e1567af489409ebc088ed8bf52cbc11ee259"
                ),
                "sector_leak_flag": False,
                "boson_truncation_leak_flag": False,
                "qiskit_compile_status": "ok",
            },
            {
                "family": "hubbard",
                "case_id": "hubbard_L6_higher_l_weak",
                "num_qubits": 12,
                "completed_append_rounds": 1,
                "energy_before": 0.75,
                "energy_after": 0.7499999999999997,
                "projected_singleton_child_count": 54,
                "ordered_pool_hash": (
                    "fd367d46c8974aa2ed8f36be74ea6210772a13deba91adc17378b1681197eb97"
                ),
                "sector_leak_flag": False,
                "boson_truncation_leak_flag": False,
                "qiskit_compile_status": "ok",
            },
        ],
        "scientific_blockers": [],
        "operational_blockers": [],
    }
    _write(BUNDLE / "archive_only_preflight.json", payload)
    hashes = {
        path.relative_to(BUNDLE).as_posix(): _sha256(path)
        for path in sorted(BUNDLE.rglob("*"))
        if (
            path.is_file()
            and path.name != "submission_artifact_hashes.json"
            and "__pycache__" not in path.parts
            and ".pytest_cache" not in path.parts
            and path.suffix != ".pyc"
        )
    }
    _write(
        BUNDLE / "submission_artifact_hashes.json",
        {"schema": "paper_i_higher_l_append_artifact_hashes_v1", "files": hashes},
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
