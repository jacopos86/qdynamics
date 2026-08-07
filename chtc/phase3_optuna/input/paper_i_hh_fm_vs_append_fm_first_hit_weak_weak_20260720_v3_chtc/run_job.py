#!/usr/bin/env python3
"""Run one sequential weak-weak matched pair from a frozen source tree."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BUNDLE_ID = "paper_i_hh_fm_vs_append_fm_first_hit_weak_weak_20260720_v3_chtc"
JOB_ID = "weak_weak_fm_vs_append_fm_first_hit"
CAMPAIGN_MODULE = "pipelines.exact_bench.paper_i_hh_fm_vs_append_fm_first_hit"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _copy_narrow_campaign(campaign: Path, transfer: Path) -> dict[str, str]:
    relatives = (
        "campaign_manifest.json",
        "settings_diff.json",
        "source_lock/formal_manifold_config.json",
        "source_lock/source_lock.json",
        "weak-weak/pair_manifest.json",
        "weak-weak/results.json",
        "weak-weak/results.md",
        "weak-weak/validation.json",
        "weak-weak/fm_snake/plan.json",
        "weak-weak/fm_snake/result.json",
        "weak-weak/fm_snake/qiskit_sidecar.json",
        "weak-weak/fm_snake/row_summary.json",
        "weak-weak/projected_singleton_append_fm/plan.json",
        "weak-weak/projected_singleton_append_fm/result.json",
        "weak-weak/projected_singleton_append_fm/qiskit_sidecar.json",
        "weak-weak/projected_singleton_append_fm/row_summary.json",
    )
    copied: dict[str, str] = {}
    for relative in relatives:
        source = campaign / relative
        if not source.is_file():
            raise RuntimeError(f"required terminal artifact is absent: {relative}")
        destination = transfer / "campaign" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied[f"campaign/{relative}"] = _sha256(destination)
    return copied


def _validate_job(job: dict[str, Any]) -> None:
    if job.get("bundle_id") != BUNDLE_ID or job.get("job_id") != JOB_ID:
        raise RuntimeError("job identity mismatch")
    if job.get("regime") != "weak-weak":
        raise RuntimeError("this bundle is weak-weak only")
    comparison = job.get("comparison", {})
    expected = {
        "initial_ansatz": "empty_hf_reference_v1",
        "automatic_hh_seed_disabled": True,
        "target_abs_delta_e": 2.0e-4,
        "max_controller_rounds": 30,
        "optimizer_maxiter": 200,
        "line_search_max_steps": 15,
        "qbroyd_qbang_enabled": False,
        "reported_query_coordinate": "winning_lineage_S_alg_only",
        "discarded_branch_work_reported": False,
    }
    for key, value in expected.items():
        if comparison.get(key) != value:
            raise RuntimeError(f"job contract drift for {key}")
    if comparison.get("routes") != ["fm_snake", "projected_singleton_append_fm"]:
        raise RuntimeError("route pair mismatch")


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: run_job.py JOB_MANIFEST OUTPUT_ROOT")
    job_path = Path(sys.argv[1])
    output = Path(sys.argv[2])
    campaign = output / "campaign"
    transfer = output / "transfer"
    transfer.mkdir(parents=True, exist_ok=True)
    job = _load(job_path)
    _validate_job(job)
    execution = {
        "schema": "paper_i_hh_fm_vs_append_fm_first_hit_execution_v1",
        "bundle_id": BUNDLE_ID,
        "job_id": JOB_ID,
        "started_utc": _utc_now(),
        "status": "running",
        "command": [
            sys.executable,
            "-m",
            CAMPAIGN_MODULE,
            "run-pair",
            "--campaign-dir",
            str(campaign),
            "--regime",
            "weak-weak",
        ],
    }
    _write(output / "execution.json", execution)
    _write(transfer / "execution.json", execution)
    _write(transfer / "normalized_job_manifest.json", job)
    env = dict(os.environ)
    env.update(
        {
            "TABLE_I_STATIC_SUITE_PROFILE": (
                "paper_i_hh_completion_samecutoff_nph3_nph7_20260718_v1"
            ),
            "GENERIC_STATIC_TABLE_RESOURCE_QUBIT_CAP": "12",
            "GENERIC_STATIC_TABLE_RESOURCE_POOL_TERM_CAP": "16384",
            "GENERIC_STATIC_TABLE_EXACT_FIDELITY_MAX_QUBITS": "12",
            "STATIC_ADAPT_HH_POOL_CACHE": "disk",
            "STATIC_ADAPT_HH_POOL_CACHE_SCOPE": "exact",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    try:
        subprocess.run(execution["command"], env=env, check=True)
        result = _load(campaign / "weak-weak" / "results.json")
        if result.get("status") != "pass":
            raise RuntimeError("campaign validation did not pass")
        if result.get("reporting_scope") != {
            "query_coordinate": "winning_lineage_S_alg_only",
            "discarded_branch_work_reported": False,
        }:
            raise RuntimeError("reporting scope drift")
        rows = result.get("rows")
        if not isinstance(rows, list) or len(rows) != 2:
            raise RuntimeError("matched pair must contain exactly two rows")
        for row in rows:
            if row.get("exact_S_alg_accounting_complete") is not True:
                raise RuntimeError("exact winning-lineage S_alg is incomplete")
            if row.get("winning_lineage_S_alg") is None:
                raise RuntimeError("winning-lineage S_alg is absent")
            if any("discarded" in str(key).lower() for key in row):
                raise RuntimeError("discarded work leaked into the comparison row")
        files = _copy_narrow_campaign(campaign, transfer)
        receipt = {
            "schema": "paper_i_hh_fm_vs_append_fm_first_hit_validation_receipt_v1",
            "generated_utc": _utc_now(),
            "status": "pass",
            "bundle_id": BUNDLE_ID,
            "job_id": JOB_ID,
            "result_status": result["status"],
            "rows": rows,
            "transferred_files": files,
            "transfer_scope": "terminal_results_qiskit_query_and_provenance_only",
        }
        _write(transfer / "validation_receipt.json", receipt)
        execution.update(
            {
                "status": "completed",
                "finished_utc": _utc_now(),
                "validation_receipt": "validation_receipt.json",
            }
        )
    except BaseException as exc:
        execution.update(
            {
                "status": "failed",
                "finished_utc": _utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        _write(output / "execution.json", execution)
        _write(transfer / "execution.json", execution)
        raise
    _write(output / "execution.json", execution)
    _write(transfer / "execution.json", execution)
    print(json.dumps(execution, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
