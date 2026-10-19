"""Process-level exit semantics of ``python -m pipelines.chemistry_bridge run``.

Review finding 5: the CLI returned 0 for a misspelled method (no calculation)
and for a run whose receipt said ``failed``.  These tests assert actual
subprocess exit codes for invalid, empty, failed, skipped and verified cases,
with receipts preserved in every case that reaches the runners.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.chemistry_bridge import methods as bm
from pipelines.chemistry_bridge.array_backend import backend_availability
from pipelines.chemistry_bridge.qdynamics_export import bridge_problem_from_qdynamics, qdynamics_matrix_elements_from_mapping, write_bridge_problem_json
from pipelines.chemistry_bridge.synthetic_witnesses import synthetic_qdynamics_mapping


def _manifest(tmp_path: Path, kind: str) -> dict:
    elements = qdynamics_matrix_elements_from_mapping(synthetic_qdynamics_mapping(kind), source={"kind": kind, "simulation": True}, declared_origin="synthetic")
    return write_bridge_problem_json(bridge_problem_from_qdynamics(elements), tmp_path / f"{kind}.json")


def _run(tmp_path: Path, manifest: dict, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "STATIC_ADAPT_HH_POOL_CACHE": "off", "PYTHONDONTWRITEBYTECODE": "1", "PYDEPHASING_TESTING": "1"}
    out_dir = tmp_path / ("out_" + "_".join(a.strip("-").replace(":", "_").replace(",", "_") for a in args)[:60])
    argv = [sys.executable, "-m", "pipelines.chemistry_bridge", "run", "--manifest", manifest["manifest_path"], "--out-dir", str(out_dir), *args]
    proc = subprocess.run(argv, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=600)
    proc.out_dir = out_dir  # type: ignore[attr-defined]
    return proc


def test_cli_rejects_unknown_and_empty_method_requests(tmp_path):
    manifest = _manifest(tmp_path, "contract4")
    misspelled = _run(tmp_path, manifest, "--methods", "qes")
    assert misspelled.returncode == bm.EXIT_USAGE, misspelled.stdout + misspelled.stderr
    assert "unknown methods" in misspelled.stderr and not (misspelled.out_dir / "bridge_receipts.json").exists()
    empty = _run(tmp_path, manifest, "--methods", "")
    assert empty.returncode == bm.EXIT_USAGE and "no methods requested" in empty.stderr


def test_cli_exit_zero_only_when_every_requested_method_is_verified(tmp_path):
    manifest = _manifest(tmp_path, "contract4")
    ok = _run(tmp_path, manifest, "--methods", "mapping,exact,qse")
    assert ok.returncode == bm.EXIT_ALL_VERIFIED, ok.stdout + ok.stderr
    payload = json.loads(ok.stdout)
    assert payload["exit_code"] == 0 and set(payload["status_by_method"].values()) == {bm.STATUS_CPU_VERIFIED}
    receipts = json.loads((ok.out_dir / "bridge_receipts.json").read_text())
    assert receipts["exit_code"] == 0 and receipts["requested_methods"] == ["mapping", "exact", "qse"]


def test_cli_exit_three_when_a_requested_backend_cannot_run(tmp_path):
    report = backend_availability("torch:cuda:float64")
    if report.available:
        pytest.skip("cuda is available here; the skipped-backend case cannot be demonstrated")
    manifest = _manifest(tmp_path, "contract4")
    proc = _run(tmp_path, manifest, "--methods", "qse", "--backend", "torch:cuda:float64")
    assert proc.returncode == bm.EXIT_NOT_EXECUTED, proc.stdout + proc.stderr
    receipts = json.loads((proc.out_dir / "bridge_receipts.json").read_text())
    assert receipts["status_by_method"] == {"qse": bm.STATUS_SKIPPED_UNSUPPORTED}
    assert receipts["receipts"][0]["backend"]["exercised_device"] is None


def test_cli_exit_one_when_a_check_fails_and_receipt_is_kept(tmp_path):
    # Review probe: per-Pauli-word APM coordinates leak particle number on random8;
    # the receipt says failed and the process must say so too.
    manifest = _manifest(tmp_path, "random8")
    proc = _run(tmp_path, manifest, "--methods", "apm", "--apm-parameterization", "per_pauli_term", "--apm-times", "0,0.005,0.01,0.015,0.02", "--apm-candidate-filter", "none")
    assert proc.returncode == bm.EXIT_CHECK_FAILED, proc.stdout + proc.stderr
    receipts = json.loads((proc.out_dir / "bridge_receipts.json").read_text())
    assert receipts["status_by_method"] == {"ap_mclachlan": bm.STATUS_FAILED}
    assert receipts["receipts"][0]["checks"]["sector_leakage_max_norm"] > 1.0e-6


def test_cli_exit_two_when_the_strict_intake_refuses_the_input(tmp_path):
    manifest = _manifest(tmp_path, "contract4")
    path = Path(manifest["problem_json_path"])
    path.write_bytes(path.read_bytes() + b"\n")
    proc = _run(tmp_path, manifest, "--methods", "mapping")
    assert proc.returncode == bm.EXIT_USAGE and "SHA256 mismatch" in proc.stderr
