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


def test_cli_exit_two_when_the_manifest_is_missing(tmp_path):
    # Recheck 1: a nonexistent manifest used to escape as an uncaught FileNotFoundError (exit 1).
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "STATIC_ADAPT_HH_POOL_CACHE": "off", "PYTHONDONTWRITEBYTECODE": "1", "PYDEPHASING_TESTING": "1"}
    out_dir = tmp_path / "out_missing"
    argv = [sys.executable, "-m", "pipelines.chemistry_bridge", "run", "--manifest", str(tmp_path / "does_not_exist.manifest.json"), "--out-dir", str(out_dir), "--methods", "mapping"]
    proc = subprocess.run(argv, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == bm.EXIT_USAGE, proc.stdout + proc.stderr
    assert "FileNotFoundError" in proc.stderr and not out_dir.exists()
    export = subprocess.run([sys.executable, "-m", "pipelines.chemistry_bridge", "export-qdynamics", "--h5", str(tmp_path / "missing_ele.h5"), "--out", str(tmp_path / "p.json")], cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=300)
    assert export.returncode == 2 and "export refused" in export.stderr


# ---------------------------------------------------------------------------
# Packet 003 (2026-10-20): P3 failures after a real intake are not usage errors
# ---------------------------------------------------------------------------


_INJECT_RUNNER = r"""
import sys, pipelines.chemistry_bridge.methods as m
from pipelines.chemistry_bridge.__main__ import main
kind, manifest, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
def boom(*a, **k):
    print("INJECTED_FAULT_AT_METHOD_BOUNDARY", file=sys.stderr)
    if kind == "keyerror": raise KeyError("injected-key")
    if kind == "oserror": raise OSError("injected-os")
    raise RuntimeError("injected-runtime")
m.run_mapping_witness = boom
raise SystemExit(main(["run", "--manifest", manifest, "--out-dir", out_dir, "--methods", "mapping"]))
"""


@pytest.mark.parametrize("kind, marker", [("keyerror", "KeyError"), ("oserror", "OSError"), ("runtime", "RuntimeError")])
def test_003_internal_failure_after_intake_exits_one_with_traceback_not_usage(kind, marker, tmp_path):
    manifest = _manifest(tmp_path, "contract4")
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "STATIC_ADAPT_HH_POOL_CACHE": "off", "PYTHONDONTWRITEBYTECODE": "1", "PYDEPHASING_TESTING": "1"}
    out_dir = tmp_path / f"out_{kind}"
    proc = subprocess.run([sys.executable, "-c", _INJECT_RUNNER, kind, manifest["manifest_path"], str(out_dir)], cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == bm.EXIT_CHECK_FAILED, proc.stdout + proc.stderr
    assert "INJECTED_FAULT_AT_METHOD_BOUNDARY" in proc.stderr  # the real intake succeeded first
    assert "run failed after intake" in proc.stderr and marker in proc.stderr and "Traceback" in proc.stderr
    assert "run refused" not in proc.stderr
    assert not (out_dir / "bridge_receipts.json").exists()


def test_003_output_failure_after_methods_is_operational_exit_one(tmp_path):
    manifest = _manifest(tmp_path, "contract4")
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    proc = _run(tmp_path, manifest, "--methods", "mapping", "--out-dir", str(blocker / "sub"))
    # argparse last-wins: our helper's own --out-dir precedes, the explicit one here wins
    assert proc.returncode == bm.EXIT_CHECK_FAILED, proc.stdout + proc.stderr
    assert "run failed after intake" in proc.stderr and ("NotADirectoryError" in proc.stderr or "FileExistsError" in proc.stderr or "OSError" in proc.stderr)


# ---------------------------------------------------------------------------
# Packet 004 (2026-10-20): invalid request configuration is usage, not execution failure
# ---------------------------------------------------------------------------


def _assert_usage(proc: subprocess.CompletedProcess, needle: str) -> None:
    assert proc.returncode == bm.EXIT_USAGE, proc.stdout + proc.stderr
    assert "run refused" in proc.stderr and needle in proc.stderr
    assert "run failed after intake" not in proc.stderr and "Traceback" not in proc.stderr
    assert not (proc.out_dir / "bridge_receipts.json").exists()  # type: ignore[attr-defined]


@pytest.mark.parametrize("backend, needle", [("not-a-backend", "unknown backend library"), ("torch", "requires a device"), ("numpy:mps:float32", "malformed backend spec"), ("torch:tpu", "unknown torch device"), ("torch:cpu:int8", "unknown real dtype")])
def test_004_unparseable_backend_spec_is_usage_exit_two(backend, needle, tmp_path):
    # Reviewer control: `--methods mapping --backend not-a-backend` exited 1 at 67f88981.
    manifest = _manifest(tmp_path, "contract4")
    _assert_usage(_run(tmp_path, manifest, "--methods", "mapping", "--backend", backend), needle)


@pytest.mark.parametrize("grid, needle", [("0,0", "strictly increasing"), ("0.1,0.05", "strictly increasing"), ("0", "at least two"), ("0,nan", "finite"), ("0,inf", "finite"), ("a,b", "real numbers"), (",", "at least two")])
def test_004_invalid_apm_grid_is_usage_exit_two(grid, needle, tmp_path):
    # Reviewer control: `--methods apm --apm-times 0,0` exited 1 at 67f88981.
    manifest = _manifest(tmp_path, "contract4")
    _assert_usage(_run(tmp_path, manifest, "--methods", "apm", "--apm-times", grid), needle)


def test_004_inadmissible_apm_pool_is_usage_exit_two(tmp_path):
    manifest = _manifest(tmp_path, "contract4")
    _assert_usage(_run(tmp_path, manifest, "--methods", "apm", "--apm-pool", "no_such_pool"), "not admissible")


def test_004_valid_but_unavailable_backend_still_exits_three(tmp_path):
    # Syntax validation must not swallow the availability path (skipped_unsupported receipt, exit 3).
    if backend_availability("torch:cuda:float64").available:
        pytest.skip("cuda available here")
    manifest = _manifest(tmp_path, "contract4")
    proc = _run(tmp_path, manifest, "--methods", "qse", "--backend", "torch:cuda:float64")
    assert proc.returncode == bm.EXIT_NOT_EXECUTED and (proc.out_dir / "bridge_receipts.json").exists()  # type: ignore[attr-defined]


def test_004_validate_apm_times_shared_with_runner():
    assert bm.validate_apm_times(("0", 0.5, 1)) == [0.0, 0.5, 1.0]
    for bad in ((0.0, 0.0), (0.0,), (), (0.0, float("nan")), (1.0, 0.5), ("x", "y")):
        with pytest.raises(ValueError):
            bm.validate_apm_times(bad)


# ---------------------------------------------------------------------------
# 2026-10-20: Hermitian pool default and controller overrides at the CLI
# ---------------------------------------------------------------------------


def test_cli_default_apm_pool_is_the_hermitian_family_and_moves(tmp_path):
    manifest = _manifest(tmp_path, "contract4")
    proc = _run(tmp_path, manifest, "--methods", "apm", "--apm-times", "0,0.02,0.04")
    assert proc.returncode == bm.EXIT_ALL_VERIFIED, proc.stdout + proc.stderr
    receipts = json.loads((proc.out_dir / "bridge_receipts.json").read_text())  # type: ignore[attr-defined]
    apm = [r for r in receipts["receipts"] if r["method"] == "ap_mclachlan"][0]
    assert apm["results"]["pool"]["pool_key"] == "hermitian_excitations" and apm["results"]["trajectory_moved"] is True


def test_cli_apm_support_config_is_validated_before_intake_and_recorded(tmp_path):
    manifest = _manifest(tmp_path, "contract4")
    _assert_usage(_run(tmp_path, manifest, "--methods", "apm", "--apm-support-config", "[1,2]"), "JSON object")
    _assert_usage(_run(tmp_path, manifest, "--methods", "apm", "--apm-support-config", '{"no_such_field": 1}'), "rejected by SupportPatchControllerConfig")
    _assert_usage(_run(tmp_path, manifest, "--methods", "apm", "--apm-support-config", "{oops"), "not valid JSON")
    proc = _run(tmp_path, manifest, "--methods", "apm", "--apm-times", "0,0.02,0.04", "--apm-support-config", '{"max_structural_pool_size": 16}')
    assert proc.returncode == bm.EXIT_ALL_VERIFIED, proc.stdout + proc.stderr
    receipts = json.loads((proc.out_dir / "bridge_receipts.json").read_text())  # type: ignore[attr-defined]
    apm = [r for r in receipts["receipts"] if r["method"] == "ap_mclachlan"][0]
    assert apm["results"]["support_patch_config"]["max_structural_pool_size"] == 16


def test_cli_repository_pool_key_still_accepted(tmp_path):
    manifest = _manifest(tmp_path, "contract4")
    proc = _run(tmp_path, manifest, "--methods", "apm", "--apm-times", "0,0.02,0.04", "--apm-pool", "full_meta")
    assert proc.returncode == bm.EXIT_ALL_VERIFIED, proc.stdout + proc.stderr
    receipts = json.loads((proc.out_dir / "bridge_receipts.json").read_text())  # type: ignore[attr-defined]
    apm = [r for r in receipts["receipts"] if r["method"] == "ap_mclachlan"][0]
    assert apm["results"]["pool"]["pool_key"] == "full_meta" and apm["results"]["trajectory_moved"] is False


def test_cli_qse_basis_choice_is_recorded_and_validated(tmp_path):
    manifest = _manifest(tmp_path, "random8")
    proc = _run(tmp_path, manifest, "--methods", "exact,qse", "--qse-basis", "uccsd_sdtq_products")
    assert proc.returncode == bm.EXIT_ALL_VERIFIED, proc.stdout + proc.stderr
    receipts = json.loads((proc.out_dir / "bridge_receipts.json").read_text())  # type: ignore[attr-defined]
    qse = [r for r in receipts["receipts"] if r["method"] == "qse"][0]
    assert qse["results"]["basis_kind"] == "uccsd_sdtq_products" and qse["results"]["basis_spans_sector"] is True
    assert qse["checks"]["ritz_minus_exact_sector_roots_max_abs"] < 1.0e-9
    bad = _run(tmp_path, manifest, "--methods", "qse", "--qse-basis", "no_such_basis")
    assert bad.returncode == bm.EXIT_USAGE and "invalid choice" in bad.stderr  # argparse refuses before intake


def test_cli_ra_arguments_are_validated_before_intake_and_run(tmp_path):
    manifest = _manifest(tmp_path, "contract4")
    _assert_usage(_run(tmp_path, manifest, "--methods", "ra", "--ra-rounds", "0"), "at least 1")
    _assert_usage(_run(tmp_path, manifest, "--methods", "ra", "--ra-pool", "hermitian_excitations"), "not a repository pool key")
    bad = _run(tmp_path, manifest, "--methods", "ra", "--ra-insertion", "beam")
    assert bad.returncode == bm.EXIT_USAGE and "invalid choice" in bad.stderr
    proc = _run(tmp_path, manifest, "--methods", "ra", "--ra-rounds", "1", "--ra-insertion", "append_only")
    assert proc.returncode == bm.EXIT_ALL_VERIFIED, proc.stdout + proc.stderr
    receipts = json.loads((proc.out_dir / "bridge_receipts.json").read_text())  # type: ignore[attr-defined]
    construction = [r for r in receipts["receipts"] if r["method"] == "ra_adapt_construction"][0]
    assert construction["status"] == bm.STATUS_CPU_VERIFIED and construction["results"]["request"] == {"maximum_controller_rounds_requested": 1, "maximum_controller_rounds_executed": 1, "insertion": "append_only", "pool_key": "uccsd", "exact_ed_stop_tolerance": None}
