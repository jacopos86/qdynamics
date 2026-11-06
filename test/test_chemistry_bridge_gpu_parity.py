"""CPU/GPU parity for the bridge at declared, dtype-aware tolerances.

Each backend is probed first; an unavailable one is reported as a pytest skip
whose reason is the backend's own obstacle (missing torch, device absent,
dtype rejected).  A backend that is available must report the device it was
asked for from a live tensor, and the method receipts must carry
``gpu_verified`` only when that device is an accelerator.  Nothing here ever
falls back to CPU on behalf of a GPU request.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.chemistry_bridge import array_backend as ab
from pipelines.chemistry_bridge import methods as bm
from pipelines.chemistry_bridge.polynomial_action import BridgePolynomialAction
from pipelines.chemistry_bridge.qdynamics_export import bridge_problem_from_qdynamics, qdynamics_matrix_elements_from_mapping, write_bridge_problem_json
from pipelines.chemistry_bridge.synthetic_witnesses import synthetic_qdynamics_mapping


ACCELERATOR_SPECS = ("torch:cpu:float64", "torch:cpu:float32", "torch:mps:float32", "torch:mps:float64", "torch:cuda:float64", "torch:cuda:float32")


def _backend_or_skip(spec: str):
    report = ab.backend_availability(spec)
    if not report.available:
        pytest.skip(f"{spec} unsupported here: {report.reason}")
    backend = ab.resolve_backend(spec)
    live = backend.report()
    assert live.exercised_device == live.requested_device, live
    return backend, live


def _manifest(tmp_path: Path, kind: str):
    mapping = synthetic_qdynamics_mapping(kind)
    elements = qdynamics_matrix_elements_from_mapping(mapping, source={"kind": kind, "simulation": True, "producer_generated": False}, declared_origin="synthetic")
    return write_bridge_problem_json(bridge_problem_from_qdynamics(elements), tmp_path / f"{kind}.json")


@pytest.fixture(autouse=True)
def _pool_cache_off(monkeypatch):
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")


@pytest.mark.parametrize("spec", ACCELERATOR_SPECS)
def test_action_parity_against_float64_numpy(spec, tmp_path):
    backend, live = _backend_or_skip(spec)
    manifest = _manifest(tmp_path, "random6")
    bridge = bm.resolve_bridge_context(manifest["manifest_path"])
    action = BridgePolynomialAction.from_polynomial(bridge.context.hamiltonian)
    reference = action.bind(ab.NumpyBackend("float64"))
    device = action.bind(backend)
    tol = ab.parity_tolerances(live.real_dtype)["vector_abs"]
    rng = np.random.default_rng(99)
    for _ in range(4):
        psi = rng.standard_normal(1 << bridge.nq) + 1j * rng.standard_normal(1 << bridge.nq)
        psi /= np.linalg.norm(psi)
        out = device.apply_numpy(psi)
        ref = reference.apply_numpy(psi)
        assert np.max(np.abs(out - ref)) <= tol * max(1.0, float(np.max(np.abs(ref))))
        assert abs(device.expectation_numpy(psi) - reference.expectation_numpy(psi)) <= ab.parity_tolerances(live.real_dtype)["scalar_abs"] * 10


@pytest.mark.parametrize("spec", ACCELERATOR_SPECS)
@pytest.mark.parametrize("kind", ("contract4", "random6"))
def test_method_receipts_on_accelerator_backend(spec, kind, tmp_path):
    backend, live = _backend_or_skip(spec)
    manifest = _manifest(tmp_path, kind)
    summary = bm.run_bridge(manifest["manifest_path"], backend_spec=spec, methods=("mapping", "exact", "qse", "adapt"))
    expected = bm.STATUS_GPU_VERIFIED if live.is_gpu else bm.STATUS_CPU_VERIFIED
    for receipt in summary["receipts"]:
        assert receipt["backend"]["exercised_device"] == live.requested_device, receipt["backend"]
        assert receipt["backend"]["real_dtype"] == live.real_dtype
        assert receipt["status"] == expected, (receipt["method"], receipt["status"], receipt["checks"])
    receipts = {receipt["method"]: receipt for receipt in summary["receipts"]}
    tol = ab.parity_tolerances(live.real_dtype)
    assert receipts["qse"]["checks"]["backend_hamiltonian_minus_core_max_abs"] <= tol["matrix_abs"] * 10
    assert receipts["exact_sector_reference"]["checks"]["repository_exact_target_minus_bridge_abs"] <= tol["scalar_abs"] * 10


def test_gpu_status_is_impossible_without_an_accelerator_device():
    for spec in ACCELERATOR_SPECS:
        report = ab.backend_availability(spec)
        if report.available and not report.is_gpu:
            assert bm.verified_status(report, checks_passed=True) == bm.STATUS_CPU_VERIFIED
        if not report.available:
            assert bm.verified_status(report, checks_passed=True) == bm.STATUS_SKIPPED_UNSUPPORTED
            assert report.reason


def test_missing_torch_is_reported_verbatim():
    try:
        import torch  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        report = ab.backend_availability("torch:mps:float32")
        assert not report.available and "torch is not importable" in str(report.reason)
        assert report.library_version is None
