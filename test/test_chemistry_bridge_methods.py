"""Method-level witnesses for the bridge on synthetic inputs (CPU).

The QSE roots are compared with exact fixed-sector diagonalization (the UCC
basis spans the sector for these sizes), the AP-McLachlan trajectory with exact
sector propagation, and the ADAPT selection gradients with the repository's
numpy formula.  The RA-ADAPT facade verdict is recorded, not assumed.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.chemistry_bridge import methods as bm
from pipelines.chemistry_bridge.qdynamics_export import FrozenCoreReduction, bridge_problem_from_qdynamics, qdynamics_matrix_elements_from_mapping, write_bridge_problem_json
from pipelines.chemistry_bridge.synthetic_witnesses import CONTRACT_FOUR_QUBIT_ROOTS, synthetic_qdynamics_mapping


def _manifest(tmp_path: Path, kind: str, frozen: int = 0):
    mapping = synthetic_qdynamics_mapping(kind)
    elements = qdynamics_matrix_elements_from_mapping(mapping, source={"payload_kind": "synthetic_witness", "kind": kind})
    bridge_problem = bridge_problem_from_qdynamics(elements, frozen_core=FrozenCoreReduction(frozen) if frozen else None)
    return write_bridge_problem_json(bridge_problem, tmp_path / f"{kind}.json")


@pytest.fixture(autouse=True)
def _pool_cache_off(monkeypatch):
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")


def test_contract_fixture_all_methods_cpu(tmp_path):
    manifest = _manifest(tmp_path, "contract4")
    summary = bm.run_bridge(manifest["manifest_path"], backend_spec="numpy:float64", out_dir=tmp_path / "out")
    statuses = summary["status_by_method"]
    assert set(statuses) == {"mapping_witness", "exact_sector_reference", "qse", "static_adapt_selection_gradients", "ap_mclachlan"}
    assert all(status == bm.STATUS_CPU_VERIFIED for status in statuses.values()), statuses
    receipts = {receipt["method"]: receipt for receipt in summary["receipts"]}
    assert summary["inputs"]["data_role"] == "synthetic_or_non_file_input"
    assert summary["inputs"]["raw_sha256"] == manifest["raw_sha256"]
    exact = receipts["exact_sector_reference"]
    np.testing.assert_allclose(exact["results"]["low_roots"], np.sort(np.asarray(CONTRACT_FOUR_QUBIT_ROOTS)), atol=1.0e-12, rtol=0.0)
    qse = receipts["qse"]
    assert qse["checks"]["ritz_minus_exact_sector_roots_max_abs"] < 1.0e-12
    assert qse["checks"]["retained_rank"] == 4
    adapt = receipts["static_adapt_selection_gradients"]
    assert adapt["checks"]["backend_gradients_minus_repository_max_abs"] == 0.0
    gate = adapt["results"]["ra_adapt_facade_gate"]
    assert gate["admitted"] is False and "locked" in gate["message"]
    assert adapt["results"]["full_route_status"] == bm.STATUS_BLOCKED
    assert adapt["results"]["top_gradients"][0]["label"] == "uccsd_dbl(ab:0,2->1,3)"
    apm = receipts["ap_mclachlan"]
    assert apm["checks"]["sector_leakage_max_norm"] == 0.0
    assert apm["checks"]["min_fidelity_vs_exact"] > 1.0 - 1.0e-9
    assert apm["checks"]["energy_drift_max_abs"] < 1.0e-6
    assert apm["results"]["parameterization_mode"] == "logical_shared"
    assert (tmp_path / "out" / "bridge_receipts.json").is_file()


def test_six_qubit_witness_qse_spans_sector_and_apm_tracks_exact(tmp_path):
    manifest = _manifest(tmp_path, "random6")
    bridge = bm.resolve_bridge_context(manifest["manifest_path"])
    exact = bm.run_exact_reference(bridge, "numpy:float64", n_roots=9)
    assert exact.status == bm.STATUS_CPU_VERIFIED and exact.checks["sector_dimension"] == 9
    qse = bm.run_qse(bridge, "numpy:float64", exact_roots=exact.results["low_roots"])
    assert qse.status == bm.STATUS_CPU_VERIFIED
    assert qse.checks["basis_size"] == 9 and qse.checks["retained_rank"] == 9
    assert qse.checks["ritz_minus_exact_sector_roots_max_abs"] < 1.0e-9
    apm = bm.run_ap_mclachlan(bridge, times=(0.0, 0.005, 0.01, 0.015, 0.02))
    assert apm.status == bm.STATUS_CPU_VERIFIED
    assert apm.checks["sector_leakage_max_norm"] < 1.0e-10
    # measured 2026-10-19 on this witness: min fidelity 0.99937, drift 3.4e-6
    assert apm.checks["min_fidelity_vs_exact"] > 0.999
    assert apm.checks["energy_drift_max_abs"] < 1.0e-4
    assert apm.checks["initial_energy_minus_hf_witness_abs"] < 1.0e-10
    assert apm.results["parameterization_mode"] == "logical_shared"


def test_per_pauli_term_coordinates_can_leak_particle_number_while_logical_shared_conserves(tmp_path):
    """Measured confirmation of DISPOSITION 026-4: executed per-word coordinates are
    not sector-safe by construction; logical_shared (one angle per generator) is."""

    manifest = _manifest(tmp_path, "random8")
    bridge = bm.resolve_bridge_context(manifest["manifest_path"])
    grid = (0.0, 0.005, 0.01, 0.015, 0.02)  # measured 2026-10-19: leak 1.1e-17 (shared) vs 3.3e-5 (per word)
    shared = bm.run_ap_mclachlan(bridge, times=grid, parameterization_mode="logical_shared")
    assert shared.checks["sector_leakage_max_norm"] < 1.0e-10 and shared.status == bm.STATUS_CPU_VERIFIED
    per_word = bm.run_ap_mclachlan(bridge, times=grid, parameterization_mode="per_pauli_term")
    assert per_word.checks["sector_leakage_max_norm"] > 1.0e-6
    assert per_word.status == bm.STATUS_FAILED


def test_frozen_core_witness_runs_end_to_end(tmp_path):
    manifest = _manifest(tmp_path, "random6_two_occupied", frozen=1)
    assert manifest["reduction"]["kind"] == "frozen_core_occupied_prefix"
    summary = bm.run_bridge(manifest["manifest_path"], backend_spec="numpy:float64", methods=("mapping", "exact", "qse", "adapt"))
    assert all(status == bm.STATUS_CPU_VERIFIED for status in summary["status_by_method"].values()), summary["status_by_method"]
    receipts = {receipt["method"]: receipt for receipt in summary["receipts"]}
    assert receipts["qse"]["checks"]["ritz_minus_exact_sector_roots_max_abs"] < 1.0e-10
    assert summary["inputs"]["reduction"]["n_frozen"] == 1
    assert summary["intake_receipt"]["declarations"]["scalar_semantics"] == "nuclear_plus_frozen_core"


def test_unavailable_backend_yields_skipped_receipt_not_cpu_fallback(tmp_path):
    manifest = _manifest(tmp_path, "contract4")
    bridge = bm.resolve_bridge_context(manifest["manifest_path"])
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            pytest.skip("cuda available; cannot demonstrate the unavailable path here")
    except ModuleNotFoundError:
        pass
    receipt = bm.run_qse(bridge, "torch:cuda:float64")
    assert receipt.status == bm.STATUS_SKIPPED_UNSUPPORTED
    assert receipt.backend["available"] is False and receipt.backend["exercised_device"] is None
    assert receipt.results == {} and receipt.notes and "unavailable" in receipt.notes[0]


def test_float32_numpy_backend_is_cpu_verified_at_its_own_tolerance(tmp_path):
    manifest = _manifest(tmp_path, "random6")
    bridge = bm.resolve_bridge_context(manifest["manifest_path"])
    receipt = bm.run_mapping_witness(bridge, "numpy:float32")
    assert receipt.status == bm.STATUS_CPU_VERIFIED
    assert receipt.backend["real_dtype"] == "float32" and receipt.backend["complex_dtype"] == "complex64"
    assert receipt.checks["hf_expectation_minus_tensor_witness_abs"] < 5.0e-5
