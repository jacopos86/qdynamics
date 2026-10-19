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
    elements = qdynamics_matrix_elements_from_mapping(mapping, source={"kind": kind, "simulation": True, "producer_generated": False}, declared_origin="synthetic")
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
    assert summary["inputs"]["data_role"] == "synthetic:in_memory_mapping"
    assert summary["inputs"]["origin"] == "synthetic" and summary["inputs"]["producer_origin_established"] is False
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
    # With the executed-sector filter the full_meta pool offers no sector-safe even-Y
    # multi-word generator (the two-body block is grouped-safe but its words do not
    # commute), so the controller cannot move the state: the fidelity trace is the
    # exact state's departure from HF (measured 0.99997516 at t=0.1), not a
    # variational error.  Unfiltered, the two-body block gave 0.9999999999992 but is
    # not execution-safe in general (finding 1).
    assert apm["results"]["executed_parameterization_sector_safe"] is True
    assert apm["results"]["sector_audit"]["audit_failing_by_class"] == {"ham_block": 1, "ham_term": 4}
    assert apm["results"]["trajectory_moved"] is False
    assert apm["checks"]["min_fidelity_vs_exact"] > 0.9999
    assert apm["checks"]["energy_drift_max_abs"] < 1.0e-12
    assert apm["results"]["parameterization_mode"] == "logical_shared"
    assert any("cannot move" in note or "zero" in note for note in apm["notes"]) or apm["results"]["sector_audit"]["even_y_eligible_count"] > 0
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
    # measured 2026-10-19: filtered pool 29/127 eligible (4+4 UCC, 21 Z-only words),
    # no sector-safe even-Y multi-word generator -> trajectory frozen at HF; the exact
    # state's departure gives min fidelity 0.99937 at t=0.02 (unfiltered: 0.99937427).
    assert apm.results["executed_parameterization_sector_safe"] is True
    assert apm.results["sector_audit"]["eligible_count"] == 29 and apm.results["sector_audit"]["generator_count"] == 127
    assert apm.checks["min_fidelity_vs_exact"] > 0.999
    assert apm.checks["initial_energy_minus_hf_witness_abs"] < 1.0e-10
    assert apm.results["parameterization_mode"] == "logical_shared"


def test_finding1_logical_shared_alone_leaks_and_the_executed_audit_filter_restricts(tmp_path):
    # Review finding 1: with the unfiltered full_meta pool, logical_shared execution on
    # random6 leaked 1.236e-4 under max_structural_pool_size=128 / mclachlan_l2 / l2 cut 1e-6.
    manifest = _manifest(tmp_path, "random6")
    bridge = bm.resolve_bridge_context(manifest["manifest_path"])
    overrides = {"max_structural_pool_size": 128, "insertion_gate_mode": "mclachlan_l2", "insertion_l2_cut": 1.0e-6}
    unfiltered = bm.run_ap_mclachlan(bridge, times=(0.0, 0.02, 0.04, 0.06), parameterization_mode="logical_shared", support_config_overrides=overrides, candidate_sector_filter="none")
    assert unfiltered.status == bm.STATUS_FAILED
    assert unfiltered.checks["sector_leakage_max_norm"] > 1.0e-6
    assert unfiltered.results["executed_parameterization_sector_safe"] is False
    assert unfiltered.results["sector_audit"]["audit_failing_count"] > 0
    filtered = bm.run_ap_mclachlan(bridge, times=(0.0, 0.02, 0.04, 0.06), parameterization_mode="logical_shared", support_config_overrides=overrides)
    assert filtered.status == bm.STATUS_CPU_VERIFIED
    assert filtered.checks["sector_leakage_max_norm"] < 1.0e-10
    assert filtered.results["executed_parameterization_sector_safe"] is True
    audit = filtered.results["sector_audit"]
    assert audit["criterion"] == "execution_preserves_fixed_counts"
    assert audit["eligible_count"] + audit["audit_failing_count"] == audit["generator_count"]
    assert audit["audit_failing_by_class"].get("ham_term", 0) > 0  # single X/Y words are not conserved by shared angles
    assert audit["audit_failing_by_class"].get("ham_block", 0) > 0  # noncommuting Hamiltonian blocks are not either
    assert set(filtered.results["sector_audit"]["eligible_by_class"]) <= {"uccsd_sing", "uccsd_dbl", "ham_term"}
    assert filtered.results["candidate_count_offered"] == audit["eligible_count"] - 1


def test_finding1_per_pauli_term_leaks_without_the_filter_and_is_restricted_with_it(tmp_path):
    manifest = _manifest(tmp_path, "random8")
    bridge = bm.resolve_bridge_context(manifest["manifest_path"])
    grid = (0.0, 0.005, 0.01, 0.015, 0.02)  # measured 2026-10-19: leak 3.3e-5 unfiltered
    per_word = bm.run_ap_mclachlan(bridge, times=grid, parameterization_mode="per_pauli_term", candidate_sector_filter="none")
    assert per_word.checks["sector_leakage_max_norm"] > 1.0e-6 and per_word.status == bm.STATUS_FAILED
    restricted = bm.run_ap_mclachlan(bridge, times=grid, parameterization_mode="per_pauli_term")
    assert restricted.status == bm.STATUS_CPU_VERIFIED and restricted.checks["sector_leakage_max_norm"] < 1.0e-10
    assert restricted.results["sector_audit"]["criterion"] == "components_individually_preserve_fixed_counts"
    with pytest.raises(ValueError, match="not an eligible generator"):
        bm.run_ap_mclachlan(bridge, times=grid, seed_term_label="ham_block::molecular_one_body")


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


def test_finding4_static_zero_drive_reference_is_translation_invariant(tmp_path):
    # Review probe: grids (0,.02,.04) and (1,1.02,1.04) gave identical energies but
    # initial fidelities 1.0 vs 0.99876 because the reference was clocked from t=0.
    manifest = _manifest(tmp_path, "contract4")
    bridge = bm.resolve_bridge_context(manifest["manifest_path"])
    zero = bm.run_ap_mclachlan(bridge, times=(0.0, 0.02, 0.04))
    shifted = bm.run_ap_mclachlan(bridge, times=(1.0, 1.02, 1.04))
    assert zero.results["initial_time"] == 0.0 and shifted.results["initial_time"] == 1.0
    assert shifted.results["elapsed_times_from_initial_checkpoint"] == pytest.approx([0.0, 0.02, 0.04], abs=1.0e-12)
    np.testing.assert_allclose(zero.results["energies"], shifted.results["energies"], atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(zero.results["fidelity_vs_exact_sector_propagation"], shifted.results["fidelity_vs_exact_sector_propagation"], atol=1.0e-12, rtol=0.0)
    assert zero.results["fidelity_vs_exact_sector_propagation"][0] > 1.0 - 1.0e-12
    assert shifted.results["fidelity_vs_exact_sector_propagation"][0] > 1.0 - 1.0e-12
    with pytest.raises(ValueError):
        bm.run_ap_mclachlan(bridge, times=(0.0, 0.02, 0.01))
    with pytest.raises(ValueError):
        bm.run_ap_mclachlan(bridge, times=(0.0,))


def test_finding5_method_requests_are_validated_and_exit_codes_follow_statuses(tmp_path):
    for bad in ("qes", "", ",", "mapping,qes"):
        with pytest.raises(ValueError):
            bm.normalize_methods(bad)
    assert bm.normalize_methods("qse, mapping ,qse") == ("qse", "mapping")
    assert bm.exit_code_for_statuses({"qse": bm.STATUS_CPU_VERIFIED, "mapping": bm.STATUS_GPU_VERIFIED}) == bm.EXIT_ALL_VERIFIED
    assert bm.exit_code_for_statuses({"qse": bm.STATUS_CPU_VERIFIED, "apm": bm.STATUS_FAILED}) == bm.EXIT_CHECK_FAILED
    assert bm.exit_code_for_statuses({"qse": bm.STATUS_SKIPPED_UNSUPPORTED}) == bm.EXIT_NOT_EXECUTED
    assert bm.exit_code_for_statuses({"qse": bm.STATUS_BLOCKED, "mapping": bm.STATUS_CPU_VERIFIED}) == bm.EXIT_NOT_EXECUTED
    assert bm.exit_code_for_statuses({}) == bm.EXIT_USAGE
    manifest = _manifest(tmp_path, "contract4")
    with pytest.raises(ValueError, match="unknown methods"):
        bm.run_bridge(manifest["manifest_path"], methods=("qes",))
    with pytest.raises(ValueError, match="no methods requested"):
        bm.run_bridge(manifest["manifest_path"], methods=())
    summary = bm.run_bridge(manifest["manifest_path"], methods=("mapping", "exact"))
    assert summary["exit_code"] == bm.EXIT_ALL_VERIFIED and summary["requested_methods"] == ["mapping", "exact"]
