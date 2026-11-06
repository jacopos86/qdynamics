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
    summary = bm.run_bridge(manifest["manifest_path"], backend_spec="numpy:float64", out_dir=tmp_path / "out", apm_pool_key="full_meta")
    statuses = summary["status_by_method"]
    assert set(statuses) == {"mapping_witness", "exact_sector_reference", "qse", "static_adapt_selection_gradients", "ra_adapt_construction", "ap_mclachlan"}
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
    apm = bm.run_ap_mclachlan(bridge, times=(0.0, 0.005, 0.01, 0.015, 0.02), candidate_pool_key="full_meta")
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
    unfiltered = bm.run_ap_mclachlan(bridge, times=(0.0, 0.02, 0.04, 0.06), candidate_pool_key="full_meta", parameterization_mode="logical_shared", support_config_overrides=overrides, candidate_sector_filter="none")
    assert unfiltered.status == bm.STATUS_FAILED
    assert unfiltered.checks["sector_leakage_max_norm"] > 1.0e-6
    assert unfiltered.results["executed_parameterization_sector_safe"] is False
    assert unfiltered.results["sector_audit"]["audit_failing_count"] > 0
    filtered = bm.run_ap_mclachlan(bridge, times=(0.0, 0.02, 0.04, 0.06), candidate_pool_key="full_meta", parameterization_mode="logical_shared", support_config_overrides=overrides)
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
    per_word = bm.run_ap_mclachlan(bridge, times=grid, candidate_pool_key="full_meta", parameterization_mode="per_pauli_term", candidate_sector_filter="none")
    assert per_word.checks["sector_leakage_max_norm"] > 1.0e-6 and per_word.status == bm.STATUS_FAILED
    restricted = bm.run_ap_mclachlan(bridge, times=grid, candidate_pool_key="full_meta", parameterization_mode="per_pauli_term")
    assert restricted.status == bm.STATUS_CPU_VERIFIED and restricted.checks["sector_leakage_max_norm"] < 1.0e-10
    assert restricted.results["sector_audit"]["criterion"] == "components_individually_preserve_fixed_counts"
    with pytest.raises(ValueError, match="not an eligible generator"):
        bm.run_ap_mclachlan(bridge, times=grid, candidate_pool_key="full_meta", seed_term_label="ham_block::molecular_one_body")


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


# ---------------------------------------------------------------------------
# 2026-10-20: Hermitian number-conserving excitation pool for real-time dynamics
# ---------------------------------------------------------------------------


def _even_y_real_nondiagonal(term) -> bool:
    words = [(str(t.pw2strng()), complex(t.p_coeff)) for t in term.polynomial.return_polynomial() if abs(complex(t.p_coeff)) > 0.0]
    return bool(words) and all(w.count("y") % 2 == 0 and c.imag == 0.0 for w, c in words) and any(("x" in w or "y" in w) for w, _ in words)


def test_hermitian_pool_is_audited_safe_even_y_and_pairs_with_uccsd(tmp_path):
    manifest = _manifest(tmp_path, "random6")
    bridge = bm.resolve_bridge_context(manifest["manifest_path"])
    assert "hermitian_excitations" in bm.admissible_pool_keys(bridge) and "full_meta" in bm.admissible_pool_keys(bridge)
    pool, meta = bm.resolve_molecular_pool(bridge, "hermitian_excitations")
    assert meta["pool_key"] == "hermitian_excitations" and meta["bridge_local"] is True and meta["pool_size"] == len(pool) == 8
    uccsd, umeta = bm.resolve_molecular_pool(bridge, "uccsd")
    assert umeta["bridge_local"] is False and len(uccsd) == len(pool)
    # one Hermitian block per UCCSD generator, same index labels
    assert sorted(str(t.label).split("(", 1)[1] for t in pool) == sorted(str(t.label).split("(", 1)[1] for t in uccsd)
    assert all(_even_y_real_nondiagonal(term) for term in pool)
    assert not any(_even_y_real_nondiagonal(term) for term in uccsd)  # odd-Y, real-orthogonal exponentials
    audit, eligible = bm._sector_safe_candidates(bridge, pool, parameterization_mode="logical_shared", candidate_sector_filter="executed_fixed_counts")
    assert audit["eligible_count"] == len(pool) and audit["audit_failing_count"] == 0 and audit["even_y_eligible_count"] == len(pool)
    # the words of one block mutually commute, so the shared-angle product is the exact exponential
    cpu = bm.NumpyBackend("float64")
    psi = bm.hf_statevector(bridge)
    h_bound = bm.hamiltonian_action(bridge).bind(cpu)
    v_herm = bm.initial_mclachlan_velocities(h_bound, pool, psi, cpu)
    v_ucc = bm.initial_mclachlan_velocities(h_bound, uccsd, psi, cpu)
    assert np.max(np.abs(v_ucc)) == 0.0  # a real reference gives every odd-Y generator zero McLachlan velocity
    assert np.max(np.abs(v_herm)) > 0.5  # measured 0.9437 on random6


def test_hermitian_pool_moves_the_trajectory_and_tracks_exact_propagation(tmp_path):
    manifest = _manifest(tmp_path, "random6")
    bridge = bm.resolve_bridge_context(manifest["manifest_path"])
    grid = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10)
    apm = bm.run_ap_mclachlan(bridge, times=grid)  # default pool = hermitian_excitations
    assert apm.status == bm.STATUS_CPU_VERIFIED
    assert apm.results["pool"]["pool_key"] == "hermitian_excitations"
    assert apm.results["seed_selection"] == "auto_max_initial_velocity_v1"
    assert apm.results["seed_term"] == "herm_dbl(ab:0,3->2,5)" and apm.results["seed_abs_initial_velocity"] > 0.5
    assert apm.results["initial_velocity_nonzero_count"] == 8
    assert apm.results["trajectory_moved"] is True and apm.results["theta_max_abs"] > 1.0e-3
    assert apm.results["final_runtime_parameter_count"] > 1  # the exchange controller appended support (measured 6)
    assert apm.checks["sector_leakage_max_norm"] < 1.0e-12
    assert apm.checks["exact_energy_drift_max_abs"] < 1.0e-9  # static H: exact propagation conserves energy
    assert apm.checks["min_fidelity_vs_exact"] > 0.999  # measured 0.99932 at t=0.10
    # the frozen control on the same witness (repository full_meta pool, class-rank fallback seed)
    frozen = bm.run_ap_mclachlan(bridge, times=grid, candidate_pool_key="full_meta")
    assert frozen.results["seed_selection"] == "auto_class_rank_fallback_all_initial_velocities_zero"
    assert frozen.results["initial_velocity_nonzero_count"] == 0 and frozen.results["trajectory_moved"] is False
    assert apm.checks["min_fidelity_vs_exact"] > frozen.checks["min_fidelity_vs_exact"]


def test_hermitian_pool_contract_witness_single_double_is_near_exact(tmp_path):
    manifest = _manifest(tmp_path, "contract4")
    bridge = bm.resolve_bridge_context(manifest["manifest_path"])
    pool, meta = bm.resolve_molecular_pool(bridge, "hermitian_excitations")
    assert [str(t.label) for t in pool] == ["herm_sing(alpha:0->1)", "herm_sing(beta:2->3)", "herm_dbl(ab:0,2->1,3)"]
    apm = bm.run_ap_mclachlan(bridge, times=(0.0, 0.02, 0.04, 0.06, 0.08, 0.10))
    assert apm.status == bm.STATUS_CPU_VERIFIED and apm.results["trajectory_moved"] is True
    assert apm.results["seed_term"] == "herm_dbl(ab:0,2->1,3)" and abs(apm.results["seed_abs_initial_velocity"] - 0.05) < 1.0e-12
    assert apm.checks["min_fidelity_vs_exact"] > 0.9999990  # measured 0.99999951 at t=0.10
    assert apm.checks["sector_leakage_max_norm"] == 0.0
    with pytest.raises(ValueError, match="rejected by SupportPatchControllerConfig"):
        bm.parse_apm_support_config('{"no_such_field": 1}')
    assert bm.parse_apm_support_config("") == {} and bm.parse_apm_support_config({"max_structural_pool_size": 16}) == {"max_structural_pool_size": 16}


# ---------------------------------------------------------------------------
# 2026-10-20: excitation-level QSE bases from products of commuting generators
# ---------------------------------------------------------------------------


def test_qse_product_bases_tighten_to_the_exact_sector_spectrum(tmp_path):
    manifest = _manifest(tmp_path, "random8")
    bridge = bm.resolve_bridge_context(manifest["manifest_path"])
    exact = bm.run_exact_reference(bridge, "numpy:float64", n_roots=36)  # every sector root, so the CISDTQ check covers all 36
    assert exact.checks["sector_dimension"] == 36 and len(exact.results["low_roots"]) == 36
    cisd = bm.run_qse(bridge, "numpy:float64", basis_kind="uccsd_singles_doubles", exact_roots=exact.results["low_roots"])
    cisdt = bm.run_qse(bridge, "numpy:float64", basis_kind="uccsd_sdt_products", exact_roots=exact.results["low_roots"])
    cisdtq = bm.run_qse(bridge, "numpy:float64", basis_kind="uccsd_sdtq_products", exact_roots=exact.results["low_roots"])
    assert (cisd.checks["basis_size"], cisd.checks["retained_rank"], cisd.results["basis_spans_sector"]) == (27, 27, False)
    assert cisd.results["basis_elements_by_excitation_level"] == {"1": 8, "2": 18}
    assert (cisdt.checks["basis_size"], cisdt.checks["retained_rank"], cisdt.results["basis_spans_sector"]) == (35, 35, False)
    assert cisdt.results["basis_elements_by_excitation_level"] == {"1": 8, "2": 18, "3": 8}
    assert (cisdtq.checks["basis_size"], cisdtq.checks["retained_rank"], cisdtq.results["basis_spans_sector"]) == (36, 36, True)
    assert cisdtq.results["basis_elements_by_excitation_level"] == {"1": 8, "2": 18, "3": 8, "4": 1}
    assert all(r.status == bm.STATUS_CPU_VERIFIED for r in (cisd, cisdt, cisdtq))
    e0 = exact.results["low_roots"][0]
    gap_cisd, gap_cisdt, gap_cisdtq = (r.results["ritz_roots"][0] - e0 for r in (cisd, cisdt, cisdtq))
    assert gap_cisd > gap_cisdt > 1.0e-6 and abs(gap_cisdtq) < 1.0e-10  # measured 0.7445, 7.05e-3, -7e-15
    assert cisdtq.results["exact_roots_compared"] == 36 and cisdtq.checks["ritz_minus_exact_sector_roots_max_abs"] < 1.0e-9  # all 36 sector roots reproduced
    for r in (cisd, cisdt, cisdtq):
        assert r.checks["ritz_below_exact_max"] < 1.0e-9  # Ritz values never fall below the exact sector roots
    with pytest.raises(ValueError, match="basis_kind must be one of"):
        bm.run_qse(bridge, "numpy:float64", basis_kind="no_such_basis")
    # product elements are labelled by their factors and deduplicated per excited determinant
    labels = cisdtq.results["basis_labels"]
    assert sum(label.startswith("prod[") for label in labels) == 9 and len(set(labels)) == len(labels)


# ---------------------------------------------------------------------------
# 2026-10-20: RA-ADAPT construction through the named molecular application
# ---------------------------------------------------------------------------


def test_ra_adapt_construction_receipt_runs_the_named_application(tmp_path):
    manifest = _manifest(tmp_path, "random6")
    bridge = bm.resolve_bridge_context(manifest["manifest_path"])
    receipt = bm.run_ra_adapt_construction(bridge, rounds=4, insertion="plateau_commutation")
    assert receipt.status == bm.STATUS_CPU_VERIFIED and receipt.method == "ra_adapt_construction" and receipt.lane == "paper_i"
    checks = receipt.checks
    assert checks["accepted_rounds"] == 4 and checks["accepted_energies_nonincreasing_from_hf"] is True and checks["final_energy_not_below_exact"] is True
    assert checks["executed_problem_is_molecular_family"] is True and checks["executed_pool_matches_request"] is True and checks["executed_phase3_cost_mode"] == "proxy"
    assert checks["route_profile_names_molecular_lane"] is True and checks["hf_energy_minus_hf_witness_abs"] < 1.0e-12
    res = receipt.results
    assert res["exact_reference_role"] == "reporting_input_sector_block_eigvalsh" and res["stop"]["primary_reason"] == "maximum_controller_rounds"
    assert res["exact_reference_used_online"] is False and len(res["route_contract_sha256"]) == 64
    assert abs(res["final_energy"] - (-1.608561)) < 1.0e-5 and abs(res["exact_ground_energy"] - (-1.750066)) < 1.0e-5  # measured 2026-10-20
    assert res["final_operators"] == ["uccsd_dbl(ab:0,3->2,5)", "uccsd_dbl(ab:0,3->2,4)", "uccsd_sing(alpha:0->2)", "uccsd_sing(alpha:0->1)"]
    assert res["parent_pool_count"] == 8 and res["executable_pool_count"] == 8
    with pytest.raises(ValueError, match="not a repository pool key"):
        bm.run_ra_adapt_construction(bridge, rounds=1, pool_key="hermitian_excitations")
    with pytest.raises(ValueError, match="at least 1"):
        bm.validate_ra_construction_request(0, "append_only", "uccsd")
    with pytest.raises(ValueError, match="insertion must be one of"):
        bm.validate_ra_construction_request(2, "beam", "uccsd")
    # the facade-gate record for the default request still says locked (unchanged contract)
    gate = bm.facade_admission_record(bridge)
    assert gate["admitted"] is False and "locked" in gate["message"]


def test_run_bridge_adapt_method_yields_screening_and_construction_receipts(tmp_path):
    manifest = _manifest(tmp_path, "contract4")
    summary = bm.run_bridge(manifest["manifest_path"], backend_spec="numpy:float64", methods=("exact", "adapt", "ra"), out_dir=tmp_path / "out", ra_rounds=1, ra_insertion="append_only")
    assert set(summary["status_by_method"]) == {"exact_sector_reference", "static_adapt_selection_gradients", "ra_adapt_construction"}
    assert all(v == bm.STATUS_CPU_VERIFIED for v in summary["status_by_method"].values()), summary["status_by_method"]
    construction = [r for r in summary["receipts"] if r["method"] == "ra_adapt_construction"][0]
    assert construction["results"]["final_operators"] == ["uccsd_dbl(ab:0,2->1,3)"]  # same generator the selection gradients rank first
    # the default 8-round horizon is capped at the 3-generator executable pool on this witness
    capped = bm.run_ra_adapt_construction(bridge := bm.resolve_bridge_context(manifest["manifest_path"]), rounds=8)
    assert capped.status == bm.STATUS_CPU_VERIFIED and capped.results["request"]["maximum_controller_rounds_executed"] == 3
    assert abs(capped.checks["final_energy_minus_exact_sector_ground"]) < 1.0e-12  # exact within the 3-generator span
    assert construction["checks"]["final_energy_minus_exact_sector_ground"] >= -1.0e-9
    assert summary["exit_code"] == bm.EXIT_ALL_VERIFIED
