"""Paper IV lane fixtures from vibronic models whose coupling is given analytically (Doron Wu's model among them)."""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "test_support") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "test_support"))

from h2o_linear_fd_fixture_factory import synthetic_three_mode_h2o_linear_fd_backend_record  # noqa: E402
from pipelines.static_adapt.ra_adapt.contracts import CANDIDATE_REPRESENTATION_MACRO  # noqa: E402
from pipelines.static_adapt.ra_adapt.h2o_application import _problem  # noqa: E402
from pipelines.static_adapt.ra_adapt.pools import (  # noqa: E402
    EXPECTED_H2O_DERIVATIVE_RESOLVED_PARENT_COUNT,
    build_parent_template_inventory,
    h2o_lane_active_space,
    require_h2o_symmetry_complete_problem,
)
from src.quantum.chemistry.generate_h2o_linear_fd_fixture import build_h2o_linear_fd_fixture_from_record  # noqa: E402
from src.quantum.chemistry.molecular_vibronic import (  # noqa: E402
    MODEL_SCHEMA,
    ActiveSpace,
    ActiveTensors,
    LinearCoupling,
    MolecularGeometry,
    NormalMode,
    assemble_vibronic_model,
    reference_force,
    subtract_reference_force,
)
from src.quantum.chemistry.paper_iv_h2o_fixture_from_builder import PAPER_IV_H2O_GEOMETRY  # noqa: E402
from src.quantum.chemistry.paper_iv_h2o_fixture_from_vibronic_model import (  # noqa: E402
    builder_model_in_his_conventions,
    fixture_from_vibronic_model,
    lane_mode_rows,
    write_paper_iv_h2o_fixture_from_vibronic_model,
)
from src.quantum.chemistry.qdynamics_interop import hartree_fock_vacuum_energy  # noqa: E402
from src.quantum.chemistry.vibronic_h2o_linear_fd import (  # noqa: E402
    H2O_LINEAR_FD_ANALYTIC_DERIVATIVE_SOURCE,
    H2O_LINEAR_FD_DERIVATIVE_SOURCE,
    load_production_vibronic_h2o_fixture,
    validate_paper_iv_h2o_linear_fd_evidence_fixture,
    validate_production_vibronic_h2o_fixture,
)

LANE = ("bend", "symmetric_stretch", "antisymmetric_stretch")
# His driver's outputs for H2O/STO-3G at the Paper IV geometry with isotope masses in the deck (his code at
# 38f40805 run here on 2026-11-04; see test_support/qdynamics_h2o/README.md), pinned by sha256.
QDYNAMICS_H2O_DIR = REPO_ROOT / "test_support" / "qdynamics_h2o"
QDYNAMICS_H2O_SHA256 = {
    "h2o_optiso.inp": "862c925076c6cdc0e773319214a6b4eefa0d4dd6de4a30a85cd211ed39b0bb79",
    "h2o_optiso_ele.h5": "c7c3d5b8c77cd7c8087a92421359c6f329a425e20a981cd2be416816de359b8f",
    "h2o_optiso_eph.h5": "6407d0e441321bb758a3957f75ccdb5c1a0c727d017945522c6e43db4d653ff8",
    "h2o_optiso_vib.h5": "fe5ed93fdc4d06b051f27a81f5c4128585bcc1f869ff32703b17860e953da350",
    "his_optiso_c1.json": "06f7e8f4afad31a249b0f6051630579db68577339f21d3897e9e667212d662c4",
    "his_optiso_c3.json": "c7866448184c545a236151ded668558e18b1ba52e9a55d7cc4549633711d00f6",
}
CHECKED_IN_FIXTURE = REPO_ROOT / "test_support" / "paper_iv_h2o" / "h2o_builder_model_consistent_work333_ref444_20261103_fixture.json"
HIS_CONVENTIONS_FREQUENCIES_CM1 = (2170.79, 4138.69, 4389.24)   # RHF Hessian, our isotope masses, the Paper IV geometry


@pytest.fixture(scope="module")
def production_fixture():
    return load_production_vibronic_h2o_fixture(CHECKED_IN_FIXTURE)


def _synthetic_model(production, *, cutoffs=(1, 1, 1), counter_term=False):
    """Two orbitals, one electron pair, the production fixture's real H2O modes handed over scrambled
    (order asym, bend, sym; the bend vector negated) with one-body couplings of the eph kind."""
    geometry = MolecularGeometry(production.geometry.symbols, np.asarray(production.geometry.coordinates_bohr))
    lane_modes = {m.label: m for m in production.normal_modes}
    scrambled = (("antisymmetric_stretch", 1.0), ("bend", -1.0), ("symmetric_stretch", 1.0))
    modes = tuple(NormalMode(a, float(lane_modes[label].frequency_hartree), sign * np.asarray(lane_modes[label].mass_weighted_eigenvector),
                             f"model_mode_{a}") for a, (label, sign) in enumerate(scrambled))
    h = np.array([[-1.2, 0.05], [0.05, -0.3]])
    g = np.zeros((2, 2, 2, 2))
    g[0, 0, 0, 0], g[1, 1, 1, 1] = 0.6, 0.5
    g[0, 0, 1, 1] = g[1, 1, 0, 0] = 0.4
    g[0, 1, 1, 0] = g[1, 0, 0, 1] = g[0, 1, 0, 1] = g[1, 0, 1, 0] = 0.1
    rng = np.random.default_rng(7)
    couplings = []
    for a in range(3):
        c = 0.03 * rng.standard_normal((2, 2))
        couplings.append(LinearCoupling(a, "mean_field_pyscf_eph", 0.0, 0.5 * (c + c.T), None, 0.0, {"synthetic": True}))
    if counter_term:
        couplings = list(subtract_reference_force(couplings, 1))
    space = ActiveSpace((), (0, 1), (), 1, 1, "synthetic_two_orbital")
    model = assemble_vibronic_model(geometry=geometry, basis="synthetic", active_space=space, active_tensors=ActiveTensors(0.1, h, g),
                                    modes=modes, couplings=couplings, cutoffs=cutoffs,
                                    provenance={"schema": MODEL_SCHEMA, "harmonic_curvature": "synthetic", "mode_basis": "synthetic",
                                                "reference_force_counter_term": counter_term,
                                                "masses_me": np.asarray(production.geometry.masses_me).tolist()})
    return model


def test_lane_rows_label_and_orient_the_scrambled_modes(production_fixture):
    model = _synthetic_model(production_fixture)
    rows = lane_mode_rows(model, np.asarray(production_fixture.geometry.masses_me))
    assert tuple(r["label"] for r in rows) == LANE
    assert [r["builder_mode_index"] for r in rows] == [1, 2, 0]
    assert [r["orientation_sign_vs_builder_mode"] for r in rows] == [-1.0, 1.0, 1.0]
    for row, production_mode in zip(rows, production_fixture.normal_modes):
        np.testing.assert_allclose(row["mass_weighted_eigenvector"], production_mode.mass_weighted_eigenvector, atol=1e-12)
        assert row["frequency_hartree"] == pytest.approx(production_mode.frequency_hartree)


def test_synthetic_model_fixture_passes_the_gate_matches_the_model_and_locks_the_pool(production_fixture, tmp_path):
    model = _synthetic_model(production_fixture)
    summary = write_paper_iv_h2o_fixture_from_vibronic_model(tmp_path, tag="syn", model=model, mode_cutoffs=(1, 1, 1),
                                                             source={"kind": "synthetic"})
    fixture_path = tmp_path / summary["fixture_file"]
    fixture = load_production_vibronic_h2o_fixture(fixture_path)
    validate_paper_iv_h2o_linear_fd_evidence_fixture(fixture)
    assert summary["derivative_source"] == H2O_LINEAR_FD_ANALYTIC_DERIVATIVE_SOURCE == fixture.manifest.derivative_source
    assert summary["mode_labels"] == list(LANE) and summary["n_total_qubits"] == 7 and summary["sector_dimension"] == 4 * 8
    assert summary["paper_iv_active_space_variant"] == "full_mo_space" == fixture.report_summary["paper_iv_active_space_variant"]
    assert summary["lane_orientation_by_builder_mode"] == {"1": -1.0, "2": 1.0, "0": 1.0}
    assert summary["builder_order_cutoffs"] == [1, 1, 1] and summary["reference_cutoffs"] == [2, 2, 2]
    # The fixture's Hamiltonian is the model's operator, term by term, after the lane's relabelling.
    own = summary["own_term_comparison"]
    assert own["compared"] and own["equal_within_atol"] and own["register_signs"] == [-1, 1, 1]
    assert own["payload_mode_order_for_lane_modes"] == [1, 2, 0]
    assert summary["model_minus_fixture_hartree"] == pytest.approx(0.0, abs=1e-10)
    assert summary["hartree_fock_vacuum_energy_hartree"] == pytest.approx(hartree_fock_vacuum_energy(model), abs=1e-10)
    # The derivative records: the model's couplings in lane orientation, no displaced tensors, the HF force.
    for deriv in fixture.first_derivatives:
        diag = deriv.finite_difference_diagnostics
        a, sign = int(diag["builder_mode_index"]), float(diag["orientation_sign_vs_builder_mode"])
        np.testing.assert_allclose(deriv.one_body_derivative, sign * model.couplings[a].one_body, atol=1e-14)
        assert deriv.plus_aligned_tensor_id is None and deriv.finite_difference_drift is None and diag["derivative_method"] == "analytic"
        assert deriv.active_equilibrium_force == pytest.approx(sign * reference_force(model.couplings[a], 1), abs=1e-14)
    assert fixture.displacements == () and fixture.aligned_tensors == () and fixture.alignment_diagnostics == ()
    # The consumer path: registry resolution, the application lock on the declared space, the parent inventory lock.
    problem = _problem(fixture_path)
    space = h2o_lane_active_space(problem)
    assert (space["n_spatial_orbitals"], space["num_particles"], space["n_fermion_qubits"], space["variant"]) == (2, (1, 1), 4, "full_mo_space")
    assert space["expected_parent_count"] == summary["derivative_resolved_pool_v2_parent_count"] != EXPECTED_H2O_DERIVATIVE_RESOLVED_PARENT_COUNT
    require_h2o_symmetry_complete_problem(problem)
    inventory = build_parent_template_inventory(problem, representation_id=CANDIDATE_REPRESENTATION_MACRO)
    assert int(inventory.receipt.count) == summary["derivative_resolved_pool_v2_parent_count"]
    # A tampered count fails the lock; a variant fixture without a recorded count cannot be locked at all.
    payload = json.loads(fixture_path.read_text())
    tampered = tmp_path / "tampered_fixture.json"
    payload["report_summary"]["derivative_resolved_pool_v2_parent_count"] += 1
    tampered.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="drifted"):
        build_parent_template_inventory(_problem(tampered), representation_id=CANDIDATE_REPRESENTATION_MACRO)
    del payload["report_summary"]["derivative_resolved_pool_v2_parent_count"]
    unlocked = tmp_path / "unlocked_fixture.json"
    unlocked.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="records no derivative-resolved parent count"):
        build_parent_template_inventory(_problem(unlocked), representation_id=CANDIDATE_REPRESENTATION_MACRO)
    with pytest.raises(FileExistsError):
        write_paper_iv_h2o_fixture_from_vibronic_model(tmp_path, tag="syn", model=model)


def test_counter_term_model_has_no_reference_force(production_fixture):
    model = _synthetic_model(production_fixture, counter_term=True)
    fixture, rows, work = fixture_from_vibronic_model(model, mode_cutoffs=(1, 1, 1))
    assert all(abs(float(d.active_equilibrium_force)) < 1e-14 for d in fixture.first_derivatives)
    assert all(float(d.scalar_derivative_hartree_per_q) != 0.0 for d in fixture.first_derivatives)
    assert fixture.report_summary["model_definition"]["reference_force_counter_term"] is True


def test_validators_admit_only_declared_derivative_sources():
    fixture = build_h2o_linear_fd_fixture_from_record(synthetic_three_mode_h2o_linear_fd_backend_record(), mode_cutoffs=(1, 1, 1))
    assert fixture.manifest.derivative_source == H2O_LINEAR_FD_DERIVATIVE_SOURCE
    with pytest.raises(ValueError, match="unexpected derivative source"):
        validate_production_vibronic_h2o_fixture(replace(fixture, manifest=replace(fixture.manifest, derivative_source="finite_difference_v0")))
    # The manifest and every derivative record must agree.
    analytic_manifest = replace(fixture.manifest, derivative_source=H2O_LINEAR_FD_ANALYTIC_DERIVATIVE_SOURCE)
    with pytest.raises(ValueError, match="unexpected derivative source"):
        validate_production_vibronic_h2o_fixture(replace(fixture, manifest=analytic_manifest))
    analytic_records = tuple(replace(d, derivative_source=H2O_LINEAR_FD_ANALYTIC_DERIVATIVE_SOURCE, plus_aligned_tensor_id=None,
                                     minus_aligned_tensor_id=None, finite_difference_drift=None,
                                     finite_difference_diagnostics={"derivative_method": "analytic"})
                             for d in fixture.first_derivatives)
    analytic = replace(fixture, manifest=analytic_manifest, first_derivatives=analytic_records)
    validate_production_vibronic_h2o_fixture(analytic)
    validate_paper_iv_h2o_linear_fd_evidence_fixture(analytic)
    # An analytic record may not point at displaced tensors, and must say it is analytic and carry no drift.
    with pytest.raises(ValueError, match="must not reference displaced tensors"):
        validate_production_vibronic_h2o_fixture(replace(analytic, first_derivatives=(
            replace(analytic_records[0], plus_aligned_tensor_id="aligned_x"), *analytic_records[1:])))
    with pytest.raises(ValueError, match="analytic derivative must declare"):
        validate_paper_iv_h2o_linear_fd_evidence_fixture(replace(analytic, first_derivatives=(
            replace(analytic_records[0], finite_difference_drift=0.0), *analytic_records[1:])))
    with pytest.raises(ValueError, match="analytic derivative must declare"):
        validate_paper_iv_h2o_linear_fd_evidence_fixture(replace(analytic, first_derivatives=(
            replace(analytic_records[0], finite_difference_diagnostics={}), *analytic_records[1:])))
    # Finite-difference records still need their drift diagnostic.
    with pytest.raises(ValueError, match="drift diagnostic missing"):
        validate_paper_iv_h2o_linear_fd_evidence_fixture(replace(fixture, first_derivatives=(
            replace(fixture.first_derivatives[0], finite_difference_drift=None), *fixture.first_derivatives[1:])))


def test_his_conventions_frozen_core_model_through_the_lane(tmp_path):
    pytest.importorskip("pyscf")
    model, computation = builder_model_in_his_conventions(PAPER_IV_H2O_GEOMETRY, active_space="frozen_core")
    assert model.coupling_model == "mean_field_pyscf_eph" and model.provenance["reference_force_counter_term"] is False
    summary = write_paper_iv_h2o_fixture_from_vibronic_model(tmp_path, tag="fc", model=model, mode_cutoffs=(1, 1, 1),
                                                             masses_me=computation.masses_me,
                                                             source={"kind": "builder_in_qdynamics_conventions", "active_space": "frozen_core"})
    fixture = load_production_vibronic_h2o_fixture(tmp_path / summary["fixture_file"])
    assert summary["mode_labels"] == list(LANE) and summary["n_total_qubits"] == 15 and summary["active_space_kind"] == "cas_8e_6o_valence"
    assert summary["paper_iv_active_space_variant"] is None
    assert summary["mode_frequencies_cm1"] == pytest.approx(list(HIS_CONVENTIONS_FREQUENCIES_CM1), abs=0.05)
    assert summary["own_term_comparison"]["equal_within_atol"] and summary["model_minus_fixture_hartree"] == pytest.approx(0.0, abs=1e-8)
    # His model's reference energy is E_RHF + sum omega / 2 (the coupling has no vacuum expectation value) ...
    reference = float(computation.center.hf_energy) + 0.5 * sum(float(m.frequency_hartree) for m in model.modes)
    assert summary["hartree_fock_vacuum_energy_hartree"] == pytest.approx(reference, abs=1e-9)
    # ... and the uncompensated mean-field force pulls the ground state well below it.
    assert summary["fixture_ground_energy_hartree"] < reference - 0.05
    # The HF force acts on the totally symmetric modes only; the B2 antisymmetric stretch has none by symmetry.
    force = {str(d.mode_label): float(d.active_equilibrium_force) for d in fixture.first_derivatives}
    assert abs(force["bend"]) > 1e-3 and abs(force["symmetric_stretch"]) > 1e-3 and abs(force["antisymmetric_stretch"]) < 1e-8
    # One-body couplings only: fewer spectral factors than the full-linear pool, so the lock follows the recorded count.
    problem = _problem(tmp_path / summary["fixture_file"])
    space = h2o_lane_active_space(problem)
    assert (space["n_spatial_orbitals"], space["num_particles"], space["variant"]) == (6, (4, 4), None)
    assert space["expected_parent_count"] == summary["derivative_resolved_pool_v2_parent_count"] < EXPECTED_H2O_DERIVATIVE_RESOLVED_PARENT_COUNT
    require_h2o_symmetry_complete_problem(problem)
    inventory = build_parent_template_inventory(problem, representation_id=CANDIDATE_REPRESENTATION_MACRO)
    assert int(inventory.receipt.count) == summary["derivative_resolved_pool_v2_parent_count"]


def test_checked_in_qdynamics_h2o_outputs_match_their_pins():
    import hashlib

    for name, expected in QDYNAMICS_H2O_SHA256.items():
        assert hashlib.sha256((QDYNAMICS_H2O_DIR / name).read_bytes()).hexdigest() == expected, name


def test_his_driver_outputs_build_the_fixture_that_equals_his_json(tmp_path):
    """The lane fixture built from his HDF5 files carries exactly the operator his qubitizer wrote from them.
    HOLSTEIN_QDYNAMICS_H2O_DIR points the test at another set of his outputs (same file names)."""
    pytest.importorskip("h5py")
    from src.quantum.chemistry.qdynamics_vendored import vibronic_model_from_qdynamics_outputs

    directory = Path(os.environ.get("HOLSTEIN_QDYNAMICS_H2O_DIR", QDYNAMICS_H2O_DIR))
    model = vibronic_model_from_qdynamics_outputs(directory / "h2o_optiso_ele.h5", directory / "h2o_optiso_vib.h5", directory / "h2o_optiso_eph.h5")
    summary = write_paper_iv_h2o_fixture_from_vibronic_model(
        tmp_path, tag="his", model=model, mode_cutoffs=(1, 1, 1), compare_qdynamics_json=directory / "his_optiso_c1.json",
        source={"kind": "qdynamics_driver_outputs"})
    assert summary["active_space_kind"] == "full_mo_space" and summary["n_total_qubits"] == 17
    external = summary["external_qdynamics_comparison"]
    assert external["compared"] and external["n_only_in_a"] == 0 and external["n_only_in_b"] == 0
    assert external["max_abs_difference_shared"] < 1e-9
    assert summary["own_term_comparison"]["equal_within_atol"]
