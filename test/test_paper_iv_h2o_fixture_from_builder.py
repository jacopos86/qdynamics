"""The Paper IV H2O fixture produced from the vibronic builder's decided model (2026-11-02)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.static_adapt.builders.problem_registry import _sha256_and_byte_count  # noqa: E402
from src.quantum.chemistry.generate_h2o_linear_fd_fixture import BACKEND_RECORD_SCHEMA, sha256_jsonable  # noqa: E402
from src.quantum.chemistry.molecular_vibronic import HARTREE_TO_CM_INV  # noqa: E402
from src.quantum.chemistry.paper_iv_h2o_fixture_from_builder import (  # noqa: E402
    LANE_MODE_LABELS,
    PAPER_IV_H2O_GEOMETRY,
    builder_backend_record,
    lane_active_space_kind,
    write_paper_iv_h2o_fixture_from_builder,
)
from src.quantum.chemistry.vibronic_h2o_linear_fd import (  # noqa: E402
    load_cached_production_vibronic_h2o_linear_fd_fixture,
    load_production_vibronic_h2o_fixture,
    validate_paper_iv_h2o_linear_fd_evidence_fixture,
)

# The decided model at the Paper IV geometry: model-consistent modes in lane order (bend,
# symmetric_stretch, antisymmetric_stretch); measured 2026-11-02/03.
EXPECTED_LANE_FREQUENCIES_CM1 = (2130.8, 5314.4, 5061.8)
PRODUCTION_GROUND_ENERGY_333 = -74.9999118


@pytest.fixture(scope="module")
def record_and_model():
    pytest.importorskip("pyscf")
    return builder_backend_record(PAPER_IV_H2O_GEOMETRY)


def test_backend_record_carries_the_decided_model_in_lane_order(record_and_model):
    record, model, computation = record_and_model
    assert record["schema"] == BACKEND_RECORD_SCHEMA and record["backend"]["name"] == "pyscf"
    definition = record["backend"]["model_definition"]
    assert (definition["harmonic_curvature"], definition["alignment"], definition["mode_basis"]) == (
        "pes_consistent", "block_diagonal", "model_consistent")
    assert definition["decided_model"] is True and definition["decided_on"] == "2026-11-02"
    assert definition["coupling_model"] == "full_linear" and definition["geometry_source"] == "paper_iv_h2o_fixture_center_20260731"
    assert record["active_space"]["active_space_kind"] == "cas_8e_6o_valence" == lane_active_space_kind(computation.active_space)
    assert record["active_space"]["num_particles"] == [4, 4] and record["active_space"]["n_spatial_orbitals"] == 6
    modes = record["normal_modes"]
    assert tuple(m["label"] for m in modes) == LANE_MODE_LABELS
    assert [m["mode_index"] for m in modes] == [0, 1, 2]
    assert [m["frequency_cm1"] for m in modes] == pytest.approx(list(EXPECTED_LANE_FREQUENCIES_CM1), abs=0.5)
    # The lane labels follow the dominant RHF character: bend <- RHF mode 0, symmetric <- 1, antisymmetric <- 2;
    # the A1 pair mixes by 2.6 degrees, the B2 stretch is pure.
    assert [m["dominant_rhf_mode_index"] for m in modes] == [0, 1, 2]
    weights = np.array([m["rhf_mode_weights"] for m in modes])
    assert abs(abs(weights[2, 2]) - 1.0) < 1e-6 and abs(weights[0, 1]) == pytest.approx(0.0453, abs=5e-4)
    rhf_vectors = np.array(definition["rhf_normal_mode_vectors"])
    np.testing.assert_allclose(rhf_vectors, [m.mass_weighted_vector for m in computation.modes], atol=1e-14)
    # The record's mode vectors are the builder's model modes, oriented by the lane policy (each mode's own
    # internal-coordinate indicator positive); the weights carry that orientation, so they reproduce the
    # stored vector from the RHF modes, and the model modes themselves have no translation/rotation content.
    for m in modes:
        vector = np.array(m["mass_weighted_eigenvector"])
        a = int(m["builder_mode_index"])
        builder_vector = np.asarray(model.modes[a].mass_weighted_vector)
        assert float(m["orientation_sign_vs_builder_mode"]) == np.sign(float(np.sum(vector * builder_vector)))
        assert abs(abs(float(np.sum(vector * builder_vector))) - 1.0) < 1e-10
        assert m["mode_character"]["dominant_label"] == m["label"] and m["mode_character"]["orientation_indicators"][m["label"]] > 0.0
        np.testing.assert_allclose(sum(w * e for w, e in zip(m["rhf_mode_weights"], rhf_vectors)), vector, atol=1e-10)
        assert float(np.linalg.norm(vector)) == pytest.approx(1.0) and m["trans_rot_overlap"] < 1e-8
    assert definition["lane_orientation_by_builder_mode"] == {str(int(m["builder_mode_index"])): float(m["orientation_sign_vs_builder_mode"]) for m in modes}
    rows = record["aligned_tensors"]
    assert len(rows) == 12 and {(r["mode_label"], r["step_kind"], r["sign"]) for r in rows} == {
        (label, kind, sign) for label in LANE_MODE_LABELS for kind in ("primary", "alt") for sign in (1, -1)}
    for r in rows:
        diag = r["alignment"]
        assert diag["policy"] == "block_diagonal" and diag["passed"] and diag["min_singular_value"] > 0.999
        assert diag["alignment_residual_fro"] < 1e-5 and min(diag["block_min_singular_values"]) > 0.999
        assert r["mass_weighted_displacement_norm_au"] == pytest.approx(r["q_displacement_au"], rel=1e-10)
    # The generator's central difference along a model mode reproduces the builder's rotated coupling
    # (same step, same alignment), up to the orientation sign of the lane's mode vector.
    for m in modes:
        label, a = m["label"], int(m["builder_mode_index"])
        plus = next(r for r in rows if r["mode_label"] == label and r["step_kind"] == "primary" and r["sign"] == 1)
        minus = next(r for r in rows if r["mode_label"] == label and r["step_kind"] == "primary" and r["sign"] == -1)
        fd = (np.array(plus["one_body_integrals"]) - np.array(minus["one_body_integrals"])) / 0.2
        builder = np.asarray(model.couplings[a].one_body)
        sign = np.sign(float(np.sum(np.array(m["mass_weighted_eigenvector"]) * np.asarray(model.modes[a].mass_weighted_vector))))
        np.testing.assert_allclose(fd, sign * builder, atol=2e-6, rtol=0)


def _check_fixture(summary, output_dir, *, work_cutoffs):
    fixture_path = Path(output_dir) / summary["fixture_file"]
    fixture = load_production_vibronic_h2o_fixture(fixture_path)
    validate_paper_iv_h2o_linear_fd_evidence_fixture(fixture)
    assert summary["evidence_validation"] == "passed" and summary["production_status"] == "production_validated"
    assert tuple(m.label for m in fixture.normal_modes) == LANE_MODE_LABELS == tuple(fixture.physical_sector.mode_labels)
    assert [m.frequency_cm1 for m in fixture.normal_modes] == pytest.approx(list(EXPECTED_LANE_FREQUENCIES_CM1), abs=0.5)
    assert tuple(int(b.n_ph_max) for b in fixture.layout.boson_modes) == tuple(work_cutoffs)
    assert fixture.layout.n_total_qubits == 12 + sum(int(np.ceil(np.log2(c + 1))) for c in work_cutoffs)
    # The fixture's exact sector energy is the builder's exact solution of the same model (cutoffs permuted
    # from lane order into the builder's ascending-frequency order).
    assert summary["builder_ground_energy_hartree"] is not None
    assert summary["builder_ground_energy_hartree"] == pytest.approx(summary["fixture_ground_energy_hartree"], abs=1e-7)
    lane_to_builder = {int(m["builder_mode_index"]): i for i, m in enumerate(fixture.report_summary["normal_mode_diagnostics"])}
    assert summary["builder_order_cutoffs"] == [list(work_cutoffs)[lane_to_builder[a]] for a in range(len(work_cutoffs))]
    assert summary["decided_model"] is True and summary["model_decided_on"] == "2026-11-02"
    assert fixture.exact_reference.ground_energy_hartree == pytest.approx(summary["fixture_ground_energy_hartree"], abs=1e-12)
    assert fixture.provenance["backend"]["model_definition"]["mode_basis"] == "model_consistent"
    assert fixture.report_summary["pyscf_backend_record_ready"] and not fixture.report_summary["psi4_backend_record_ready"]
    assert all(d.passed and d.finite_difference_drift < 1.0 for d in fixture.first_derivatives)
    assert summary["derivative_validation"] == {"tier": "production", "rel_tol": 1e-5}
    # The consumer path: the cached loader the problem registry uses, and the attested identity.
    cached = load_cached_production_vibronic_h2o_linear_fd_fixture(fixture_path)
    assert tuple(cached.model.mode_labels) == LANE_MODE_LABELS and int(cached.model.n_total_qubits) == fixture.layout.n_total_qubits
    _check_identity(summary, fixture, fixture_path)
    return fixture


def _check_identity(summary, fixture, fixture_path):
    """The summary's identities are the ones the lane attests: the sha256 of the fixture BYTES (the
    registry's _sha256_and_byte_count, not the canonical-JSON hash the generator CLI prints) and the
    canonical-JSON hash of the backend record that the fixture manifest carries."""
    assert (summary["fixture_sha256"], summary["fixture_byte_count"]) == _sha256_and_byte_count(fixture_path)
    record_path = fixture_path.with_name(fixture_path.name.replace("_fixture.json", "_record.json"))
    record_sha = sha256_jsonable(json.loads(record_path.read_text(encoding="utf-8")))
    assert summary["record_payload_sha256"] == record_sha == fixture.manifest.provenance_hashes["backend_record_sha256"]
    assert summary["record_file_sha256"] == _sha256_and_byte_count(record_path)[0]
    assert summary["fixture_payload_sha256"] != summary["fixture_sha256"]


def test_fixture_from_builder_passes_the_lane_gate_and_matches_the_builder(tmp_path):
    pytest.importorskip("pyscf")
    # Non-uniform work cutoffs: the symmetric stretch (lane mode 1, builder mode 2) gets the larger one,
    # and the reference cutoffs default to one more per mode.
    summary = write_paper_iv_h2o_fixture_from_builder(tmp_path, tag="small", mode_cutoffs=(1, 2, 1))
    fixture = _check_fixture(summary, tmp_path, work_cutoffs=(1, 2, 1))
    assert fixture.layout.n_total_qubits == 16 and summary["reference_cutoffs"] == [2, 3, 2]
    assert summary["builder_order_cutoffs"] == [1, 1, 2]
    assert json.loads((tmp_path / "small_summary.json").read_text())["fixture_sha256"] == summary["fixture_sha256"]
    assert (tmp_path / "small_record.json").is_file()
    with pytest.raises(FileExistsError):
        write_paper_iv_h2o_fixture_from_builder(tmp_path, tag="small", mode_cutoffs=(1, 1, 1))
    with pytest.raises(ValueError, match="exceed"):
        write_paper_iv_h2o_fixture_from_builder(tmp_path, tag="other", mode_cutoffs=(1, 1, 1), reference_cutoffs=(1, 2, 2))


@pytest.mark.skipif(os.environ.get("PAPER_IV_FIXTURE_FROM_BUILDER_FULL") != "1",
                    reason="production cutoffs take minutes; set PAPER_IV_FIXTURE_FROM_BUILDER_FULL=1")
def test_production_fixture_from_builder(tmp_path):
    pytest.importorskip("pyscf")
    summary = write_paper_iv_h2o_fixture_from_builder(tmp_path, tag="production", mode_cutoffs=(3, 3, 3))
    fixture = _check_fixture(summary, tmp_path, work_cutoffs=(3, 3, 3))
    assert fixture.layout.n_total_qubits == 18 and fixture.exact_reference.sector_dimension == 14400
    assert summary["fixture_ground_energy_hartree"] == pytest.approx(PRODUCTION_GROUND_ENERGY_333, abs=1e-6)
    assert summary["cutoff_diagnostics_passed"]


CHECKED_IN_TAG = "h2o_builder_model_consistent_work333_ref444_20261103"
CHECKED_IN_FIXTURE_SHA256 = "de32cd4d1e29b8dcc07e94ad132ee43893fe03fd36cf4538eb6dadf6d5e194df"
CHECKED_IN_FIXTURE_BYTES = 15673312


def test_checked_in_production_fixture_matches_its_summary():
    directory = REPO_ROOT / "test_support" / "paper_iv_h2o"
    summary = json.loads((directory / f"{CHECKED_IN_TAG}_summary.json").read_text())
    fixture_path = directory / summary["fixture_file"]
    assert fixture_path.name == f"{CHECKED_IN_TAG}_fixture.json" and fixture_path.is_file()
    fixture = load_production_vibronic_h2o_fixture(fixture_path)
    validate_paper_iv_h2o_linear_fd_evidence_fixture(fixture)
    # The identity pinned here is what every Paper IV checkpoint on this fixture attests.
    _check_identity(summary, fixture, fixture_path)
    assert summary["fixture_sha256"] == CHECKED_IN_FIXTURE_SHA256 and summary["fixture_byte_count"] == CHECKED_IN_FIXTURE_BYTES
    assert tuple(m.label for m in fixture.normal_modes) == LANE_MODE_LABELS
    assert fixture.exact_reference.ground_energy_hartree == pytest.approx(summary["fixture_ground_energy_hartree"], abs=1e-12)
    assert summary["fixture_ground_energy_hartree"] == pytest.approx(PRODUCTION_GROUND_ENERGY_333, abs=1e-6)
    assert summary["model_definition"]["mode_basis"] == "model_consistent" and summary["model_decided_on"] == "2026-11-02"
    assert [m.frequency_hartree * HARTREE_TO_CM_INV for m in fixture.normal_modes] == pytest.approx(list(EXPECTED_LANE_FREQUENCIES_CM1), abs=0.5)
