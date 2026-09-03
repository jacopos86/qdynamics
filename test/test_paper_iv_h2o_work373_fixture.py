"""Identity pin for the converged Paper-IV H2O fixture (work (3,7,3) / ref (4,8,4)).

The tracked copies under ``test_support/paper_iv_h2o/`` were taken verbatim from
``chtc_rescue/home/chtc/paper_iv_h2o_reencode_20260826/results/`` on 2026-09-03.
These sha256 values are the fixture's identity: a mismatch means the tracked
file was altered, not that the constant is wrong.

NOTE ON LINEAGE: this fixture is NOT the July fixture at
``chtc/paper_iv_h2o_sr_paper_i_noprune_nobeam_derivative_resolved_d50_20260724/input/h2o_fixture.json``
(sha256 570690bd..., 3.3 MB, unconverged cutoffs). The two are different
lineages by design; do not "fix" either sha to match the other.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.quantum.chemistry.vibronic_h2o_linear_fd import (
    load_cached_production_vibronic_h2o_linear_fd_fixture,
)

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE_DIR = _ROOT / "test_support" / "paper_iv_h2o"
FIXTURE_PATH = _FIXTURE_DIR / "h2o_linear_fd_work373_ref484_fixture_20260826.json"
SUMMARY_PATH = _FIXTURE_DIR / "reencode_summary_work373_ref484.json"

FIXTURE_SHA256 = "190387449b556904c73fd424635e494eb92af2c13e8b0e0d71234a4bc391256e"
FIXTURE_SIZE_BYTES = 28_025_185
SUMMARY_SHA256 = "c0d41c5ad896188f64de3a54c6dd10f98b92061b8b47fb43489e632814bced33"

# July (unconverged, d50 campaign) fixture -- a different lineage. Kept here only
# so the divergence is asserted and documented.
JULY_FIXTURE_SHA256 = "570690bd126787305b340bd2f7493499c0f3101e3e2820c2d355c55c16afa594"

WORK_CUTOFFS = (3, 7, 3)
REF_CUTOFFS = (4, 8, 4)
N_TOTAL_QUBITS = 19
SECTOR_DIMENSION = 28_800
E_WORK_HARTREE = -75.01446351337651
E_REF_HARTREE = -75.01513494325133
DELTA_E_HARTREE = 6.714298748136116e-4
TOTAL_BOUNDARY_WEIGHT = 6.206639167990617e-3
REENCODE_GENERATOR_VERSION = "h2o_linear_fd_generator_v2_reencode_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_fixture_sha256_pinned() -> None:
    assert FIXTURE_PATH.is_file(), FIXTURE_PATH
    assert FIXTURE_PATH.stat().st_size == FIXTURE_SIZE_BYTES
    assert _sha256(FIXTURE_PATH) == FIXTURE_SHA256


def test_summary_sha256_pinned() -> None:
    assert SUMMARY_PATH.is_file(), SUMMARY_PATH
    assert _sha256(SUMMARY_PATH) == SUMMARY_SHA256


def test_fixture_is_not_the_july_lineage() -> None:
    # Different lineage: converged (3,7,3)/(4,8,4) reencode vs the July d50 input.
    assert FIXTURE_SHA256 != JULY_FIXTURE_SHA256


@pytest.fixture(scope="module")
def cached():
    return load_cached_production_vibronic_h2o_linear_fd_fixture(
        FIXTURE_PATH,
        require_exact_state=True,
        require_reference_cutoff=True,
        require_cutoff_converged=True,
    )


def test_loaded_model_register_and_cutoffs(cached) -> None:
    # ProductionVibronicH2OLinearFDRuntimeModel.n_total_qubits / .mode_cutoffs
    # (src/quantum/chemistry/vibronic_h2o_linear_fd.py:372,374).
    model = cached.model
    assert model.n_total_qubits == N_TOTAL_QUBITS
    assert tuple(model.mode_cutoffs) == WORK_CUTOFFS
    assert model.n_fermion_qubits + model.n_boson_qubits == N_TOTAL_QUBITS
    # RegisterLayout.n_total_qubits property (vibronic_h2o_linear_fd.py:229).
    assert cached.fixture.layout.n_total_qubits == N_TOTAL_QUBITS
    assert tuple(cached.fixture.physical_sector.n_ph_max_by_mode) == WORK_CUTOFFS
    assert tuple(cached.metadata["mode_cutoffs"]) == WORK_CUTOFFS


def test_loaded_fixture_cutoff_diagnostics(cached) -> None:
    # CutoffDiagnosticsRecord (vibronic_h2o_linear_fd.py:296-308).
    diag = cached.fixture.cutoff_diagnostics
    assert diag is not None
    assert tuple(diag.work_cutoffs) == WORK_CUTOFFS
    assert tuple(diag.reference_cutoffs) == REF_CUTOFFS
    assert diag.passed is True
    assert diag.work_ground_energy_hartree == E_WORK_HARTREE
    assert diag.reference_ground_energy_hartree == E_REF_HARTREE
    assert diag.delta_energy_hartree == DELTA_E_HARTREE
    assert diag.work_boundary_weight is not None
    assert diag.work_boundary_weight.total_boundary_weight == TOTAL_BOUNDARY_WEIGHT
    # ExactReferenceRecord (vibronic_h2o_linear_fd.py:282-292).
    exact = cached.fixture.exact_reference
    assert exact.available is True
    assert exact.sector_dimension == SECTOR_DIMENSION
    assert exact.ground_energy_hartree == E_WORK_HARTREE
    assert cached.metadata["vibrational_cutoff_converged"] is True
    assert cached.metadata["cutoff_delta_energy_hartree"] == DELTA_E_HARTREE
    assert cached.metadata["cutoff_total_boundary_weight"] == TOTAL_BOUNDARY_WEIGHT


def test_fixture_embedded_provenance_is_reencode_v2(cached) -> None:
    # ProductionVibronicH2OFixture.provenance / .manifest
    # (vibronic_h2o_linear_fd.py:337-355, 322-334).
    provenance = cached.fixture.provenance
    assert provenance["reencoding_generator"].endswith(REENCODE_GENERATOR_VERSION)
    assert cached.fixture.manifest.generator_version == REENCODE_GENERATOR_VERSION
    assert cached.fixture.report_summary["reencoded_from_retained_tensor_evidence"] is True
    assert cached.fixture.report_summary["vibrational_cutoff_converged"] is True
    assert cached.metadata["production_status"] == "production_validated"
    assert cached.metadata["family_key"] == "molecular_vibronic_h2o_linear_fd"


def test_summary_json_matches_expected_physics() -> None:
    summary = json.loads(SUMMARY_PATH.read_text())
    assert summary["tag"] == "work373_ref484"
    assert summary["fixture_file"] == FIXTURE_PATH.name
    assert summary["status"] == "completed"
    assert summary["passed"] is True
    assert tuple(summary["work"]) == WORK_CUTOFFS
    assert tuple(summary["ref"]) == REF_CUTOFFS
    assert summary["n_total_qubits"] == N_TOTAL_QUBITS
    assert summary["sector_dimension"] == SECTOR_DIMENSION
    assert summary["work_ground_energy_hartree"] == E_WORK_HARTREE
    assert summary["reference_ground_energy_hartree"] == E_REF_HARTREE
    assert summary["delta_energy_hartree"] == DELTA_E_HARTREE
    assert summary["total_boundary_weight"] == TOTAL_BOUNDARY_WEIGHT
    assert summary["evidence_validation"] == "passed_cutoff_converged_required"
