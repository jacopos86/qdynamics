"""Focused tests for the Paper III SIAM static-spectroscopy adapter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipelines.qse_spectra.siam_static_adapter import (
    GROUND_SOURCE_DIAGNOSTIC,
    GROUND_SOURCE_IMPORTED,
    SIAMStaticAdapterError,
    build_siam_hamiltonian_for_key,
    build_siam_pool_elements,
    declared_siam_cases,
    exact_sector_ground_fixture,
    graph_span_compiled_cost_receipt,
    load_imported_ground_state,
    load_siam_problem,
    ordered_pool_digest,
    siam_physics_key,
    siam_sector_spectrum,
    validate_spectroscopy_ground_state,
)
from src.quantum.siam_model import build_initial_state, sector_indices


def _write_state_manifest(path: Path, psi: np.ndarray, nq: int) -> None:
    amplitudes = {
        format(index, f"0{nq}b"): {
            "re": float(value.real),
            "im": float(value.imag),
        }
        for index, value in enumerate(np.asarray(psi, dtype=complex))
        if abs(value) > 1.0e-14
    }
    payload = {"nq_total": int(nq), "amplitudes_qn_to_q0": amplitudes}
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_declared_cases_have_the_specified_sectors() -> None:
    assert declared_siam_cases() == ("siam_nb2_fixture", "siam_nb4_pilot")

    fixture = siam_physics_key("siam_nb2_fixture")
    assert fixture.n_qubits == 6
    assert (fixture.n_up, fixture.n_down) == (2, 1)
    assert len(sector_indices(2)) == 9
    assert fixture.u == 1.5
    assert fixture.v == pytest.approx(1.0 / np.sqrt(2.0))
    assert fixture.hopping == 1.0

    pilot = siam_physics_key("siam_nb4_pilot")
    assert pilot.n_qubits == 10
    assert (pilot.n_up, pilot.n_down) == (3, 2)
    assert len(sector_indices(4)) == 100
    assert pilot.u == 1.5


def test_physics_digest_is_stable_and_case_sensitive() -> None:
    a = siam_physics_key("siam_nb2_fixture")
    b = siam_physics_key("siam_nb2_fixture")
    c = siam_physics_key("siam_nb4_pilot")
    assert a.digest() == b.digest()
    assert a.digest() != c.digest()
    assert siam_physics_key("siam_nb2_fixture").canonical_payload()["cutoff_kind"] == "n_bath"


def test_unknown_case_fails_closed() -> None:
    with pytest.raises(SIAMStaticAdapterError, match="unknown SIAM case"):
        siam_physics_key("siam_nb6_unplanned")


def test_pool_elements_are_deterministic_and_labeled() -> None:
    key = siam_physics_key("siam_nb2_fixture")
    elements = build_siam_pool_elements(key)
    names = [str(e.name) for e in elements]
    assert len(set(names)) == len(names)
    assert all(name.startswith("siam::") for name in names)
    assert ordered_pool_digest(elements) == ordered_pool_digest(
        build_siam_pool_elements(key)
    )
    families = {str((e.metadata or {}).get("family")) for e in elements}
    assert families == {"K", "J", "Qd", "A", "X", "P"}


def test_spectroscopy_hamiltonian_is_the_interacting_one() -> None:
    key = siam_physics_key("siam_nb2_fixture")
    ground, energies = siam_sector_spectrum(key, count=3)
    # Ground vector is a true eigenvector of the interacting Hamiltonian.
    from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix

    dense = np.asarray(
        hamiltonian_matrix(build_siam_hamiltonian_for_key(key)), dtype=complex
    )
    residual = dense @ ground - energies[0] * ground
    assert float(np.linalg.norm(residual)) < 1.0e-10
    assert energies[0] < energies[1] <= energies[2]


def test_quench_seed_is_rejected_numerically() -> None:
    key = siam_physics_key("siam_nb2_fixture")
    quench = build_initial_state(2)
    with pytest.raises(SIAMStaticAdapterError, match="decoupled bath/quench seed"):
        validate_spectroscopy_ground_state(key, quench)


def test_quench_seed_is_rejected_by_provenance_marker() -> None:
    key = siam_physics_key("siam_nb2_fixture")
    ground, _ = siam_sector_spectrum(key, count=1)
    with pytest.raises(SIAMStaticAdapterError, match="quench"):
        validate_spectroscopy_ground_state(
            key, ground, provenance={"state_source": "paper_ii_quench_seed"}
        )


def test_state_shape_and_sector_mismatches_fail() -> None:
    key = siam_physics_key("siam_nb2_fixture")
    with pytest.raises(SIAMStaticAdapterError, match="register size"):
        validate_spectroscopy_ground_state(key, np.ones(16, dtype=complex) / 4.0)

    outside = np.zeros(64, dtype=complex)
    outside[0] = 1.0  # vacuum: zero electrons, not the (2,1) sector
    with pytest.raises(SIAMStaticAdapterError, match="sector"):
        validate_spectroscopy_ground_state(key, outside)


def test_exact_fixture_is_labeled_diagnostic_only() -> None:
    key = siam_physics_key("siam_nb2_fixture")
    ground, provenance = exact_sector_ground_fixture(key)
    assert provenance["ground_source"] == GROUND_SOURCE_DIAGNOSTIC
    assert provenance["diagnostic_only"] is True
    assert provenance["paper_facing"] is False
    checks = validate_spectroscopy_ground_state(key, ground)
    assert checks["sector_leakage"] < 1.0e-12


def test_production_requires_an_artifact_or_explicit_fixture_opt_in() -> None:
    with pytest.raises(SIAMStaticAdapterError, match="ground-state artifact"):
        load_siam_problem("siam_nb2_fixture")


def test_imported_artifact_round_trip(tmp_path: Path) -> None:
    key = siam_physics_key("siam_nb2_fixture")
    ground, _ = siam_sector_spectrum(key, count=1)
    manifest = tmp_path / "siam_nb2_ground.json"
    _write_state_manifest(manifest, ground, key.n_qubits)

    psi, provenance = load_imported_ground_state(key, manifest)
    assert provenance["ground_source"] == GROUND_SOURCE_IMPORTED
    assert provenance["diagnostic_only"] is False
    assert provenance["manifest_sha256"]
    assert provenance["state_payload_sha256"]
    assert abs(abs(complex(np.vdot(psi, ground))) - 1.0) < 1.0e-10


def test_imported_quench_manifest_is_rejected(tmp_path: Path) -> None:
    key = siam_physics_key("siam_nb2_fixture")
    manifest = tmp_path / "siam_nb2_quench.json"
    _write_state_manifest(manifest, build_initial_state(2), key.n_qubits)
    with pytest.raises(SIAMStaticAdapterError, match="quench"):
        load_imported_ground_state(key, manifest)


def test_problem_assembly_from_diagnostic_fixture() -> None:
    problem = load_siam_problem(
        "siam_nb2_fixture", allow_exact_diagnostic_fixture=True, target_roots=6
    )
    assert problem.num_qubits == 6
    assert len(problem.basis) == len(problem.costs) == len(problem.resources)
    assert len(problem.spectrum) == 7
    assert len(problem.references) == 6
    assert problem.ground_energy == pytest.approx(problem.spectrum[0])
    receipt = problem.arm_receipt()
    assert receipt["ground_source"] == GROUND_SOURCE_DIAGNOSTIC
    assert receipt["diagnostic_only"] is True
    assert receipt["pool_digest"] == problem.pool_digest
    triple = problem.resource_triple([0, 1])
    assert set(triple) == {"n2q", "d2q", "dc"}
    assert triple["dc"] >= triple["d2q"]


def test_cost_receipt_with_injected_compiled_counts() -> None:
    key = siam_physics_key("siam_nb2_fixture")
    pool_size = len(build_siam_pool_elements(key))
    injected = [float(3 * i + 1) for i in range(pool_size)]
    receipt = graph_span_compiled_cost_receipt(
        "siam_nb2_fixture", compiled_two_qubit_counts=injected
    )
    assert receipt["sample_policy"] == "full_pool"
    assert receipt["sample_size"] == pool_size
    assert receipt["compiled_source"] == "injected_for_test"
    assert len(receipt["records"]) == pool_size
    assert receipt["backend"]["seed_transpiler"] == 7
    assert receipt["controller_boundary"]["feeds_controller_decisions"] is False
    assert -1.0 <= receipt["spearman_rank_correlation"] <= 1.0
    assert 0.0 <= receipt["top_decile_overlap"] <= 1.0


def test_cost_receipt_stratifies_the_pilot_pool() -> None:
    key = siam_physics_key("siam_nb4_pilot")
    elements = build_siam_pool_elements(key)
    from pipelines.qse_spectra.siam_static_adapter import _family_stratified_sample

    sample = _family_stratified_sample(elements, per_family=4)
    assert sample == sorted(set(sample))
    assert len(sample) < len(elements)
    families = {str((elements[i].metadata or {}).get("family")) for i in sample}
    assert families == {"K", "J", "Qd", "A", "X", "P"}
