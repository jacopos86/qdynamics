from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.quantum.chemistry.generate_h2o_linear_fd_fixture as generator_mod
from pipelines.static_adapt.builders import problem_registry
from pipelines.static_adapt.builders.pool_resolution import resolve_pool_plan
from src.quantum.chemistry.generate_h2o_linear_fd_fixture import (
    BACKEND_RECORD_SCHEMA,
    _freeze_core_active_tensors,
    _derivative_validation_config,
    _displace_coordinates_along_mass_weighted_mode,
    _h2o_mode_character,
    _mass_weighted_vibrational_modes_from_hessian,
    _mass_weighted_displacement_norm,
    _validate_derivative_pair,
    build_h2o_linear_fd_backend_record_with_psi4,
    build_h2o_linear_fd_fixture_from_record,
    main,
    parse_args,
    reencode_h2o_linear_fd_fixture,
)
from src.quantum.chemistry.psi4_adapter import RestrictedClosedShellMolecularProblem
from src.quantum.chemistry.vibronic_h2o_linear_fd import (
    H2O_LINEAR_FD_DERIVATIVE_RESOLVED_POOL_KEY,
    H2O_LINEAR_FD_FAMILY_KEY,
    _canonicalized_symmetric_spectral_factors,
    _chemist_eri_spectral_factors,
    build_h2o_linear_fd_derivative_resolved_pool_v2,
    load_cached_production_vibronic_h2o_linear_fd_fixture,
    production_vibronic_h2o_fixture_to_jsonable,
    production_vibronic_h2o_fixture_from_jsonable,
    validate_production_vibronic_h2o_fixture,
    validate_paper_iv_h2o_linear_fd_evidence_fixture,
)
from test_support.h2o_linear_fd_fixture_factory import (
    synthetic_three_mode_h2o_linear_fd_backend_record,
    write_synthetic_three_mode_h2o_linear_fd_backend_record_json,
)


def _request(
    *,
    fixture_json: str | Path,
    n_ph_max: int,
) -> problem_registry.ProblemRequest:
    return problem_registry.ProblemRequest(
        problem_key=H2O_LINEAR_FD_FAMILY_KEY,
        num_sites=2,
        t=1.0,
        u=0.0,
        dv=0.0,
        omega0=0.017,
        g_ep=1.0,
        n_ph_max=int(n_ph_max),
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        include_zero_point=True,
        molecular_vibronic_h2o_linear_fd_fixture_json=str(fixture_json),
    )


def test_generator_parse_args_record_mode_is_import_safe() -> None:
    args = parse_args(
        [
            "--backend",
            "record",
            "--input-record-json",
            "backend_record.json",
            "--output-fixture-json",
            "fixture.json",
            "--mode-cutoffs",
            "2,2,2",
        ]
    )

    assert args.backend == "record"
    assert str(args.input_record_json) == "backend_record.json"
    assert str(args.output_fixture_json) == "fixture.json"
    assert args.mode_cutoffs == "2,2,2"


def test_generator_parse_args_psi4_record_only_surface_is_import_safe() -> None:
    args = parse_args(
        [
            "--backend",
            "psi4",
            "--record-only",
            "--output-record-json",
            "backend_record.json",
            "--psi4-active-space",
            "frontier_2e_2o",
            "--psi4-no-optimize",
        ]
    )

    assert args.backend == "psi4"
    assert args.record_only is True
    assert str(args.output_record_json) == "backend_record.json"
    assert args.psi4_active_space == "frontier_2e_2o"
    assert args.psi4_no_optimize is True


def test_generator_psi4_record_only_writes_backend_record_without_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = synthetic_three_mode_h2o_linear_fd_backend_record()

    def _fake_backend(**_kwargs: object) -> dict[str, object]:
        return record

    def _unexpected_fixture_build(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("record-only mode must not build a fixture")

    monkeypatch.setattr(generator_mod, "build_h2o_linear_fd_backend_record_with_psi4", _fake_backend)
    monkeypatch.setattr(generator_mod, "build_h2o_linear_fd_fixture_from_record", _unexpected_fixture_build)
    output_record = tmp_path / "backend_record.json"

    generator_mod.main(
        [
            "--backend",
            "psi4",
            "--record-only",
            "--output-record-json",
            str(output_record),
            "--psi4-active-space",
            "frontier_2e_2o",
            "--force",
        ]
    )

    assert output_record.exists()
    assert output_record.read_text(encoding="utf-8")


def test_optional_psi4_backend_record_smoke() -> None:
    if os.environ.get("RUN_OPTIONAL_CHEMISTRY") != "1":
        pytest.skip("set RUN_OPTIONAL_CHEMISTRY=1 to run the optional Psi4 backend smoke")
    pytest.importorskip("psi4")

    record = build_h2o_linear_fd_backend_record_with_psi4(
        optimize_geometry=False,
        active_space="frontier_2e_2o",
        q_step_au=0.05,
        q_step_alt_au=0.025,
    )

    assert record["schema"] == BACKEND_RECORD_SCHEMA
    assert record["backend"]["name"] == "psi4"
    assert record["active_space"]["n_spatial_orbitals"] == 2
    assert len(record["normal_modes"]) == 3
    assert len(record["aligned_tensors"]) == 12


def test_derivative_validation_allows_near_zero_suppressed_derivative() -> None:
    config = _derivative_validation_config(
        tier="production",
        max_derivative_drift=1.0e-6,
    )

    diagnostics = _validate_derivative_pair(
        scalar_primary=1.9895196601282805e-12,
        scalar_alt=1.7053025658242404e-12,
        one_body_primary=np.array([[4.445498765875962e-14, 0.0], [0.0, 0.0]]),
        one_body_alt=np.array([[3.9033842801124816e-13, 0.0], [0.0, 0.0]]),
        two_body_primary=np.full((2, 2, 2, 2), 3.832557179949574e-14 / 4.0),
        two_body_alt=np.full((2, 2, 2, 2), 5.3018944513004916e-14 / 4.0),
        config=config,
    )

    assert diagnostics["passed"] is True
    assert diagnostics["classification"] == "numerically_suppressed"
    assert diagnostics["legacy_relative_drift"] > 0.8
    assert diagnostics["components"]["one_body"]["classification"] == "suppressed"


def test_derivative_validation_rejects_same_norm_wrong_direction_tensor() -> None:
    config = _derivative_validation_config(
        tier="production",
        max_derivative_drift=1.0e-6,
    )

    diagnostics = _validate_derivative_pair(
        scalar_primary=0.0,
        scalar_alt=0.0,
        one_body_primary=np.array([[1.0e-3, 0.0], [0.0, 0.0]]),
        one_body_alt=np.array([[0.0, 1.0e-3], [0.0, 0.0]]),
        two_body_primary=np.zeros((2, 2, 2, 2), dtype=float),
        two_body_alt=np.zeros((2, 2, 2, 2), dtype=float),
        config=config,
    )

    assert diagnostics["passed"] is False
    assert diagnostics["classification"] == "failed"
    assert diagnostics["legacy_relative_drift"] == pytest.approx(0.0)
    assert diagnostics["components"]["one_body"]["passed"] is False
    assert diagnostics["components"]["one_body"]["direction_cosine"] == pytest.approx(0.0)


def test_mass_weighted_displacement_norm_matches_q_step() -> None:
    coordinates = np.zeros((3, 3), dtype=float)
    masses = np.array([16.0, 1.0, 1.0], dtype=float) * 1822.888486209
    mode = np.zeros((3, 3), dtype=float)
    mode[1, 1] = 1.0 / np.sqrt(2.0)
    mode[2, 1] = -1.0 / np.sqrt(2.0)
    q_step = 0.125

    displaced = _displace_coordinates_along_mass_weighted_mode(
        coordinates,
        mode_vector=mode,
        masses_me=masses,
        q_displacement_au=q_step,
    )

    assert _mass_weighted_displacement_norm(displaced - coordinates, masses_me=masses) == pytest.approx(q_step)


def test_h2o_mode_character_distinguishes_symmetric_and_antisymmetric_stretch() -> None:
    symbols = ("O", "H", "H")
    bond = 1.8
    half_angle = np.deg2rad(104.52 / 2.0)
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, bond * np.sin(half_angle), bond * np.cos(half_angle)],
            [0.0, -bond * np.sin(half_angle), bond * np.cos(half_angle)],
        ],
        dtype=float,
    )
    masses = np.array([16.0, 1.0, 1.0], dtype=float) * 1822.888486209
    r1_unit = coordinates[1] / np.linalg.norm(coordinates[1])
    r2_unit = coordinates[2] / np.linalg.norm(coordinates[2])
    sym_mode = np.zeros_like(coordinates)
    sym_mode[1] = np.sqrt(masses[1]) * r1_unit
    sym_mode[2] = np.sqrt(masses[2]) * r2_unit
    sym_mode /= np.linalg.norm(sym_mode)
    asym_mode = np.zeros_like(coordinates)
    asym_mode[1] = np.sqrt(masses[1]) * r1_unit
    asym_mode[2] = -np.sqrt(masses[2]) * r2_unit
    asym_mode /= np.linalg.norm(asym_mode)

    sym_character = _h2o_mode_character(
        symbols=symbols,
        coordinates_bohr=coordinates,
        mode_vector=sym_mode,
        masses_me=masses,
    )
    asym_character = _h2o_mode_character(
        symbols=symbols,
        coordinates_bohr=coordinates,
        mode_vector=asym_mode,
        masses_me=masses,
    )

    assert sym_character["dominant_label"] == "symmetric_stretch"
    assert asym_character["dominant_label"] == "antisymmetric_stretch"
    assert sym_character["symmetric_stretch_score"] > sym_character["antisymmetric_stretch_score"]
    assert asym_character["antisymmetric_stretch_score"] > asym_character["symmetric_stretch_score"]


def test_mass_weighted_vibrational_modes_from_hessian_selects_three_positive_modes() -> None:
    masses = np.array([16.0, 1.0, 1.0], dtype=float) * 1822.888486209
    lambdas = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01**2, 0.02**2, 0.03**2])
    masses_by_coord = np.repeat(masses, 3)
    hessian = np.diag(lambdas * masses_by_coord)

    modes = _mass_weighted_vibrational_modes_from_hessian(hessian, masses_me=masses)

    assert [row["label"] for row in modes] == ["bend", "symmetric_stretch", "antisymmetric_stretch"]
    assert [row["frequency_hartree"] for row in modes] == pytest.approx([0.01, 0.02, 0.03])
    for row in modes:
        vector = np.asarray(row["mass_weighted_eigenvector"], dtype=float)
        assert vector.shape == (3, 3)
        assert float(np.linalg.norm(vector)) == pytest.approx(1.0)


def test_freeze_core_active_tensors_adds_closed_shell_core_scalar_and_one_body_shift() -> None:
    h = np.diag([1.0, 2.0, 3.0])
    g = np.zeros((3, 3, 3, 3), dtype=float)
    g[0, 0, 0, 0] = 0.4
    g[1, 1, 0, 0] = 0.2
    g[1, 0, 0, 1] = 0.05
    g[2, 2, 0, 0] = 0.3
    g[2, 0, 0, 2] = 0.07
    problem = RestrictedClosedShellMolecularProblem(
        geometry_spec="synthetic",
        basis="synthetic",
        charge=0,
        multiplicity=1,
        reference="rhf",
        n_spatial_orbitals=3,
        n_alpha=2,
        n_beta=2,
        hf_energy=0.0,
        nuclear_repulsion_energy=0.5,
        one_body_integrals_mo=h,
        two_body_integrals_mo=g,
    )

    scalar, h_active, g_active = _freeze_core_active_tensors(
        problem,
        active_indices=(1, 2),
        frozen_core_indices=(0,),
    )

    assert scalar == pytest.approx(0.5 + 2.0 * 1.0 + 2.0 * 0.4 - 0.4)
    assert h_active[0, 0] == pytest.approx(2.0 + 2.0 * 0.2 - 0.05)
    assert h_active[1, 1] == pytest.approx(3.0 + 2.0 * 0.3 - 0.07)
    assert g_active.shape == (2, 2, 2, 2)


def test_generator_builds_strict_fixture_from_synthetic_backend_record() -> None:
    record = synthetic_three_mode_h2o_linear_fd_backend_record()

    fixture = build_h2o_linear_fd_fixture_from_record(
        record,
        mode_cutoffs=(2, 2, 2),
        dense_full_dim_cap=4096,
        embed_exact_state=True,
    )

    validate_paper_iv_h2o_linear_fd_evidence_fixture(
        fixture,
        require_exact_state=True,
        require_reference_cutoff=True,
    )
    assert record["schema"] == BACKEND_RECORD_SCHEMA
    assert fixture.manifest.family_key == H2O_LINEAR_FD_FAMILY_KEY
    assert fixture.layout.n_total_qubits == 10
    assert fixture.cutoff_diagnostics is not None
    assert fixture.cutoff_diagnostics.reference_cutoffs == (3, 3, 3)
    assert fixture.exact_reference.ground_state is not None
    assert fixture.exact_reference.ground_state.available is True
    assert np.isfinite(float(fixture.exact_reference.ground_energy_hartree))
    assert len(fixture.first_derivatives) == 3
    assert all(row.finite_difference_diagnostics["passed"] is True for row in fixture.first_derivatives)
    assert all(float(row.finite_difference_drift or 0.0) <= 1.0e-6 for row in fixture.first_derivatives)
    assert all(
        float(row.finite_difference_diagnostics["legacy_relative_drift"]) <= 1.0e-12
        for row in fixture.first_derivatives
    )

    labels = [str(row["label"]) for row in fixture.pool]
    assert any(label.startswith("el::") for label in labels)
    assert "boson::bend::p" in labels
    assert "coupled::symmetric_stretch::dH_dQ_times_p" in labels
    coupled_rows = [
        row
        for row in fixture.pool
        if row.get("generator_family") == "linear_vibronic_derivative_momentum"
    ]
    assert len(coupled_rows) == 3
    assert all(row.get("execution_mode") == "grouped_exact" for row in coupled_rows)

    payload = production_vibronic_h2o_fixture_to_jsonable(fixture)
    assert payload["manifest"]["provenance_hashes"]["backend_record_sha256"]
    unsafe_payload = deepcopy(payload)
    unsafe_coupled = next(
        row
        for row in unsafe_payload["pool"]
        if row.get("generator_family") == "linear_vibronic_derivative_momentum"
    )
    unsafe_coupled["execution_mode"] = "termwise_product"
    unsafe_fixture = production_vibronic_h2o_fixture_from_jsonable(unsafe_payload)
    with pytest.raises(ValueError, match="require grouped_exact execution"):
        validate_paper_iv_h2o_linear_fd_evidence_fixture(unsafe_fixture)


def test_derivative_resolved_factorizations_reconstruct_chemistry_tensors() -> None:
    rng = np.random.default_rng(41)
    one_body_raw = rng.normal(size=(4, 4))
    one_body = 0.5 * (one_body_raw + one_body_raw.T)
    one_body_factors = _canonicalized_symmetric_spectral_factors(
        one_body,
        absolute_tolerance=1.0e-12,
        relative_tolerance=1.0e-12,
    )
    one_body_reconstructed = sum(
        (weight * support for weight, support in one_body_factors),
        np.zeros_like(one_body),
    )
    assert np.allclose(one_body_reconstructed, one_body, atol=1.0e-11)

    pair_factors = []
    for weight in (0.7, -0.25, 0.11):
        raw = rng.normal(size=(4, 4))
        symmetric = 0.5 * (raw + raw.T)
        pair_factors.append(weight * np.einsum("pq,rs->pqrs", symmetric, symmetric))
    two_body = sum(pair_factors, np.zeros((4, 4, 4, 4), dtype=float))
    two_body_factors = _chemist_eri_spectral_factors(
        two_body,
        absolute_tolerance=1.0e-12,
        relative_tolerance=1.0e-12,
    )
    two_body_reconstructed = sum(
        (weight * support for weight, support in two_body_factors),
        np.zeros_like(two_body),
    )
    assert np.allclose(two_body_reconstructed, two_body, atol=1.0e-10)


def test_derivative_resolved_pool_is_additive_and_deterministic() -> None:
    fixture = build_h2o_linear_fd_fixture_from_record(
        synthetic_three_mode_h2o_linear_fd_backend_record(),
        mode_cutoffs=(1, 1, 1),
        reference_cutoffs=(1, 1, 1),
        dense_full_dim_cap=512,
        embed_exact_state=False,
    )

    first = build_h2o_linear_fd_derivative_resolved_pool_v2(fixture)
    second = build_h2o_linear_fd_derivative_resolved_pool_v2(fixture)
    base_labels = [str(row["label"]) for row in fixture.pool]
    first_labels = [term.label for term in first]

    assert first_labels[: len(base_labels)] == base_labels
    assert first_labels == [term.label for term in second]
    assert len(first_labels) == len(set(first_labels))
    assert sum(label.startswith("conditional::") for label in first_labels) == 9
    assert all(
        term.execution_mode == "grouped_exact"
        for term in first
        if term.label.startswith("conditional::")
    )
    assert H2O_LINEAR_FD_DERIVATIVE_RESOLVED_POOL_KEY == (
        "full_meta_derivative_resolved_v2"
    )


def test_generator_builds_candidate_fixture_without_exact_reference() -> None:
    fixture = build_h2o_linear_fd_fixture_from_record(
        synthetic_three_mode_h2o_linear_fd_backend_record(),
        mode_cutoffs=(2, 2, 2),
        dense_full_dim_cap=1,
        embed_exact_state=False,
        require_reference_cutoff=False,
        exact_reference_policy="candidate_without_exact",
    )

    assert fixture.manifest.production_status == "production_candidate"
    assert fixture.exact_reference.available is False
    assert fixture.exact_reference.method == "not_computed"
    assert fixture.exact_reference.ground_energy_hartree is None
    assert fixture.exact_reference.reason_unavailable
    assert fixture.cutoff_diagnostics is None
    assert fixture.evidence_hooks.exact_reference_ready is False
    assert fixture.report_summary["candidate_without_exact_reference"] is True
    validate_production_vibronic_h2o_fixture(
        fixture,
        require_production_validated=False,
    )
    with pytest.raises(ValueError, match="fixture is not production_validated"):
        validate_paper_iv_h2o_linear_fd_evidence_fixture(fixture)

    payload = production_vibronic_h2o_fixture_to_jsonable(fixture)
    loaded = production_vibronic_h2o_fixture_from_jsonable(json.loads(json.dumps(payload)))
    assert loaded.manifest.production_status == "production_candidate"
    assert loaded.exact_reference.available is False


def test_generator_sparse_sector_reference_matches_dense_for_synthetic_record() -> None:
    pytest.importorskip("scipy")
    record = synthetic_three_mode_h2o_linear_fd_backend_record()
    dense_fixture = build_h2o_linear_fd_fixture_from_record(
        record,
        mode_cutoffs=(2, 2, 2),
        dense_full_dim_cap=4096,
        embed_exact_state=True,
    )
    sparse_fixture = build_h2o_linear_fd_fixture_from_record(
        record,
        mode_cutoffs=(2, 2, 2),
        dense_full_dim_cap=1,
        embed_exact_state=True,
        exact_reference_policy="sparse_sector_eigsh",
    )

    assert sparse_fixture.manifest.production_status == "production_validated"
    assert sparse_fixture.exact_reference.method == "sparse_sector_eigsh"
    assert sparse_fixture.exact_reference.solver_tolerance == pytest.approx(1.0e-10)
    assert sparse_fixture.exact_reference.ground_state is not None
    assert sparse_fixture.exact_reference.ground_state.available is True
    assert sparse_fixture.exact_reference.ground_energy_hartree == pytest.approx(
        dense_fixture.exact_reference.ground_energy_hartree,
        abs=1.0e-9,
    )
    assert sparse_fixture.cutoff_diagnostics is not None
    assert dense_fixture.cutoff_diagnostics is not None
    assert sparse_fixture.cutoff_diagnostics.policy == "sparse_same_model_reference_cutoff_v1"
    assert sparse_fixture.cutoff_diagnostics.delta_energy_hartree == pytest.approx(
        dense_fixture.cutoff_diagnostics.delta_energy_hartree,
        abs=1.0e-9,
    )
    validate_paper_iv_h2o_linear_fd_evidence_fixture(
        sparse_fixture,
        require_exact_state=True,
        require_reference_cutoff=True,
    )


def test_reencode_fixture_rebuilds_operators_and_reference_from_retained_tensors() -> None:
    source = build_h2o_linear_fd_fixture_from_record(
        synthetic_three_mode_h2o_linear_fd_backend_record(),
        mode_cutoffs=(1, 1, 1),
        reference_cutoffs=(1, 1, 1),
        dense_full_dim_cap=512,
        embed_exact_state=True,
    )

    rebuilt = reencode_h2o_linear_fd_fixture(
        source,
        mode_cutoffs=(1, 1, 1),
        reference_cutoffs=(2, 2, 2),
        dense_full_dim_cap=4096,
        embed_exact_state=True,
    )

    assert rebuilt.manifest.generator_version.endswith("_reencode_v2")
    assert rebuilt.manifest.provenance_hashes["source_fixture_sha256"]
    assert rebuilt.cutoff_diagnostics is not None
    assert rebuilt.cutoff_diagnostics.reference_cutoffs == (2, 2, 2)
    assert rebuilt.report_summary["reencoded_from_retained_tensor_evidence"] is True
    assert rebuilt.exact_reference.ground_state is not None
    validate_paper_iv_h2o_linear_fd_evidence_fixture(
        rebuilt,
        require_exact_state=True,
        require_reference_cutoff=True,
    )


def test_generator_cli_record_mode_writes_loadable_fixture(tmp_path: Path) -> None:
    record_json = write_synthetic_three_mode_h2o_linear_fd_backend_record_json(
        tmp_path / "backend_record.json"
    )
    fixture_json = tmp_path / "h2o_linear_fd_fixture.json"

    main(
        [
            "--backend",
            "record",
            "--input-record-json",
            str(record_json),
            "--output-fixture-json",
            str(fixture_json),
            "--mode-cutoffs",
            "1,1,1",
            "--reference-mode-cutoffs",
            "1,1,1",
            "--dense-full-dim-cap",
            "512",
            "--embed-exact-state",
            "--force",
        ]
    )

    cached = load_cached_production_vibronic_h2o_linear_fd_fixture(
        fixture_json,
        require_exact_state=True,
        require_reference_cutoff=True,
    )
    assert cached.metadata["family_key"] == H2O_LINEAR_FD_FAMILY_KEY
    assert cached.metadata["mode_cutoffs"] == (1, 1, 1)
    assert cached.model.n_total_qubits == 7
    assert len(cached.model.pool) >= 3


def test_generator_cli_record_mode_writes_sparse_reference_fixture(tmp_path: Path) -> None:
    pytest.importorskip("scipy")
    record_json = write_synthetic_three_mode_h2o_linear_fd_backend_record_json(
        tmp_path / "backend_record.json"
    )
    fixture_json = tmp_path / "h2o_linear_fd_sparse_fixture.json"

    main(
        [
            "--backend",
            "record",
            "--input-record-json",
            str(record_json),
            "--output-fixture-json",
            str(fixture_json),
            "--mode-cutoffs",
            "1,1,1",
            "--reference-mode-cutoffs",
            "1,1,1",
            "--dense-full-dim-cap",
            "1",
            "--embed-exact-state",
            "--exact-reference-policy",
            "sparse_sector_eigsh",
            "--force",
        ]
    )

    cached = load_cached_production_vibronic_h2o_linear_fd_fixture(
        fixture_json,
        require_exact_state=True,
        require_reference_cutoff=True,
    )
    assert cached.fixture.exact_reference.method == "sparse_sector_eigsh"
    assert cached.fixture.cutoff_diagnostics is not None
    assert cached.fixture.cutoff_diagnostics.policy == "sparse_same_model_reference_cutoff_v1"


def test_generator_cli_record_mode_writes_candidate_fixture_without_exact(tmp_path: Path) -> None:
    record_json = write_synthetic_three_mode_h2o_linear_fd_backend_record_json(
        tmp_path / "backend_record.json"
    )
    fixture_json = tmp_path / "h2o_linear_fd_candidate_fixture.json"

    main(
        [
            "--backend",
            "record",
            "--input-record-json",
            str(record_json),
            "--output-fixture-json",
            str(fixture_json),
            "--mode-cutoffs",
            "1,1,1",
            "--no-reference-cutoff",
            "--dense-full-dim-cap",
            "1",
            "--exact-reference-policy",
            "candidate_without_exact",
            "--force",
        ]
    )

    payload = json.loads(fixture_json.read_text(encoding="utf-8"))
    assert payload["manifest"]["production_status"] == "production_candidate"
    assert payload["exact_reference"]["available"] is False
    assert payload["cutoff_diagnostics"] is None
    with pytest.raises(ValueError, match="production_validated"):
        load_cached_production_vibronic_h2o_linear_fd_fixture(fixture_json)


def test_generated_fixture_resolves_static_adapt_context(tmp_path: Path) -> None:
    record_json = write_synthetic_three_mode_h2o_linear_fd_backend_record_json(
        tmp_path / "backend_record.json"
    )
    fixture_json = tmp_path / "h2o_linear_fd_fixture.json"
    main(
        [
            "--backend",
            "record",
            "--input-record-json",
            str(record_json),
            "--output-fixture-json",
            str(fixture_json),
            "--mode-cutoffs",
            "1,1,1",
            "--reference-mode-cutoffs",
            "1,1,1",
            "--dense-full-dim-cap",
            "512",
            "--embed-exact-state",
            "--force",
        ]
    )

    resolved = problem_registry.resolve_problem_context(
        _request(fixture_json=fixture_json, n_ph_max=1)
    )

    assert resolved.family_key == H2O_LINEAR_FD_FAMILY_KEY
    assert resolved.layout.total_qubits == 7
    assert resolved.runtime_data["vibronic_h2o_linear_fd_mode_cutoffs"] == (1, 1, 1)
    assert resolved.runtime_data["vibronic_h2o_linear_fd_fixture_path"] == str(fixture_json)
    assert np.isfinite(float(resolved.exact_target.resolve_energy()))
    assert H2O_LINEAR_FD_DERIVATIVE_RESOLVED_POOL_KEY in (
        resolved.admissible_pool_keys
    )

    pool_resolution = resolve_pool_plan(
        resolved_problem=resolved,
        continuation_mode="phase3_v1",
        adapt_pool=H2O_LINEAR_FD_DERIVATIVE_RESOLVED_POOL_KEY,
        paop_r=1,
        paop_split_paulis=False,
        paop_prune_eps=1.0e-12,
        paop_normalization="none",
        phase3_symmetry_mitigation_mode="off",
    )
    assert pool_resolution.pool_key == H2O_LINEAR_FD_DERIVATIVE_RESOLVED_POOL_KEY
    assert any(
        term.label.startswith("conditional::") for term in pool_resolution.pool
    )


def test_generator_no_reference_cutoff_branch_is_explicit() -> None:
    fixture = build_h2o_linear_fd_fixture_from_record(
        synthetic_three_mode_h2o_linear_fd_backend_record(),
        mode_cutoffs=(1, 1, 1),
        reference_cutoffs=None,
        dense_full_dim_cap=512,
        embed_exact_state=True,
        require_reference_cutoff=False,
    )

    validate_paper_iv_h2o_linear_fd_evidence_fixture(
        fixture,
        require_exact_state=True,
        require_reference_cutoff=False,
    )
    assert fixture.cutoff_diagnostics is not None
    assert fixture.cutoff_diagnostics.reference_cutoffs is None
    assert fixture.cutoff_diagnostics.reference_ground_energy_hartree is None


def test_generator_rejects_missing_alt_displacements() -> None:
    record = deepcopy(synthetic_three_mode_h2o_linear_fd_backend_record())
    record["aligned_tensors"] = [
        row
        for row in record["aligned_tensors"]
        if not (row["mode_label"] == "bend" and row["step_kind"] == "alt")
    ]

    with pytest.raises(ValueError, match="alt plus/minus"):
        build_h2o_linear_fd_fixture_from_record(
            record,
            mode_cutoffs=(1, 1, 1),
            reference_cutoffs=(1, 1, 1),
            dense_full_dim_cap=512,
            embed_exact_state=True,
        )


def test_generator_rejects_failed_alignment_diagnostics() -> None:
    record = deepcopy(synthetic_three_mode_h2o_linear_fd_backend_record())
    record["aligned_tensors"][0]["alignment"]["min_singular_value"] = 0.1

    with pytest.raises(ValueError, match="alignment diagnostic failed"):
        build_h2o_linear_fd_fixture_from_record(
            record,
            mode_cutoffs=(1, 1, 1),
            reference_cutoffs=(1, 1, 1),
            dense_full_dim_cap=512,
            embed_exact_state=True,
        )
