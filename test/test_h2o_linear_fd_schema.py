from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from src.quantum.chemistry.vibronic_h2o_linear_fd import (
    H2O_LINEAR_FD_DERIVATIVE_SOURCE,
    H2O_LINEAR_FD_FAMILY_KEY,
    H2O_LINEAR_FD_FIXTURE_SCHEMA,
    H2O_LINEAR_FD_MODEL_ROLE,
    H2O_UMBRELLA_FAMILY_KEY,
    ActiveSpaceRecord,
    AlignedActiveTensorRecord,
    AlignmentDiagnosticsRecord,
    AlignmentThresholds,
    BosonModeRegister,
    BoundaryWeightRecord,
    CutoffDiagnosticsRecord,
    DisplacedGeometryRecord,
    EncodedOperatorBundle,
    EvidenceHooksRecord,
    ExactReferenceRecord,
    FirstDerivativeRecord,
    FixtureManifest,
    GeometryRecord,
    NormalModeRecord,
    PhysicalSectorRecord,
    ProductionVibronicH2OFixture,
    RegisterLayout,
    fixed_sector_dimension,
    production_vibronic_h2o_fixture_from_jsonable,
    production_vibronic_h2o_fixture_to_jsonable,
    validate_production_vibronic_h2o_fixture,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _reference_state(layout: RegisterLayout) -> np.ndarray:
    state = np.zeros(2**layout.n_total_qubits, dtype=complex)
    # n_spatial=2, (N_alpha,N_beta)=(1,1), blocked ordering: alpha 0 and beta 0.
    state[(1 << 0) | (1 << 2)] = 1.0
    return state


def _synthetic_fixture(*, production_status: str = "production_validated") -> ProductionVibronicH2OFixture:
    n_spatial = 2
    n_atoms = 3
    h = np.array([[0.2, 0.01], [0.01, 0.3]], dtype=float)
    g = np.zeros((n_spatial, n_spatial, n_spatial, n_spatial), dtype=float)
    mode_vec = np.zeros((n_atoms, 3), dtype=float)
    mode_vec[0, 0] = 1.0

    layout = RegisterLayout(
        n_fermion_qubits=4,
        fermion_qubits=(0, 1, 2, 3),
        boson_modes=(
            BosonModeRegister(
                mode_index=0,
                mode_label="bend",
                qubit_start=4,
                n_qubits=1,
                n_ph_max=1,
            ),
        ),
    )
    sector_dim = fixed_sector_dimension(
        n_spatial_orbitals=n_spatial,
        num_particles=(1, 1),
        mode_cutoffs=(1,),
    )
    return ProductionVibronicH2OFixture(
        manifest=FixtureManifest(
            schema=H2O_LINEAR_FD_FIXTURE_SCHEMA,
            schema_version=1,
            family_key=H2O_LINEAR_FD_FAMILY_KEY,
            molecule_family_key=H2O_UMBRELLA_FAMILY_KEY,
            model_role=H2O_LINEAR_FD_MODEL_ROLE,
            production_status=production_status,  # type: ignore[arg-type]
            derivative_source=H2O_LINEAR_FD_DERIVATIVE_SOURCE,
            created_utc="2026-06-25T00:00:00Z",
            generator_version="unit_test",
            repository_commit=None,
        ),
        geometry=GeometryRecord(
            geometry_id="h2o_center",
            molecule="H2O",
            symbols=("O", "H", "H"),
            coordinates_bohr=np.zeros((n_atoms, 3), dtype=float),
            masses_me=np.array([29156.945, 1837.153, 1837.153], dtype=float),
            charge=0,
            multiplicity=1,
            method="synthetic",
            basis="synthetic",
            reference="RHF",
            optimized=True,
        ),
        normal_modes=(
            NormalModeRecord(
                mode_index=0,
                label="bend",
                frequency_hartree=0.01,
                frequency_cm1=None,
                mass_weighted_eigenvector=mode_vec,
                q_step_au=0.1,
                q_step_alt_au=0.05,
            ),
        ),
        displacements=(
            DisplacedGeometryRecord(
                displacement_id="plus_mode0",
                purpose="first_derivative",
                mode_indices=(0,),
                signs=(1,),
                q_displacements_au=(0.1,),
                geometry_id="h2o_plus",
                snapshot_id="snap_plus",
                coordinates_bohr=np.zeros((n_atoms, 3), dtype=float),
            ),
            DisplacedGeometryRecord(
                displacement_id="minus_mode0",
                purpose="first_derivative",
                mode_indices=(0,),
                signs=(-1,),
                q_displacements_au=(0.1,),
                geometry_id="h2o_minus",
                snapshot_id="snap_minus",
                coordinates_bohr=np.zeros((n_atoms, 3), dtype=float),
            ),
        ),
        active_space=ActiveSpaceRecord(
            active_space_kind="synthetic_active2",
            frozen_core_indices=(),
            active_indices_center=(0, 1),
            external_indices=(),
            n_spatial_orbitals=n_spatial,
            num_particles=(1, 1),
            scalar_energy_hartree=0.0,
            one_body_integrals=h,
            two_body_integrals=g,
        ),
        aligned_tensors=(
            AlignedActiveTensorRecord(
                aligned_tensor_id="aligned_plus",
                source_snapshot_id="snap_plus",
                displacement_id="plus_mode0",
                scalar_energy_hartree=0.01,
                one_body_integrals=h,
                two_body_integrals=g,
                alignment_id="align_plus",
            ),
            AlignedActiveTensorRecord(
                aligned_tensor_id="aligned_minus",
                source_snapshot_id="snap_minus",
                displacement_id="minus_mode0",
                scalar_energy_hartree=-0.01,
                one_body_integrals=h,
                two_body_integrals=g,
                alignment_id="align_minus",
            ),
        ),
        alignment_diagnostics=(
            AlignmentDiagnosticsRecord(
                alignment_id="align_plus",
                center_snapshot_id="snap_center",
                displaced_snapshot_id="snap_plus",
                displacement_id="plus_mode0",
                block="active",
                singular_values=np.array([1.0, 0.999], dtype=float),
                min_singular_value=0.999,
                alignment_residual_fro=1.0e-12,
                active_to_external_leakage_fro=0.0,
                external_to_active_leakage_fro=0.0,
                hermiticity_residual=0.0,
                eri_symmetry_residual=0.0,
                rotation_orthogonality_residual=0.0,
                thresholds=AlignmentThresholds(),
                passed=True,
            ),
        ),
        first_derivatives=(
            FirstDerivativeRecord(
                derivative_id="d_bend",
                mode_index=0,
                mode_label="bend",
                q_step_au=0.1,
                plus_aligned_tensor_id="aligned_plus",
                minus_aligned_tensor_id="aligned_minus",
                scalar_derivative_hartree_per_q=0.1,
                one_body_derivative=np.zeros_like(h),
                two_body_derivative=np.zeros_like(g),
                scalar_derivative_included=True,
                scalar_derivative_convention="nuclear_repulsion_plus_closed_shell_frozen_core_scalar",
                passed=True,
            ),
        ),
        layout=layout,
        physical_sector=PhysicalSectorRecord(
            n_alpha=1,
            n_beta=1,
            n_ph_max_by_mode=(1,),
            mode_labels=("bend",),
        ),
        encoded_operators=EncodedOperatorBundle(
            h_electronic={"terms": []},
            dH_dQ_by_mode=({"terms": []},),
            h_vibronic={"terms": []},
        ),
        reference_state=_reference_state(layout),
        exact_reference=ExactReferenceRecord(
            available=True,
            method="dense_sector_eigh",
            sector_dimension=sector_dim,
            full_qubit_dimension=2**layout.n_total_qubits,
            ground_energy_hartree=-1.0,
            low_energies_hartree=(-1.0,),
            boundary_weight=BoundaryWeightRecord(
                total_boundary_weight=0.0,
                per_mode_boundary_weight={"bend": 0.0},
                state_source="exact_ground_state",
            ),
        ),
        cutoff_diagnostics=CutoffDiagnosticsRecord(
            work_cutoffs=(1,),
            reference_cutoffs=(2,),
            work_ground_energy_hartree=-1.0,
            reference_ground_energy_hartree=-1.01,
            delta_energy_hartree=0.01,
            work_boundary_weight=BoundaryWeightRecord(
                total_boundary_weight=0.0,
                per_mode_boundary_weight={"bend": 0.0},
                state_source="exact_ground_state",
            ),
            passed=True,
            policy="synthetic",
        ),
        evidence_hooks=EvidenceHooksRecord(
            static_ground_state_ready=True,
            exact_reference_ready=True,
            qse_hooks_ready=False,
            qse_probe_families=("Q_mu", "P_mu"),
        ),
    )


def test_h2o_linear_fd_schema_round_trip_without_psi4() -> None:
    fixture = _synthetic_fixture()
    validate_production_vibronic_h2o_fixture(fixture)

    payload = production_vibronic_h2o_fixture_to_jsonable(fixture)
    encoded = json.loads(json.dumps(payload))
    loaded = production_vibronic_h2o_fixture_from_jsonable(encoded)

    validate_production_vibronic_h2o_fixture(loaded)
    assert loaded.manifest.schema == H2O_LINEAR_FD_FIXTURE_SCHEMA
    assert loaded.manifest.family_key == H2O_LINEAR_FD_FAMILY_KEY
    assert loaded.layout.n_total_qubits == fixture.layout.n_total_qubits
    assert np.vdot(loaded.reference_state, loaded.reference_state).real == pytest.approx(1.0)


def test_h2o_linear_fd_rejects_v1_surrogate_as_production() -> None:
    v1_path = REPO_ROOT / "test_support" / "molecular_vibronic_h2o_sto3g_active2_fd001.json"
    payload = json.loads(v1_path.read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="not a production H2O linear-FD fixture"):
        production_vibronic_h2o_fixture_from_jsonable(payload)


def test_h2o_linear_fd_candidate_requires_explicit_validation_policy() -> None:
    fixture = _synthetic_fixture(production_status="production_candidate")

    with pytest.raises(ValueError, match="not production_validated"):
        validate_production_vibronic_h2o_fixture(fixture)

    validate_production_vibronic_h2o_fixture(fixture, require_production_validated=False)


def test_h2o_linear_fd_mode_metadata_validation_fails_closed() -> None:
    fixture = _synthetic_fixture()
    bad = replace(fixture, first_derivatives=())

    with pytest.raises(ValueError, match="number of derivative records"):
        validate_production_vibronic_h2o_fixture(bad)


def test_h2o_linear_fd_alignment_diagnostics_fail_closed() -> None:
    fixture = _synthetic_fixture()
    bad_diag = replace(
        fixture.alignment_diagnostics[0],
        min_singular_value=0.90,
        passed=False,
    )
    bad = replace(fixture, alignment_diagnostics=(bad_diag,))

    with pytest.raises(ValueError, match="alignment diagnostic failed"):
        validate_production_vibronic_h2o_fixture(bad)


def test_h2o_linear_fd_fixed_sector_dimension_cas_8e_6o() -> None:
    assert fixed_sector_dimension(
        n_spatial_orbitals=6,
        num_particles=(4, 4),
        mode_cutoffs=(2, 2, 2),
    ) == 225 * 27
    assert fixed_sector_dimension(
        n_spatial_orbitals=6,
        num_particles=(4, 4),
        mode_cutoffs=(3, 3, 3),
    ) == 225 * 64
    assert fixed_sector_dimension(
        n_spatial_orbitals=6,
        num_particles=(4, 4),
        mode_cutoffs=(4, 4, 4),
    ) == 225 * 125

