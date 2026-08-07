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
    ExactStateVectorRecord,
    FirstDerivativeRecord,
    FixtureManifest,
    GeometryRecord,
    NormalModeRecord,
    PhysicalSectorRecord,
    ProductionVibronicH2OFixture,
    RegisterLayout,
    build_production_vibronic_h2o_linear_fd_runtime_model,
    fixed_sector_dimension,
    h2o_linear_fd_boundary_weight_for_state,
    load_cached_production_vibronic_h2o_linear_fd_fixture,
    production_vibronic_h2o_fixture_from_jsonable,
    production_vibronic_h2o_fixture_to_jsonable,
    validate_paper_iv_h2o_linear_fd_evidence_fixture,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _poly(n_qubits: int, *, label: str | None = None, coeff: float = 1.0) -> dict[str, object]:
    pauli = "e" * int(n_qubits) if label is None else str(label)
    return {
        "repr": "JW",
        "n_qubits": int(n_qubits),
        "terms": [{"pauli": pauli, "coeff": {"re": float(coeff), "im": 0.0}}],
    }


def _reference_index() -> int:
    # Synthetic blocked active2 sector: alpha orbital 0 and beta orbital 0 occupied.
    return (1 << 0) | (1 << 2)


def _reference_state(layout: RegisterLayout) -> np.ndarray:
    state = np.zeros(2**layout.n_total_qubits, dtype=complex)
    state[_reference_index()] = 1.0
    return state


def _mode_vector(mode_index: int) -> np.ndarray:
    vec = np.zeros((3, 3), dtype=float)
    vec[int(mode_index), int(mode_index)] = 1.0
    return vec


def _exact_state(layout: RegisterLayout) -> ExactStateVectorRecord:
    bitstr = format(_reference_index(), f"0{layout.n_total_qubits}b")
    return ExactStateVectorRecord(
        available=True,
        representation="sparse_full_register_qn_to_q0",
        n_qubits=int(layout.n_total_qubits),
        norm=1.0,
        amplitudes_qn_to_q0={bitstr: {"re": 1.0, "im": 0.0}},
    )


def _synthetic_three_mode_fixture() -> ProductionVibronicH2OFixture:
    n_spatial = 2
    n_atoms = 3
    h = np.array([[0.2, 0.01], [0.01, 0.3]], dtype=float)
    g = np.zeros((n_spatial, n_spatial, n_spatial, n_spatial), dtype=float)
    mode_labels = ("bend", "symmetric_stretch", "antisymmetric_stretch")
    layout = RegisterLayout(
        n_fermion_qubits=4,
        fermion_qubits=(0, 1, 2, 3),
        boson_modes=tuple(
            BosonModeRegister(
                mode_index=idx,
                mode_label=label,
                qubit_start=4 + 2 * idx,
                n_qubits=2,
                n_ph_max=2,
            )
            for idx, label in enumerate(mode_labels)
        ),
    )
    aligned_tensors: list[AlignedActiveTensorRecord] = []
    alignment_diagnostics: list[AlignmentDiagnosticsRecord] = []
    displacements: list[DisplacedGeometryRecord] = []
    derivatives: list[FirstDerivativeRecord] = []
    for idx, label in enumerate(mode_labels):
        plus_id = f"plus_{label}"
        minus_id = f"minus_{label}"
        aligned_plus = f"aligned_plus_{label}"
        aligned_minus = f"aligned_minus_{label}"
        align_plus = f"align_plus_{label}"
        align_minus = f"align_minus_{label}"
        for sign, displacement_id, snapshot_id in (
            (1, plus_id, f"snap_plus_{label}"),
            (-1, minus_id, f"snap_minus_{label}"),
        ):
            displacements.append(
                DisplacedGeometryRecord(
                    displacement_id=displacement_id,
                    purpose="first_derivative",
                    mode_indices=(idx,),
                    signs=(sign,),
                    q_displacements_au=(0.1,),
                    geometry_id=f"h2o_{displacement_id}",
                    snapshot_id=snapshot_id,
                    coordinates_bohr=np.zeros((n_atoms, 3), dtype=float),
                )
            )
        aligned_tensors.extend(
            (
                AlignedActiveTensorRecord(
                    aligned_tensor_id=aligned_plus,
                    source_snapshot_id=f"snap_plus_{label}",
                    displacement_id=plus_id,
                    scalar_energy_hartree=0.01,
                    one_body_integrals=h,
                    two_body_integrals=g,
                    alignment_id=align_plus,
                ),
                AlignedActiveTensorRecord(
                    aligned_tensor_id=aligned_minus,
                    source_snapshot_id=f"snap_minus_{label}",
                    displacement_id=minus_id,
                    scalar_energy_hartree=-0.01,
                    one_body_integrals=h,
                    two_body_integrals=g,
                    alignment_id=align_minus,
                ),
            )
        )
        for alignment_id, displacement_id, snapshot_id in (
            (align_plus, plus_id, f"snap_plus_{label}"),
            (align_minus, minus_id, f"snap_minus_{label}"),
        ):
            alignment_diagnostics.append(
                AlignmentDiagnosticsRecord(
                    alignment_id=alignment_id,
                    center_snapshot_id="snap_center",
                    displaced_snapshot_id=snapshot_id,
                    displacement_id=displacement_id,
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
                )
            )
        derivatives.append(
            FirstDerivativeRecord(
                derivative_id=f"d_{label}",
                mode_index=idx,
                mode_label=label,
                q_step_au=0.1,
                plus_aligned_tensor_id=aligned_plus,
                minus_aligned_tensor_id=aligned_minus,
                scalar_derivative_hartree_per_q=0.1,
                one_body_derivative=np.zeros_like(h),
                two_body_derivative=np.zeros_like(g),
                scalar_derivative_included=True,
                scalar_derivative_convention="nuclear_repulsion_plus_closed_shell_frozen_core_scalar",
                finite_difference_drift=0.0,
                passed=True,
            )
        )

    sector_dim = fixed_sector_dimension(
        n_spatial_orbitals=n_spatial,
        num_particles=(1, 1),
        mode_cutoffs=(2, 2, 2),
    )
    exact_boundary = BoundaryWeightRecord(
        total_boundary_weight=0.0,
        per_mode_boundary_weight={label: 0.0 for label in mode_labels},
        state_source="exact_ground_state",
    )
    return ProductionVibronicH2OFixture(
        manifest=FixtureManifest(
            schema=H2O_LINEAR_FD_FIXTURE_SCHEMA,
            schema_version=1,
            family_key=H2O_LINEAR_FD_FAMILY_KEY,
            molecule_family_key=H2O_UMBRELLA_FAMILY_KEY,
            model_role=H2O_LINEAR_FD_MODEL_ROLE,
            production_status="production_validated",
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
        normal_modes=tuple(
            NormalModeRecord(
                mode_index=idx,
                label=label,
                frequency_hartree=0.01 + 0.001 * idx,
                frequency_cm1=None,
                mass_weighted_eigenvector=_mode_vector(idx),
                q_step_au=0.1,
                q_step_alt_au=0.05,
            )
            for idx, label in enumerate(mode_labels)
        ),
        displacements=tuple(displacements),
        active_space=ActiveSpaceRecord(
            active_space_kind="synthetic_active2_three_mode_variant",
            frozen_core_indices=(),
            active_indices_center=(0, 1),
            external_indices=(),
            n_spatial_orbitals=n_spatial,
            num_particles=(1, 1),
            scalar_energy_hartree=0.0,
            one_body_integrals=h,
            two_body_integrals=g,
        ),
        aligned_tensors=tuple(aligned_tensors),
        alignment_diagnostics=tuple(alignment_diagnostics),
        first_derivatives=tuple(derivatives),
        layout=layout,
        physical_sector=PhysicalSectorRecord(
            n_alpha=1,
            n_beta=1,
            n_ph_max_by_mode=(2, 2, 2),
            mode_labels=mode_labels,
        ),
        encoded_operators=EncodedOperatorBundle(
            h_electronic=_poly(4, coeff=0.1),
            dH_dQ_by_mode=tuple(_poly(4, coeff=0.01 * (idx + 1)) for idx in range(3)),
            h_vibronic=_poly(layout.n_total_qubits, coeff=-1.0),
            q_by_mode=tuple(_poly(layout.n_total_qubits, label="e" * (layout.n_total_qubits - 1) + "x") for _ in range(3)),
            p_by_mode=tuple(_poly(layout.n_total_qubits, label="e" * (layout.n_total_qubits - 1) + "y") for _ in range(3)),
            n_by_mode=tuple(_poly(layout.n_total_qubits, label="e" * (layout.n_total_qubits - 1) + "z") for _ in range(3)),
        ),
        reference_state=_reference_state(layout),
        exact_reference=ExactReferenceRecord(
            available=True,
            method="dense_sector_eigh",
            sector_dimension=sector_dim,
            full_qubit_dimension=2**layout.n_total_qubits,
            ground_energy_hartree=-1.0,
            low_energies_hartree=(-1.0,),
            boundary_weight=exact_boundary,
            ground_state=_exact_state(layout),
        ),
        cutoff_diagnostics=CutoffDiagnosticsRecord(
            work_cutoffs=(2, 2, 2),
            reference_cutoffs=(3, 3, 3),
            work_ground_energy_hartree=-1.0,
            reference_ground_energy_hartree=-1.01,
            delta_energy_hartree=0.01,
            work_boundary_weight=exact_boundary,
            passed=True,
            policy="synthetic",
            energy_tolerance_hartree=2.0e-2,
            boundary_weight_tolerance=1.0e-2,
            energy_passed=True,
            boundary_passed=True,
        ),
        evidence_hooks=EvidenceHooksRecord(
            static_ground_state_ready=True,
            exact_reference_ready=True,
            qse_hooks_ready=False,
            qse_probe_families=("Q_mu", "P_mu"),
        ),
        pool=(
            {
                "label": "D_bend_P_bend",
                "polynomial": _poly(layout.n_total_qubits, label="e" * (layout.n_total_qubits - 1) + "y"),
            },
        ),
        report_summary={"paper_iv_active_space_variant": "synthetic_active2_three_mode_test"},
    )


def test_strict_evidence_validator_and_runtime_model() -> None:
    fixture = _synthetic_three_mode_fixture()

    validate_paper_iv_h2o_linear_fd_evidence_fixture(
        fixture,
        require_exact_state=True,
        require_reference_cutoff=True,
    )
    model = build_production_vibronic_h2o_linear_fd_runtime_model(
        fixture,
        require_paper_iv_evidence=True,
        require_exact_state=True,
    )

    assert model.n_total_qubits == 10
    assert model.n_fermion_qubits == 4
    assert model.n_boson_qubits == 6
    assert model.mode_labels == ("bend", "symmetric_stretch", "antisymmetric_stretch")
    assert model.mode_cutoffs == (2, 2, 2)
    assert model.num_particles == (1, 1)
    assert len(model.dH_dQ_by_mode) == 3
    assert len(model.q_by_mode) == 3
    assert len(model.p_by_mode) == 3
    assert len(model.n_by_mode) == 3
    assert len(model.pool) == 1
    assert model.pool[0].label == "D_bend_P_bend"
    assert np.vdot(model.psi_ref, model.psi_ref).real == pytest.approx(1.0)


def test_exact_state_round_trip_and_cached_loader(tmp_path: Path) -> None:
    fixture = _synthetic_three_mode_fixture()
    payload = production_vibronic_h2o_fixture_to_jsonable(fixture)
    loaded = production_vibronic_h2o_fixture_from_jsonable(json.loads(json.dumps(payload)))

    assert loaded.exact_reference.ground_state is not None
    assert loaded.exact_reference.ground_state.available
    validate_paper_iv_h2o_linear_fd_evidence_fixture(loaded, require_exact_state=True)

    path = tmp_path / "h2o_linear_fd_fixture.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    cached = load_cached_production_vibronic_h2o_linear_fd_fixture(path, require_exact_state=True)

    assert cached.fixture_path == path
    assert cached.metadata["family_key"] == H2O_LINEAR_FD_FAMILY_KEY
    assert cached.model.n_total_qubits == loaded.layout.n_total_qubits


def test_strict_validator_rejects_old_h2o_schema() -> None:
    v1_path = REPO_ROOT / "test_support" / "molecular_vibronic_h2o_sto3g_active2_fd001.json"
    payload = json.loads(v1_path.read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="not a production H2O linear-FD fixture"):
        production_vibronic_h2o_fixture_from_jsonable(payload)


def test_strict_validator_rejects_missing_scalar_derivative() -> None:
    fixture = _synthetic_three_mode_fixture()
    bad_derivative = replace(
        fixture.first_derivatives[0],
        scalar_derivative_included=False,
    )
    bad = replace(fixture, first_derivatives=(bad_derivative, *fixture.first_derivatives[1:]))

    with pytest.raises(ValueError, match="scalar derivative"):
        validate_paper_iv_h2o_linear_fd_evidence_fixture(bad)


def test_strict_validator_requires_exact_state_when_requested() -> None:
    fixture = _synthetic_three_mode_fixture()
    exact_without_state = replace(fixture.exact_reference, ground_state=None)
    bad = replace(fixture, exact_reference=exact_without_state)

    validate_paper_iv_h2o_linear_fd_evidence_fixture(bad, require_exact_state=False)
    with pytest.raises(ValueError, match="exact state"):
        validate_paper_iv_h2o_linear_fd_evidence_fixture(bad, require_exact_state=True)


def test_strict_validator_rejects_embedded_exact_state_amplitude_norm_mismatch() -> None:
    fixture = _synthetic_three_mode_fixture()
    assert fixture.exact_reference.ground_state is not None
    bad_state = replace(
        fixture.exact_reference.ground_state,
        amplitudes_qn_to_q0={
            bitstr: {"re": 0.5 * float(coeff["re"]), "im": 0.5 * float(coeff["im"])}
            for bitstr, coeff in fixture.exact_reference.ground_state.amplitudes_qn_to_q0.items()
        },
    )
    bad_exact = replace(fixture.exact_reference, ground_state=bad_state)
    bad = replace(fixture, exact_reference=bad_exact)

    with pytest.raises(ValueError, match="embedded exact-state amplitudes"):
        validate_paper_iv_h2o_linear_fd_evidence_fixture(bad, require_exact_state=True)


def test_strict_validator_requires_reference_cutoff_when_requested() -> None:
    fixture = _synthetic_three_mode_fixture()
    bad_cutoff = replace(
        fixture.cutoff_diagnostics,
        reference_cutoffs=None,
        reference_ground_energy_hartree=None,
        delta_energy_hartree=None,
        passed=False,
        energy_passed=None,
    )
    bad = replace(fixture, cutoff_diagnostics=bad_cutoff)

    validate_paper_iv_h2o_linear_fd_evidence_fixture(bad, require_reference_cutoff=False)
    with pytest.raises(ValueError, match="reference cutoffs"):
        validate_paper_iv_h2o_linear_fd_evidence_fixture(bad, require_reference_cutoff=True)


def test_cutoff_convergence_gate_is_quantitative_and_optional() -> None:
    fixture = _synthetic_three_mode_fixture()
    failed_cutoff = replace(
        fixture.cutoff_diagnostics,
        energy_tolerance_hartree=1.0e-3,
        energy_passed=False,
        passed=False,
    )
    diagnostic_fixture = replace(fixture, cutoff_diagnostics=failed_cutoff)

    validate_paper_iv_h2o_linear_fd_evidence_fixture(
        diagnostic_fixture,
        require_cutoff_converged=False,
    )
    with pytest.raises(ValueError, match="cutoff convergence criteria"):
        validate_paper_iv_h2o_linear_fd_evidence_fixture(
            diagnostic_fixture,
            require_cutoff_converged=True,
        )


def test_strict_validator_rejects_failed_alignment() -> None:
    fixture = _synthetic_three_mode_fixture()
    bad_diag = replace(fixture.alignment_diagnostics[0], passed=False)
    bad = replace(fixture, alignment_diagnostics=(bad_diag, *fixture.alignment_diagnostics[1:]))

    with pytest.raises(ValueError, match="alignment diagnostic failed"):
        validate_paper_iv_h2o_linear_fd_evidence_fixture(bad)


def test_boundary_weight_uses_valid_mode_occupations_and_rejects_padded_codes() -> None:
    fixture = _synthetic_three_mode_fixture()
    layout = fixture.layout
    state = np.zeros(2**layout.n_total_qubits, dtype=complex)
    state[_reference_index()] = np.sqrt(0.75)
    bend_boundary_index = _reference_index() | (2 << layout.boson_modes[0].qubit_start)
    state[bend_boundary_index] = np.sqrt(0.25)

    weights = h2o_linear_fd_boundary_weight_for_state(
        state,
        layout=layout,
        state_source="unit_test",
    )

    assert weights.total_boundary_weight == pytest.approx(0.25)
    assert weights.per_mode_boundary_weight["bend"] == pytest.approx(0.25)
    assert weights.per_mode_boundary_weight["symmetric_stretch"] == pytest.approx(0.0)
    assert weights.per_mode_boundary_weight["antisymmetric_stretch"] == pytest.approx(0.0)

    invalid = np.zeros_like(state)
    invalid_index = _reference_index() | (3 << layout.boson_modes[0].qubit_start)
    invalid[invalid_index] = 1.0
    with pytest.raises(ValueError, match="invalid padded binary boson code"):
        h2o_linear_fd_boundary_weight_for_state(
            invalid,
            layout=layout,
            state_source="invalid",
        )


def test_strict_validator_rejects_legacy_pool_labels() -> None:
    fixture = _synthetic_three_mode_fixture()
    bad = replace(
        fixture,
        pool=(
            {
                "label": "frontier_surrogate_active2_generator",
                "polynomial": _poly(fixture.layout.n_total_qubits),
            },
        ),
    )

    with pytest.raises(ValueError, match="legacy smoke/prototype pool label"):
        validate_paper_iv_h2o_linear_fd_evidence_fixture(bad)
