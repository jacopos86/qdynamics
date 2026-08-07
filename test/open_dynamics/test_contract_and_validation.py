from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pipelines.open_dynamics import (
    ANOMALOUS_PHONON_KEY,
    COHERENT_PHONON_KEY,
    MethodIdentity,
    MomentSeries,
    NORMAL_PHONON_KEY,
    ReducedTrajectory,
    TrajectoryComparisonError,
    TrajectoryProvenance,
    TrajectoryValidationError,
    load_dimer_protocol,
    require_comparable,
    require_structurally_valid,
    validate_trajectory,
)


def _trajectory(
    *,
    time: np.ndarray | None = None,
    rho: np.ndarray | None = None,
    electron_number: float = 1.0,
    moments: dict[str, MomentSeries] | None = None,
) -> ReducedTrajectory:
    samples = np.array([0.0, 0.5]) if time is None else np.asarray(time)
    density = (
        np.repeat(np.diag([0.75, 0.25])[None, :, :], samples.size, axis=0)
        if rho is None
        else np.asarray(rho)
    )
    return ReducedTrajectory(
        time=np.asarray(samples, dtype=float),
        time_unit="1/t_hop",
        electron_1rdm=np.asarray(density, dtype=complex),
        electronic_basis=("site_1_spin_up", "site_2_spin_up"),
        electronic_basis_convention="test site basis",
        electron_number=electron_number,
        method=MethodIdentity(
            method_id="test",
            family="other",
            approximation="test",
            implementation="test",
            reference_access="none",
        ),
        provenance=TrajectoryProvenance(
            citations=("https://example.invalid/source",),
            paper_stated={"source.parameter": 1.0},
            repository_choices={"numerics.step": 0.5},
        ),
        moments={} if moments is None else moments,
    )


def test_protocol_manifest_separates_source_facts_and_repository_choices(
    dimer_protocol_path,
) -> None:
    protocol = load_dimer_protocol(dimer_protocol_path)

    assert protocol.paper_stated["model.gamma"] == 0.5
    assert protocol.paper_stated["model.lambda"] == 0.5
    assert protocol.repository_choices["numerics.integrator"] == "DOP853"
    assert not set(protocol.paper_stated).intersection(
        protocol.repository_choices
    )
    assert "Not an ab initio result" in protocol.claim_boundary


def test_contract_preserves_complex_moments_and_axis_conventions() -> None:
    coherent = np.array([[1.0 + 2.0j], [3.0 - 4.0j]])
    trajectory = _trajectory(
        moments={
            COHERENT_PHONON_KEY: MomentSeries(
                values=coherent,
                axes=("time", "mode"),
                unit="dimensionless",
                convention="test complex amplitude",
            )
        }
    )

    require_structurally_valid(trajectory)
    np.testing.assert_array_equal(
        trajectory.moments[COHERENT_PHONON_KEY].values,
        coherent,
    )
    assert trajectory.moments[COHERENT_PHONON_KEY].axes == ("time", "mode")


def test_validation_rejects_nonmonotone_time_and_misaligned_moments() -> None:
    bad = _trajectory(
        time=np.array([0.0, 0.5, 0.25]),
        rho=np.repeat(np.diag([0.75, 0.25])[None, :, :], 3, axis=0),
        moments={
            COHERENT_PHONON_KEY: MomentSeries(
                values=np.zeros((2, 1), dtype=complex),
                axes=("time", "mode"),
                unit="dimensionless",
                convention="test",
            )
        },
    )

    diagnostics = validate_trajectory(bad)
    assert any("strictly increasing" in item for item in diagnostics.structure_issues)
    assert any("time axis" in item for item in diagnostics.structure_issues)
    with pytest.raises(TrajectoryValidationError):
        require_structurally_valid(bad)


def test_validation_uses_declared_electron_number_not_unit_trace() -> None:
    density = np.repeat(np.eye(2, dtype=complex)[None, :, :], 2, axis=0)
    diagnostics = validate_trajectory(
        _trajectory(rho=density, electron_number=2.0)
    )

    assert diagnostics.structure_issues == ()
    assert not any("trace" in item for item in diagnostics.physicality_issues)
    assert diagnostics.metrics["electron_trace_max_abs_error"] == 0.0


def test_physicality_is_reported_without_mutation_or_rejection() -> None:
    density = np.array(
        [
            [[1.1, 0.2j], [0.0, -0.1]],
            [[1.1, 0.2j], [0.0, -0.1]],
        ],
        dtype=complex,
    )
    before = density.copy()
    trajectory = _trajectory(rho=density)

    diagnostics = validate_trajectory(trajectory)
    require_structurally_valid(trajectory)

    assert diagnostics.physicality_issues
    assert any("Hermitian" in item for item in diagnostics.physicality_issues)
    assert any("negative" in item for item in diagnostics.physicality_issues)
    np.testing.assert_array_equal(trajectory.electron_1rdm, before)


def test_normal_and_anomalous_symmetries_are_diagnosed() -> None:
    normal = np.array(
        [[[1.0, 1.0j], [0.0, 1.0]], [[1.0, 1.0j], [0.0, 1.0]]]
    )
    anomalous = np.array(
        [[[0.0, 1.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]],
        dtype=complex,
    )
    trajectory = _trajectory(
        moments={
            NORMAL_PHONON_KEY: MomentSeries(
                normal,
                ("time", "mode", "mode_prime"),
                "dimensionless",
                "test normal",
            ),
            ANOMALOUS_PHONON_KEY: MomentSeries(
                anomalous,
                ("time", "mode", "mode_prime"),
                "dimensionless",
                "test anomalous",
            ),
        }
    )

    diagnostics = validate_trajectory(trajectory)
    assert any("normal phonon" in item for item in diagnostics.physicality_issues)
    assert any("anomalous phonon" in item for item in diagnostics.physicality_issues)


def test_require_comparable_checks_grid_units_basis_and_initial_moments() -> None:
    anchor = _trajectory()
    require_comparable(anchor, _trajectory())

    with pytest.raises(TrajectoryComparisonError, match="time units"):
        require_comparable(anchor, replace(_trajectory(), time_unit="fs"))
    shifted = _trajectory()
    shifted.electron_1rdm[0, 0, 0] += 0.1
    with pytest.raises(TrajectoryComparisonError, match="initial electronic"):
        require_comparable(anchor, shifted)
