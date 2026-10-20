"""The conic instance recorder is off by default and changes no number."""

from __future__ import annotations

import clarabel
import numpy as np
import pytest

from paper5.stability.apcm_carried_witness import CarriedWitnessModel
from paper5.stability.cwrmf_solver_instances import (
    CONIC_INSTANCE_FORMAT,
    ConicInstanceRecorder,
    build_cones,
    build_solver_settings,
    load_instance_context,
    load_instances,
)
from paper5.stability.hubbard_dimer import DimerParameters

_STEP = 1e-8


@pytest.fixture(scope="module")
def guarded_atom_state():
    """A state whose radial atom enters the conic guard, cheaply.

    The completion coordinate ``0`` is displaced by ``1e-5``, which is the
    same displacement the event-guard tests use to push the endpoint 62-row
    Gram below the entry threshold.
    """

    model = CarriedWitnessModel(
        DimerParameters(lambda_ep=1.5, gamma=0.5, drive_amplitude=1.0)
    )
    preparation = model.prepare(phonon_cutoff=16)
    retained, completion = model.geometry.unpack_state(preparation.state)
    perturbed = completion.copy()
    perturbed[0] += 1e-5
    return model, model.geometry.pack_state(retained, perturbed)


def _recorder(model, directory):
    recorder = ConicInstanceRecorder(directory)
    recorder.write_context(
        step_scaled_completion_coefficients=(
            _STEP * model.geometry.scaled_completion_coefficients()
        ),
        readable_completion_indices=(
            model.geometry.readable_completion_indices
        ),
        completion_count=model.geometry.completion_count,
        numerical_profile="strict",
        schur_safety_margin=model.settings.schur_safety_margin,
        solver_tolerance=model.settings.solver_tolerance,
        psd_inflation=model.settings.psd_inflation,
    )
    return recorder


def test_instance_recording_is_off_by_default(guarded_atom_state) -> None:
    model, _ = guarded_atom_state

    assert model.instance_recorder is None


def test_recorded_atom_reproduces_the_default_path_bitwise(
    guarded_atom_state, tmp_path
) -> None:
    model, state = guarded_atom_state
    default = model.radial_atom(0.0, state, _STEP)

    recorder = _recorder(model, tmp_path / "instances")
    model.enable_instance_recording(recorder)
    try:
        recorded = model.radial_atom(0.0, state, _STEP)
    finally:
        model._instance_recorder = None

    assert model.instance_recorder is None
    assert recorder.instance_count >= 1
    assert recorded.message == default.message
    assert recorded.success == default.success
    for name in (
        "endpoint",
        "archive_velocity",
        "completion_velocity",
        "desired_completion_velocity",
    ):
        assert np.array_equal(
            getattr(recorded, name), getattr(default, name)
        ), name
    assert (
        recorded.readable_rate_residual == default.readable_rate_residual
    )
    assert recorded.critical_modes == default.critical_modes


def test_recorded_instance_replays_to_the_same_clarabel_solution(
    guarded_atom_state, tmp_path
) -> None:
    model, state = guarded_atom_state
    recorder = _recorder(model, tmp_path / "instances")
    model.enable_instance_recording(recorder)
    try:
        model.radial_atom(0.0, state, _STEP)
    finally:
        model._instance_recorder = None

    coefficients, readable, metadata = load_instance_context(
        tmp_path / "instances"
    )
    instances = load_instances(tmp_path / "instances")

    assert metadata["format"] == CONIC_INSTANCE_FORMAT
    assert metadata["numerical_profile"] == "strict"
    assert coefficients.shape == (340, 62, 62)
    assert readable.size == 231
    assert instances

    instance = instances[0]
    assert instance.tier in {"bundle_cone", "critical_schur"}
    assert instance.atom_index == 0
    assert instance.step_size == _STEP
    assert instance.solver_settings["max_threads"] == 1
    assert instance.cone_specification[0][0] == "PSDTriangleConeT"

    solution = clarabel.DefaultSolver(
        instance.objective_matrix,
        instance.objective_vector,
        instance.cone_matrix,
        instance.cone_rhs,
        build_cones(instance.cone_specification),
        build_solver_settings(instance.solver_settings),
    ).solve()

    assert str(solution.status) == instance.recorded_status
    assert np.array_equal(
        np.asarray(solution.x, dtype=float), instance.recorded_solution
    )


def test_reconstructed_gram_is_the_endpoint_unified_matrix(
    guarded_atom_state, tmp_path
) -> None:
    model, state = guarded_atom_state
    recorder = _recorder(model, tmp_path / "instances")
    model.enable_instance_recording(recorder)
    try:
        model.radial_atom(0.0, state, _STEP)
    finally:
        model._instance_recorder = None

    coefficients, _, _ = load_instance_context(tmp_path / "instances")
    instance = load_instances(tmp_path / "instances")[0]
    retained, completion = model.geometry.unpack_state(state)
    endpoint_retained = retained + _STEP * model.retained_velocity(
        0.0, retained, completion
    )
    candidate = instance.candidate_velocity(instance.recorded_solution)
    expected = (
        model.geometry.scaled_unified_matrix(
            endpoint_retained, completion + _STEP * candidate
        )
        + model.settings.psd_inflation * np.eye(62, dtype=complex)
    )

    reconstructed = instance.reconstructed_gram(
        coefficients, instance.recorded_solution
    )

    # The reconstruction sums the coefficient contraction in a different
    # order from the literal matrix build, so it agrees to roundoff, not
    # bitwise: the observed departure here is 1.39e-17 absolute and
    # 3.40e-16 relative, five decades below the strict acceptance margin
    # 2e-12 the minimum eigenvalue is compared against.
    np.testing.assert_allclose(
        reconstructed, expected, rtol=0.0, atol=1e-15
    )


def test_build_solver_settings_rejects_an_unknown_field() -> None:
    with pytest.raises(ValueError, match="unknown Clarabel setting"):
        build_solver_settings({"max_threads": 1}, {"not_a_setting": 3})
