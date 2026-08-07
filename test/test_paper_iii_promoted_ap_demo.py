from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.excited_dynamics.paper_iii_promoted_ap_demo import (
    PromotedAPDemoError,
    _assert_normalized_pool_preserves_sector,
    _convergence_metrics,
    _controller_drive,
    _locked_drive,
    _trajectory_metrics,
)


def _row(
    *,
    time: float,
    density_ap: float,
    density_exact: float,
    phonon_ap: float,
    phonon_exact: float,
    fidelity: float,
) -> dict[str, float | int | None]:
    return {
        "step_index": int(round(10.0 * time)),
        "time": float(time),
        "ap_exact_state_fidelity": float(fidelity),
        "ap_sector_weight": 1.0,
        "staggered_density_ap": float(density_ap),
        "staggered_density_exact": float(density_exact),
        "staggered_phonon_displacement_ap": float(phonon_ap),
        "staggered_phonon_displacement_exact": float(phonon_exact),
        "static_energy_ap": -1.0,
        "static_energy_exact": -1.0,
        "mclachlan_residual_ratio": 0.01,
        "mclachlan_rank": 2,
        "mclachlan_condition_number": 3.0,
    }


def test_trajectory_metrics_use_raw_locked_observable_errors() -> None:
    rows = [
        _row(
            time=0.0,
            density_ap=0.0,
            density_exact=0.0,
            phonon_ap=0.0,
            phonon_exact=0.0,
            fidelity=1.0,
        ),
        _row(
            time=0.1,
            density_ap=0.13,
            density_exact=0.10,
            phonon_ap=-0.16,
            phonon_exact=-0.10,
            fidelity=0.997,
        ),
    ]

    metrics = _trajectory_metrics(rows)

    assert metrics["maximum_staggered_density_abs_error"] == pytest.approx(0.03)
    assert metrics["maximum_staggered_phonon_abs_error"] == pytest.approx(0.06)
    assert metrics["minimum_ap_exact_state_fidelity"] == pytest.approx(0.997)


def test_convergence_metrics_compare_only_nested_common_times() -> None:
    coarse_rows = [
        _row(
            time=0.0,
            density_ap=0.0,
            density_exact=0.0,
            phonon_ap=0.0,
            phonon_exact=0.0,
            fidelity=1.0,
        ),
        _row(
            time=0.1,
            density_ap=0.11,
            density_exact=0.1,
            phonon_ap=0.21,
            phonon_exact=0.2,
            fidelity=1.0,
        ),
    ]
    fine_rows = [
        copy.deepcopy(coarse_rows[0]),
        _row(
            time=0.05,
            density_ap=0.05,
            density_exact=0.05,
            phonon_ap=0.1,
            phonon_exact=0.1,
            fidelity=1.0,
        ),
        _row(
            time=0.1,
            density_ap=0.10,
            density_exact=0.1,
            phonon_ap=0.20,
            phonon_exact=0.2,
            fidelity=1.0,
        ),
    ]
    zero = np.asarray([1.0, 0.0], dtype=complex)
    tilted = np.asarray([np.sqrt(0.999), np.sqrt(0.001)], dtype=complex)

    metrics = _convergence_metrics(
        coarse={"dt": 0.1, "trajectory": coarse_rows},
        fine={"dt": 0.05, "trajectory": fine_rows},
        coarse_states=(zero, tilted),
        fine_states=(zero, zero, zero),
    )

    assert metrics["maximum_density_abs_delta"] == pytest.approx(0.01)
    assert metrics["maximum_phonon_abs_delta"] == pytest.approx(0.01)
    assert metrics["minimum_state_fidelity"] == pytest.approx(0.999)


def test_locked_drive_fails_closed_if_observable_identity_changes() -> None:
    payload = {
        "dynamics": {
            "drive": {
                "amplitude": 0.05,
                "omega": 0.8,
                "tbar": 4.0,
                "phi": 0.0,
                "operator": "wrong_operator",
                "spatial_pattern": "staggered",
            },
            "metrics": {
                "dt": 0.05,
                "exact_reference_method": "fixed_sector_exponential_midpoint_magnus2_order2",
                "exact_reference_used_for_controller_or_drive_selection": False,
            },
            "trajectory": [{"time": 0.0}, {"time": 0.05}],
        }
    }

    with pytest.raises(PromotedAPDemoError, match="operator"):
        _locked_drive(payload)


def test_controller_drive_strips_locked_reference_rows() -> None:
    drive = {
        "amplitude": 0.05,
        "omega": 0.8,
        "tbar": 4.0,
        "phi": 0.0,
        "t_final": 10.0,
        "locked_rows": [{"staggered_density_exact": 1.0}],
        "payload": {"operator": "hh_n[staggered]"},
    }

    sanitized = _controller_drive(drive)

    assert sanitized == {
        "amplitude": 0.05,
        "omega": 0.8,
        "tbar": 4.0,
        "phi": 0.0,
        "t_final": 10.0,
    }


def test_normalized_pool_legality_fails_closed_on_sector_leakage() -> None:
    legal = SimpleNamespace(
        profile="toy",
        atoms=(SimpleNamespace(pauli_exyz="z", nq=1),),
    )
    leaking = SimpleNamespace(
        profile="toy",
        atoms=(SimpleNamespace(pauli_exyz="x", nq=1),),
    )

    receipt = _assert_normalized_pool_preserves_sector(
        legal,
        sector_indices=np.asarray([0], dtype=int),
    )

    assert receipt["status"] == "passed"
    with pytest.raises(PromotedAPDemoError, match="leave the locked particle sector"):
        _assert_normalized_pool_preserves_sector(
            leaking,
            sector_indices=np.asarray([0], dtype=int),
        )
