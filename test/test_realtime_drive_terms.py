from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipelines.time_dynamics.legacy.checkpoint_controller as controller_mod
from pipelines.time_dynamics.legacy.checkpoint_controller import (
    RealtimeCheckpointController,
)
from pipelines.time_dynamics.runners.hh_from_adapt_artifact import (
    build_controller_config,
    build_drive_config,
    build_oracle_config,
)
from pipelines.time_dynamics.adapters.drive_terms import (
    resolve_realtime_drive_model,
)
from src.quantum.hubbard_latex_python_pairs import jw_number_operator
from src.quantum.hubbard_latex_python_pairs import build_hubbard_holstein_drive
from src.quantum.operator_pools.boson_chains import (
    build_harmonic_kerr_chain_hamiltonian,
    make_boson_chain_observables,
)
from pipelines.time_dynamics.runners.generic_from_adapt_artifact import (
    build_controller_seed_from_args,
    build_parser,
)
from src.quantum.operator_pools.spin_boson import (
    build_spin_boson_hamiltonian,
    make_spin_boson_observables,
)
from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix


def _spin_boson_fixture_path() -> Path:
    return REPO_ROOT / "test_support" / "fixtures" / "spin_boson_realtime_seed.json"


def _poly_coeff_map(poly, *, tol: float = 1.0e-12) -> dict[str, complex]:
    coeff_map: dict[str, complex] = {}
    for term in tuple(poly.return_polynomial()):
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tol):
            continue
        coeff_map[label] = coeff_map.get(label, 0.0 + 0.0j) + coeff
    return {
        str(label): complex(coeff)
        for label, coeff in coeff_map.items()
        if abs(complex(coeff)) > float(tol)
    }


def _driven_args(tmp_path: Path, *, drive_A: float = 0.6):
    return build_parser().parse_args(
        [
            "--artifact-json",
            str(_spin_boson_fixture_path()),
            "--output-json",
            str(tmp_path / "spin_boson_drive_terms.json"),
            "--checkpoint-controller-mode",
            "off",
            "--checkpoint-controller-reference-mode",
            "off",
            "--enable-drive",
            "--drive-A",
            str(drive_A),
            "--num-times",
            "5",
            "--t-final",
            "0.2",
        ]
    )


def _strict_driven_args(tmp_path: Path, *, drive_A: float = 0.6):
    return build_parser().parse_args(
        [
            "--artifact-json",
            str(_spin_boson_fixture_path()),
            "--output-json",
            str(tmp_path / "spin_boson_strict_drive_terms.json"),
            "--checkpoint-controller-strict-qpu-faithful",
            "--checkpoint-controller-mode",
            "oracle_v1",
            "--checkpoint-controller-reference-mode",
            "off",
            "--enable-drive",
            "--drive-A",
            str(drive_A),
            "--drive-omega",
            "1.7",
            "--drive-tbar",
            "1.0",
            "--drive-phi",
            "1.0",
            "--num-times",
            "5",
            "--t-final",
            "0.2",
        ]
    )


def test_spin_boson_drive_model_matches_static_dv_operator_and_observable(
    tmp_path: Path,
) -> None:
    seed = build_controller_seed_from_args(_driven_args(tmp_path))
    request = seed.runtime_input.resolved_problem.request
    drive_model = resolve_realtime_drive_model(
        resolved_problem=seed.runtime_input.resolved_problem,
        drive_config=seed.drive_config,
    )

    observable = make_spin_boson_observables(
        n_ph_max=int(request.n_ph_max),
        boson_encoding=str(request.boson_encoding),
        ordering=str(request.ordering),
    )["imbalance"]
    h_dv0 = build_spin_boson_hamiltonian(
        num_sites=int(request.num_sites),
        t=float(request.t),
        u=float(request.u),
        dv=0.0,
        omega0=float(request.omega0),
        g_ep=float(request.g_ep),
        n_ph_max=int(request.n_ph_max),
        boson_encoding=str(request.boson_encoding),
        ordering=str(request.ordering),
        include_zero_point=bool(request.include_zero_point),
    )
    h_dv1 = build_spin_boson_hamiltonian(
        num_sites=int(request.num_sites),
        t=float(request.t),
        u=float(request.u),
        dv=1.0,
        omega0=float(request.omega0),
        g_ep=float(request.g_ep),
        n_ph_max=int(request.n_ph_max),
        boson_encoding=str(request.boson_encoding),
        ordering=str(request.ordering),
        include_zero_point=bool(request.include_zero_point),
    )
    dv_operator = h_dv1 + ((-1.0) * h_dv0)
    dv_operator._reduce()

    assert drive_model.family_key == "spin_boson"
    assert drive_model.operator_label == "imbalance"
    assert _poly_coeff_map(drive_model.drive_poly) == _poly_coeff_map(observable)
    assert _poly_coeff_map(drive_model.drive_poly) == _poly_coeff_map(dv_operator)
    assert tuple(drive_model.spatial_weights) == pytest.approx((1.0,))
    assert int(drive_model.drive_term_count) > 0


def _adapter_drive_config(
    *,
    n_sites: int = 2,
    drive_A: float = 0.7,
    drive_pattern: str = "staggered",
    drive_custom_weights: tuple[float, ...] | None = None,
):
    return SimpleNamespace(
        enabled=True,
        n_sites=int(n_sites),
        ordering="blocked",
        drive_A=float(drive_A),
        drive_omega=1.3,
        drive_tbar=0.2,
        drive_phi=0.4,
        drive_pattern=str(drive_pattern),
        drive_custom_weights=drive_custom_weights,
        drive_include_identity=False,
        drive_time_sampling="midpoint",
        drive_t0=0.0,
    )


def _adapter_resolved_problem(family_key: str, *, n_sites: int = 2):
    return SimpleNamespace(
        family_key=str(family_key),
        request=SimpleNamespace(
            num_sites=int(n_sites),
            ordering="blocked",
            n_ph_max=1,
            boson_encoding="binary",
            t=1.0,
            u=0.5,
            dv=0.0,
            omega0=1.0,
            g_ep=0.25,
            boundary="open",
            include_zero_point=True,
        ),
    )


@pytest.mark.parametrize(
    ("family_key", "operator_label"),
    (
        ("hubbard", "spinful_onsite_density"),
        ("hh", "hh_spinful_onsite_density"),
        ("spinless_tv", "spinless_onsite_density"),
        ("bose_hubbard", "boson_onsite_number"),
        ("harmonic_kerr_chain", "harmonic_kerr_displacement"),
    ),
)
def test_adapter_drive_dispatch_builds_nonzero_models(
    family_key: str,
    operator_label: str,
) -> None:
    drive_model = resolve_realtime_drive_model(
        resolved_problem=_adapter_resolved_problem(family_key),
        drive_config=_adapter_drive_config(),
    )

    assert drive_model.family_key == family_key
    assert drive_model.operator_label == operator_label
    assert drive_model.drive_A == pytest.approx(0.7)
    assert int(drive_model.drive_term_count) > 0
    assert abs(float(drive_model.coefficient_at(0.2))) > 1.0e-12


@pytest.mark.parametrize("family_key", ["hh", "spinless_tv", "bose_hubbard", "harmonic_kerr_chain", "spin_boson"])
def test_adapter_drive_a0_has_zero_waveform_coefficient(family_key: str) -> None:
    resolved_problem = _adapter_resolved_problem(
        family_key,
        n_sites=(1 if family_key == "spin_boson" else 2),
    )
    drive_model = resolve_realtime_drive_model(
        resolved_problem=resolved_problem,
        drive_config=_adapter_drive_config(
            n_sites=(1 if family_key == "spin_boson" else 2),
            drive_A=0.0,
        ),
    )

    assert int(drive_model.drive_term_count) > 0
    for time_value in (0.0, 0.2, 1.0):
        assert float(drive_model.coefficient_at(time_value)) == pytest.approx(0.0, abs=1.0e-15)


def test_spinless_tv_drive_model_is_staggered_onsite_number() -> None:
    drive_model = resolve_realtime_drive_model(
        resolved_problem=_adapter_resolved_problem("spinless_tv"),
        drive_config=_adapter_drive_config(),
    )
    expected = jw_number_operator("JW", 2, 0) + ((-1.0) * jw_number_operator("JW", 2, 1))
    expected._reduce()

    assert drive_model.operator_label == "spinless_onsite_density"
    assert _poly_coeff_map(drive_model.drive_poly) == _poly_coeff_map(expected)


def test_bose_hubbard_drive_model_is_staggered_boson_number() -> None:
    drive_model = resolve_realtime_drive_model(
        resolved_problem=_adapter_resolved_problem("bose_hubbard"),
        drive_config=_adapter_drive_config(),
    )
    observables = make_boson_chain_observables(
        num_sites=2,
        n_ph_max=1,
        boson_encoding="binary",
    )
    expected = observables["n_site_0"] + ((-1.0) * observables["n_site_1"])
    expected._reduce()

    assert drive_model.operator_label == "boson_onsite_number"
    assert _poly_coeff_map(drive_model.drive_poly) == _poly_coeff_map(expected)


def test_harmonic_kerr_drive_model_matches_static_displacement_drive() -> None:
    drive_model = resolve_realtime_drive_model(
        resolved_problem=_adapter_resolved_problem("harmonic_kerr_chain"),
        drive_config=_adapter_drive_config(
            drive_pattern="custom",
            drive_custom_weights=(1.0, 1.0),
        ),
    )
    h_dv0 = build_harmonic_kerr_chain_hamiltonian(
        num_sites=2,
        t=1.0,
        u=0.5,
        dv=0.0,
        omega0=1.0,
        n_ph_max=1,
        boson_encoding="binary",
        boundary="open",
        include_zero_point=True,
    )
    h_dv1 = build_harmonic_kerr_chain_hamiltonian(
        num_sites=2,
        t=1.0,
        u=0.5,
        dv=1.0,
        omega0=1.0,
        n_ph_max=1,
        boson_encoding="binary",
        boundary="open",
        include_zero_point=True,
    )
    static_dv_operator = h_dv1 + ((-1.0) * h_dv0)
    static_dv_operator._reduce()

    assert drive_model.operator_label == "harmonic_kerr_displacement"
    assert tuple(drive_model.spatial_weights) == pytest.approx((1.0, 1.0))
    assert _poly_coeff_map(drive_model.drive_poly) == _poly_coeff_map(static_dv_operator)


def test_hh_drive_model_is_lifted_staggered_spinful_density() -> None:
    drive_model = resolve_realtime_drive_model(
        resolved_problem=_adapter_resolved_problem("hh"),
        drive_config=_adapter_drive_config(),
    )
    expected = build_hubbard_holstein_drive(
        dims=2,
        v_t=(1.0, -1.0),
        v0=(0.0, 0.0),
        repr_mode="JW",
        indexing="blocked",
        nq_override=6,
    )
    expected._reduce()

    assert drive_model.family_key == "hh"
    assert drive_model.operator_label == "hh_spinful_onsite_density"
    assert tuple(drive_model.spatial_weights) == pytest.approx((1.0, -1.0))
    assert _poly_coeff_map(drive_model.drive_poly) == _poly_coeff_map(expected)


def test_spin_boson_driven_run_emits_nondecision_drive_and_motion_telemetry(
    tmp_path: Path,
) -> None:
    args = _driven_args(tmp_path, drive_A=0.6)
    bundle = build_controller_seed_from_args(args)
    from pipelines.time_dynamics.runners.generic_from_adapt_artifact import (
        finalize_controller_bundle_from_seed,
    )

    controller = finalize_controller_bundle_from_seed(args, seed=bundle)["controller"]
    result = controller.run()

    assert result.trajectory
    assert result.ledger
    row = next(item for item in result.trajectory if int(item["drive_term_count"]) > 0)
    ledger = next(item for item in result.ledger if int(item["drive_term_count"]) > 0)
    assert row["drive_enabled"] is True
    assert row["drive_operator_label"] == "imbalance"
    assert row["drive_family_key"] == "spin_boson"
    assert row["drive_coefficient"] is not None
    assert row["theta_dot_l2"] >= 0.0
    assert row["theta_update_l2"] >= 0.0
    assert row["runtime_parameter_count_before"] == row["runtime_parameter_count"]
    assert row["runtime_parameter_count_after"] >= row["runtime_parameter_count_before"]
    assert ledger["drive_operator_label"] == "imbalance"
    assert ledger["theta_dot_l2"] >= 0.0
    assert ledger["observable_family"] == "spin_boson"
    assert ledger["primary_density_mode"] == "imbalance"


def test_spin_boson_step_hamiltonian_matches_static_plus_drive_term(
    tmp_path: Path,
) -> None:
    args = _driven_args(tmp_path, drive_A=0.6)
    bundle = build_controller_seed_from_args(args)
    from pipelines.time_dynamics.runners.generic_from_adapt_artifact import (
        finalize_controller_bundle_from_seed,
    )

    controller = finalize_controller_bundle_from_seed(args, seed=bundle)["controller"]
    drive_model = getattr(controller, "_drive_model", None)

    assert drive_model is not None

    for time_value in (0.0, 0.1, 0.2):
        step = controller._step_hamiltonian_artifacts(float(time_value))
        physical_time = controller._physical_time(float(time_value))
        drive_coeff = float(drive_model.coefficient_at(float(physical_time)))
        expected_poly = controller.h_poly
        expected_drive_terms = 0
        if abs(drive_coeff) > 1.0e-15:
            expected_poly = controller.h_poly + (drive_coeff * drive_model.drive_poly)
            expected_poly._reduce()
            expected_drive_terms = int(drive_model.drive_term_count)
        expected_hmat = np.asarray(hamiltonian_matrix(expected_poly), dtype=complex)

        assert float(step.physical_time) == pytest.approx(float(physical_time))
        assert int(step.drive_term_count) == expected_drive_terms
        assert _poly_coeff_map(step.h_poly) == _poly_coeff_map(expected_poly)
        assert np.allclose(np.asarray(step.hmat, dtype=complex), expected_hmat)


def test_strict_spin_boson_drive_model_step_is_dense_matrix_free(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seed = build_controller_seed_from_args(_driven_args(tmp_path, drive_A=0.6))
    strict_args = _strict_driven_args(tmp_path, drive_A=0.6)
    cfg = build_controller_config(strict_args)
    oracle_cfg = build_oracle_config(strict_args)
    request = seed.runtime_input.resolved_problem.request
    strict_drive_config = build_drive_config(
        strict_args,
        n_sites=int(request.num_sites),
        ordering=str(request.ordering),
    )

    def _raise_dense_hamiltonian(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("strict RealtimeDriveModel path materialized a dense Hamiltonian")

    monkeypatch.setattr(controller_mod, "hamiltonian_matrix", _raise_dense_hamiltonian)
    controller = RealtimeCheckpointController(
        cfg=cfg,
        replay_context=seed.replay_context,
        h_poly=seed.h_poly,
        hmat=None,
        psi_initial=np.asarray(seed.runtime_input.psi_initial, dtype=complex),
        best_theta=np.asarray(seed.runtime_input.theta_runtime, dtype=float),
        allow_repeats=bool(strict_args.allow_repeats),
        t_final=float(strict_args.t_final),
        num_times=int(strict_args.num_times),
        drive_config=strict_drive_config,
        oracle_base_config=oracle_cfg,
        resolved_problem=seed.runtime_input.resolved_problem,
        strict_qpu_faithful=True,
        strict_qpu_hh=False,
    )

    drive_model = getattr(controller, "_drive_model", None)
    assert controller.strict_qpu_faithful is True
    assert controller.strict_qpu_hh is False
    assert controller.hmat is None
    assert drive_model is not None

    sample_times = (0.0, 0.05, 0.1, 0.15, 0.2)
    step = None
    for time_value in sample_times:
        if abs(float(drive_model.coefficient_at(controller._physical_time(float(time_value))))) > 1.0e-15:
            step = controller._step_hamiltonian_artifacts(float(time_value))
            break
    assert step is not None
    assert step.hmat.shape == (0, 0)
    assert step.oracle_observable is not None
    assert int(step.drive_term_count) == int(drive_model.drive_term_count)
    assert int(step.drive_term_count) > 0


def test_spin_boson_is_hamiltonian_flow_projective_family() -> None:
    from pipelines.time_dynamics.adapters.hamiltonian import (
        HAMILTONIAN_FLOW_FAMILIES,
        family_supports_hamiltonian_flow_projective,
    )

    assert "spin_boson" in HAMILTONIAN_FLOW_FAMILIES
    assert family_supports_hamiltonian_flow_projective("spin_boson") is True


def test_molecular_vibronic_h2_is_hamiltonian_flow_projective_family() -> None:
    from pipelines.time_dynamics.adapters.hamiltonian import (
        HAMILTONIAN_FLOW_FAMILIES,
        family_supports_hamiltonian_flow_projective,
    )

    assert "molecular_vibronic_h2" in HAMILTONIAN_FLOW_FAMILIES
    assert family_supports_hamiltonian_flow_projective("molecular_vibronic_h2") is True
