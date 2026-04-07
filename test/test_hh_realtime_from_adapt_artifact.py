from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_realtime_checkpoint_controller import (
    ControllerDriveConfig,
    RealtimeCheckpointController,
)
from pipelines.hardcoded.hh_realtime_checkpoint_types import RealtimeCheckpointConfig
from pipelines.hardcoded.hh_realtime_from_adapt_artifact import main as realtime_main
from pipelines.hardcoded.hh_vqe_from_adapt_family import ReplayScaffoldContext
from src.quantum.ansatz_parameterization import build_parameter_layout
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, PauliPolynomial, PauliTerm, hamiltonian_matrix


def _two_qubit_drive_context(
    theta_x: float = 0.2,
) -> tuple[ReplayScaffoldContext, PauliPolynomial, np.ndarray, np.ndarray]:
    x_term = AnsatzTerm(
        label="op_x0",
        polynomial=PauliPolynomial("JW", [PauliTerm(2, ps="ex", pc=1.0)]),
    )
    y_term = AnsatzTerm(
        label="op_y0",
        polynomial=PauliPolynomial("JW", [PauliTerm(2, ps="ey", pc=1.0)]),
    )
    h_poly = PauliPolynomial("JW", [PauliTerm(2, ps="ez", pc=1.0)])
    hmat = np.asarray(hamiltonian_matrix(h_poly), dtype=complex)
    psi_ref = np.zeros(4, dtype=complex)
    psi_ref[0] = 1.0
    base_layout = build_parameter_layout(
        [x_term],
        ignore_identity=True,
        coefficient_tolerance=1e-12,
        sort_terms=True,
    )
    executor = CompiledAnsatzExecutor(
        [x_term],
        parameterization_mode="per_pauli_term",
        parameterization_layout=base_layout,
    )
    best_theta = np.array([float(theta_x)], dtype=float)
    psi_initial = executor.prepare_state(best_theta, psi_ref)
    replay_context = ReplayScaffoldContext(
        cfg=SimpleNamespace(reps=1, L=1, ordering="blocked"),
        h_poly=h_poly,
        psi_ref=psi_ref,
        payload_in={"adapt_vqe": {"pool_type": "phase3_v1"}},
        family_info={"resolved": "toy_pool_drive"},
        family_pool=(x_term, y_term),
        pool_meta={"candidate_pool_complete": True},
        replay_terms=(x_term,),
        base_layout=base_layout,
        adapt_theta_runtime=np.array([float(theta_x)], dtype=float),
        adapt_theta_logical=np.array([float(theta_x)], dtype=float),
        adapt_depth=1,
        handoff_state_kind="prepared_state",
        provenance_source="explicit",
        family_terms_count=2,
    )
    return replay_context, h_poly, hmat, psi_initial


def test_hh_realtime_from_adapt_artifact_defaults_match_current_standalone_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _two_qubit_drive_context(theta_x=0.2)

    def _fake_load_run_context(spec, *, tag: str, lock_fixed_manifold: bool):
        del spec, tag, lock_fixed_manifold
        return SimpleNamespace(
            cfg=SimpleNamespace(L=2, ordering="blocked"),
            replay_context=replay_context,
            psi_initial=psi_initial,
        )

    monkeypatch.setattr(
        "pipelines.hardcoded.hh_realtime_from_adapt_artifact.load_run_context",
        _fake_load_run_context,
    )
    output_json = tmp_path / "step2.json"
    exit_code = realtime_main(
        [
            "--artifact-json",
            str(tmp_path / "artifact.json"),
            "--output-json",
            str(output_json),
            "--run-tag",
            "parity_test",
        ]
    )

    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(
            mode="exact_v1",
            miss_threshold=0.05,
            gain_ratio_threshold=0.02,
            append_margin_abs=1e-6,
            candidate_step_scales=(0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0),
            exact_forecast_baseline_blend_weights=(-0.25, -0.125, 0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0),
            exact_forecast_include_tangent_secant_proposal=True,
            exact_forecast_tangent_secant_trust_radius=0.75,
            exact_forecast_tangent_secant_signed_energy_lead_limit=1.0,
            exact_forecast_tracking_horizon_steps=2,
            exact_forecast_tracking_horizon_weights=(2.0, 1.0),
        ),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=np.asarray(replay_context.adapt_theta_runtime, dtype=float),
        allow_repeats=False,
        t_final=8.0,
        num_times=161,
        drive_config=ControllerDriveConfig(
            enabled=True,
            n_sites=2,
            ordering="blocked",
            drive_A=1.5,
            drive_omega=1.2,
            drive_tbar=4.0,
            drive_phi=0.0,
            drive_pattern="staggered",
            drive_custom_weights=None,
            drive_include_identity=False,
            drive_time_sampling="midpoint",
            drive_t0=0.0,
            exact_steps_multiplier=4,
        ),
    )
    direct = controller.run()
    payload = json.loads(output_json.read_text(encoding="utf-8"))

    assert int(exit_code) == 0
    assert payload["controller_config"]["exact_forecast_tangent_secant_trust_radius"] == pytest.approx(0.75)
    assert payload["controller_config"]["exact_forecast_tracking_horizon_weights"] == pytest.approx([2.0, 1.0])
    assert payload["summary"]["mode"] == direct.summary["mode"]
    assert payload["summary"]["exact_forecast_tangent_secant_signed_energy_lead_limit"] == pytest.approx(
        float(direct.summary["exact_forecast_tangent_secant_signed_energy_lead_limit"])
    )
    assert len(payload["trajectory"]) == len(direct.trajectory)
    assert payload["trajectory"][0]["time_start"] == pytest.approx(float(direct.trajectory[0]["time_start"]))
    assert payload["trajectory"][0]["time_stop"] == pytest.approx(float(direct.trajectory[0]["time_stop"]))
    assert payload["trajectory"][0]["tracking_score_horizon"] == pytest.approx(
        float(direct.trajectory[0]["tracking_score_horizon"])
    )


def test_hh_realtime_from_adapt_artifact_serializes_nondefault_controller_weights(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    replay_context, _h_poly, _hmat, psi_initial = _two_qubit_drive_context(theta_x=0.2)

    def _fake_load_run_context(spec, *, tag: str, lock_fixed_manifold: bool):
        del spec, tag, lock_fixed_manifold
        return SimpleNamespace(
            cfg=SimpleNamespace(L=2, ordering="blocked"),
            replay_context=replay_context,
            psi_initial=psi_initial,
        )

    monkeypatch.setattr(
        "pipelines.hardcoded.hh_realtime_from_adapt_artifact.load_run_context",
        _fake_load_run_context,
    )
    output_json = tmp_path / "weights.json"
    realtime_main(
        [
            "--artifact-json",
            str(tmp_path / "artifact.json"),
            "--output-json",
            str(output_json),
            "--disable-drive",
            "--checkpoint-controller-exact-forecast-tangent-secant-trust-radius",
            "0.5",
            "--checkpoint-controller-exact-forecast-horizon-weights",
            "3.0,1.5",
            "--checkpoint-controller-exact-forecast-energy-slope-weight",
            "50.0",
            "--checkpoint-controller-exact-forecast-tracking-doublon-error-weight",
            "4.0",
            "--checkpoint-controller-exact-forecast-tracking-site-occupations-error-weight",
            "2.5",
        ]
    )
    payload = json.loads(output_json.read_text(encoding="utf-8"))

    assert payload["drive_config"] is None
    assert payload["controller_config"]["exact_forecast_tangent_secant_trust_radius"] == pytest.approx(0.5)
    assert payload["controller_config"]["exact_forecast_tracking_horizon_weights"] == pytest.approx([3.0, 1.5])
    assert payload["controller_config"]["exact_forecast_energy_slope_weight"] == pytest.approx(50.0)
    assert payload["controller_config"]["exact_forecast_tracking_doublon_error_weight"] == pytest.approx(4.0)
    assert payload["controller_config"]["exact_forecast_tracking_site_occupations_error_weight"] == pytest.approx(2.5)
