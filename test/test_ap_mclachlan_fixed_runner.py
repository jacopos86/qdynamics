from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.contracts.scaffold import ScaffoldRuntimeInput
from pipelines.time_dynamics.runners.ap_fixed_from_adapt_artifact import (
    RUNNER_SCHEMA_V1,
    run_fixed_ap_mclachlan_from_artifact,
    run_fixed_ap_mclachlan_from_runtime_input,
)
from src.quantum.ansatz_parameterization import build_parameter_layout
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _poly(label: str, coeff: float = 1.0) -> PauliPolynomial:
    poly = PauliPolynomial("JW")
    poly.add_term(PauliTerm(1, ps=str(label), pc=float(coeff)))
    poly._reduce()
    return poly


def _multi_poly(labels: tuple[tuple[str, float], ...]) -> PauliPolynomial:
    poly = PauliPolynomial("JW")
    for label, coeff in labels:
        poly.add_term(PauliTerm(1, ps=str(label), pc=float(coeff)))
    poly._reduce()
    return poly


def _runtime_input() -> ScaffoldRuntimeInput:
    selected = (AnsatzTerm(label="seed_x", polynomial=_poly("x")),)
    layout = build_parameter_layout(selected)
    return ScaffoldRuntimeInput(
        resolved_problem=SimpleNamespace(family_key="toy", hamiltonian=_poly("x")),
        psi_ref=np.array([1.0, 0.0], dtype=complex),
        psi_initial=np.array([1.0, 0.0], dtype=complex),
        base_layout=layout,
        theta_runtime=np.array([0.0], dtype=float),
        theta_logical=np.array([0.0], dtype=float),
        structure_locked=False,
        exact_energy=None,
        selected_terms=selected,
        candidate_pool_terms=selected,
        provenance={"artifact_json": "toy.json"},
    )


def test_fixed_runner_payload_is_plot_ready_and_decision_clean() -> None:
    payload = run_fixed_ap_mclachlan_from_runtime_input(
        _runtime_input(),
        times=(0.0, 0.1, 0.2),
    )

    assert payload["schema"] == RUNNER_SCHEMA_V1
    assert payload["summary"]["point_count"] == 3
    assert payload["summary"]["integrator_method"] == "euler"
    assert payload["summary"]["energy_initial"] == pytest.approx(0.0)
    assert payload["summary"]["energy_final"] == pytest.approx(0.0)
    assert payload["summary"]["runtime_parameter_count"] == 1
    assert payload["plot_rows"][0]["time"] == pytest.approx(0.0)
    assert payload["plot_rows"][-1]["time"] == pytest.approx(0.2)
    assert payload["plot_rows"][0]["theta_dot_l2"] == pytest.approx(1.0)
    assert payload["trajectory"]["metadata"]["uses_reference_for_decision"] is False
    assert payload["decision_data_flow"]["uses_future_exact_forecast_for_decision"] is False


def test_fixed_runner_exposes_logical_shared_parameterization() -> None:
    selected = (
        AnsatzTerm(
            label="seed_xz",
            polynomial=_multi_poly((("x", 1.0), ("z", 0.5))),
        ),
    )
    layout = build_parameter_layout(selected)
    runtime_input = ScaffoldRuntimeInput(
        resolved_problem=SimpleNamespace(family_key="toy", hamiltonian=_poly("x")),
        psi_ref=np.array([1.0, 0.0], dtype=complex),
        psi_initial=np.array([1.0, 0.0], dtype=complex),
        base_layout=layout,
        theta_runtime=np.array([0.0, 0.0], dtype=float),
        theta_logical=np.array([0.0], dtype=float),
        structure_locked=False,
        exact_energy=None,
        selected_terms=selected,
        candidate_pool_terms=selected,
        provenance={"artifact_json": "toy.json"},
    )

    payload = run_fixed_ap_mclachlan_from_runtime_input(
        runtime_input,
        times=(0.0, 0.1),
        parameterization_mode="logical_shared",
    )

    assert payload["state"]["parameterization_mode"] == "logical_shared"
    assert payload["state"]["parameterization_label"] == "per logical / macro generator"
    assert payload["summary"]["parameterization_mode"] == "logical_shared"
    assert payload["summary"]["active_parameter_count"] == 1
    assert payload["summary"]["runtime_parameter_count"] == 1
    assert payload["summary"]["runtime_pauli_parameter_count"] == 2
    assert payload["summary"]["logical_parameter_count"] == 1


def test_fixed_runner_requires_drive_config_when_drive_enabled() -> None:
    with pytest.raises(ValueError, match="requires a drive_config"):
        run_fixed_ap_mclachlan_from_runtime_input(
            _runtime_input(),
            times=(0.0,),
            enable_drive=True,
        )


def test_fixed_runner_rejects_promotion_wrapper_json(tmp_path) -> None:
    wrapper = tmp_path / "promotion_wrapper.json"
    wrapper.write_text('{"schema":"paper_i_table_wrapper_v1"}', encoding="utf-8")

    with pytest.raises(ValueError, match="raw scaffold artifact JSON"):
        run_fixed_ap_mclachlan_from_artifact(
            artifact_json=wrapper,
            times=(0.0,),
        )


def test_fixed_runner_accepts_generic_static_comparator_wrapper(tmp_path) -> None:
    wrapper = tmp_path / "generic_static_single.json"
    wrapper.write_text(
        json.dumps(
            {
                "schema": "generic_static_adapt_variant_single_v1",
                "family": "hh",
                "runtime_seed_schema": "paper_ii_static_seed_runtime_payload_v1",
                "guardrails": {"pool_name": "full_meta"},
                "spec": {
                    "base_pipeline_args": [
                        "--problem",
                        "hh",
                        "--L",
                        "1",
                        "--t",
                        "1.0",
                        "--u",
                        "0.25",
                        "--dv",
                        "0.0",
                        "--omega0",
                        "1.0",
                        "--g-ep",
                        "0.1",
                        "--n-ph-max",
                        "1",
                        "--boson-encoding",
                        "binary",
                        "--ordering",
                        "blocked",
                        "--boundary",
                        "open",
                    ]
                },
                "result": {
                    "selected_operators": ["hh_phonon::x(site=0)"],
                    "theta": [0.01],
                    "same_cutoff_exact_gs_energy": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )

    payload = run_fixed_ap_mclachlan_from_artifact(
        artifact_json=wrapper,
        times=(0.0,),
        integrator_method="euler",
    )

    assert payload["schema"] == RUNNER_SCHEMA_V1
    assert payload["summary"]["runtime_parameter_count"] == 1
    assert payload["summary"]["logical_parameter_count"] == 1
    assert (
        payload["state"]["extensions"]["scaffold_runtime_normalization"]["source"]
        == "generic_static_comparator_wrapper"
    )
    assert (
        payload["state"]["extensions"]["legacy_loader_summary"][
            "prepared_state_reconstruction_error"
        ]
        == pytest.approx(0.0)
    )
