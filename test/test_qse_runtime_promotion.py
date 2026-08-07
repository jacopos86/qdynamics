from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.qse_spectra.io import statevector_from_manifest
from pipelines.scaffold.qse_root_refit import QSERootRefitConfig, run_qse_root_refit
from pipelines.scaffold.qse_runtime_promotion import (
    QSERuntimePromotionConfig,
    QSERuntimePromotionError,
    main as qse_runtime_promotion_main,
    promote_qse_root_refit,
    reconstruct_promoted_ansatz_state_from_payload,
)


def _state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=complex).reshape(-1)
    bb = np.asarray(b, dtype=complex).reshape(-1)
    aa = aa / np.linalg.norm(aa)
    bb = bb / np.linalg.norm(bb)
    return float(abs(np.vdot(aa, bb)) ** 2)


def _write_minus_z_hamiltonian(path: Path) -> Path:
    path.write_text(
        json.dumps({"terms": [{"pauli_exyz": "z", "coeff_re": -1.0, "coeff_im": 0.0}]}),
        encoding="utf-8",
    )
    return path


def _generate_one_qubit_refit(tmp_path: Path) -> Path:
    from pipelines.qse_spectra.__main__ import main as qse_main

    ham_path = _write_minus_z_hamiltonian(tmp_path / "minus_z_ham.json")
    qse_path = tmp_path / "qse_source.json"
    rc = qse_main(
        [
            "--hamiltonian-json",
            str(ham_path),
            "--state-bitstring",
            "0",
            "--operator-basis-label",
            "I",
            "--operator-basis-label",
            "X",
            "--output-json",
            str(qse_path),
            "--omit-matrices",
        ]
    )
    assert rc == 0
    out_path = tmp_path / "qse_root_refit.json"
    run_qse_root_refit(
        QSERootRefitConfig(
            qse_result_json=qse_path,
            state_index=1,
            output_json=out_path,
            hamiltonian_json=ham_path,
            max_infidelity=1.0e-10,
            max_energy_error=1.0e-10,
            maxiter=0,
        )
    )
    return out_path


def test_one_qubit_p5a_promotes_to_sanitized_non_controller_artifact(tmp_path: Path) -> None:
    source_path = _generate_one_qubit_refit(tmp_path)
    out_path = tmp_path / "qse_runtime_promoted_ansatz.json"

    artifact = promote_qse_root_refit(
        QSERuntimePromotionConfig(
            qse_root_refit_json=source_path,
            output_json=out_path,
        )
    )

    assert out_path.exists()
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == "qse_runtime_promoted_ansatz_v1"
    assert artifact["runtime_contract"]["status"] == "not_representable"
    assert artifact["controller_boundary"]["controller_usable"] is False
    assert artifact["controller_boundary"]["matches_scaffold_runtime_contract"] is False
    assert artifact["controller_boundary"]["realtime_wiring"] is False
    assert artifact["runtime_payload"] is None
    assert artifact["visibility"]["controller_visible_payload_refs"] == []

    ansatz = artifact["sanitized_ansatz"]
    assert ansatz["runtime_parameter_count"] == 1
    assert ansatz["logical_operator_count"] == 1
    assert ansatz["operators"] == ["promoted_generator_0"]
    assert ansatz["generator_terms"][0]["label"] == "promoted_generator_0"
    assert ansatz["parameterization"]["blocks"][0]["candidate_label"] == "promoted_generator_0"

    sanitized_json = json.dumps(ansatz, sort_keys=True)
    assert "qse_ritz_diagnostics" not in sanitized_json
    assert "basis_coefficients" not in sanitized_json
    assert "target_state_diagnostics" not in sanitized_json
    assert "fit_summary" not in sanitized_json
    assert "qse_basis" not in sanitized_json

    initial_state, _ = statevector_from_manifest(ansatz["initial_state"], state_key="auto")
    replayed = reconstruct_promoted_ansatz_state_from_payload(artifact)
    expected_one = np.asarray([0.0, 1.0], dtype=complex)
    assert _state_fidelity(initial_state, expected_one) == pytest.approx(1.0, abs=1.0e-12)
    assert _state_fidelity(replayed, expected_one) == pytest.approx(1.0, abs=1.0e-12)
    assert ansatz["prepared_state_replay_error"] <= 1.0e-12


def test_require_runtime_contract_fails_for_no_template_one_qubit_source(tmp_path: Path) -> None:
    source_path = _generate_one_qubit_refit(tmp_path)

    with pytest.raises(QSERuntimePromotionError, match="runtime contract validation required"):
        promote_qse_root_refit(
            QSERuntimePromotionConfig(
                qse_root_refit_json=source_path,
                output_json=tmp_path / "required.json",
                require_runtime_contract=True,
            )
        )
    assert not (tmp_path / "required.json").exists()


def _tampered_source_path(
    tmp_path: Path,
    source_path: Path,
    mutate: Callable[[dict], None],
) -> Path:
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    mutate(payload)
    bad_path = tmp_path / f"tampered_{abs(hash(json.dumps(payload, sort_keys=True, default=str)))}.json"
    bad_path.write_text(json.dumps(payload), encoding="utf-8")
    return bad_path


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.__setitem__("schema_version", "not_qse_root_refit_v1"), "schema_version"),
        (lambda payload: payload.pop("ansatz_payload"), "ansatz_payload"),
        (
            lambda payload: payload["ansatz_payload"].__setitem__("qpu_preparable_in_principle", False),
            "qpu_preparable_in_principle",
        ),
        (
            lambda payload: payload["fit_summary"]["passes"].__setitem__("all_thresholds", False),
            "all_thresholds",
        ),
        (lambda payload: payload["ansatz_payload"]["theta_runtime"].append(0.0), "theta_runtime length"),
        (lambda payload: payload["ansatz_payload"]["theta_runtime"].__setitem__(0, float("nan")), "finite"),
        (
            lambda payload: payload["visibility"].__setitem__(
                "controller_visible_payload_refs",
                ["qse_ritz_diagnostics.basis_coefficients"],
            ),
            "controller-visible refs",
        ),
    ],
)
def test_tampered_sources_fail_closed(
    tmp_path: Path,
    mutate: Callable[[dict], None],
    message: str,
) -> None:
    source_path = _generate_one_qubit_refit(tmp_path)
    bad_path = _tampered_source_path(tmp_path, source_path, mutate)

    with pytest.raises(QSERuntimePromotionError, match=message):
        promote_qse_root_refit(
            QSERuntimePromotionConfig(
                qse_root_refit_json=bad_path,
                output_json=tmp_path / "bad_promoted.json",
            )
        )


def _spin_boson_fixture_path() -> Path:
    return REPO_ROOT / "test_support" / "fixtures" / "spin_boson_realtime_seed.json"


def _generator_terms_from_parameterization(parameterization: dict) -> list[dict]:
    out: list[dict] = []
    for block in parameterization["blocks"]:
        out.append(
            {
                "logical_index": int(block["logical_index"]),
                "label": str(block["candidate_label"]),
                "execution_mode": "termwise_product",
                "terms": [dict(term) for term in block["runtime_terms_exyz"]],
            }
        )
    return out


def _synthetic_spin_boson_qse_root_refit(tmp_path: Path) -> Path:
    fixture = json.loads(_spin_boson_fixture_path().read_text(encoding="utf-8"))
    adapt = fixture["adapt_vqe"]
    payload = {
        "schema_version": "qse_root_refit_v1",
        "pipeline": "qse_root_refit",
        "generated_utc": "2026-05-16T00:00:00Z",
        "backend": "offline_statevector",
        "uses_qiskit": False,
        "controller_boundary": {
            "controller_usable": False,
            "feeds_controller_decisions": False,
            "decision_path_allowed": False,
            "realtime_wiring": False,
            "ansatz_payload_potentially_promotable": True,
            "promotion_requires_runtime_contract_validation": True,
            "matches_scaffold_runtime_contract": False,
            "qse_coefficients_forbidden_to_controller": True,
            "target_state_diagnostics_forbidden_to_controller": True,
        },
        "qse_ritz_diagnostics": {
            "state_index": 1,
            "basis_coefficients": [{"basis_index": 0, "re": 1.0, "im": 0.0}],
            "forbidden_to_controller": True,
        },
        "target_state_diagnostics": {
            "forbidden_to_controller": True,
            "amplitudes_qn_to_q0": {"poison_if_copied": {"re": 1.0, "im": 0.0}},
        },
        "ansatz_payload": {
            "ansatz_schema": "pauli_rotation_ansatz_v1",
            "parameterization_mode": "per_pauli_term",
            "operator_basis_source": "synthetic_spin_boson_fixture",
            "selected_operator_labels": list(adapt["operators"]),
            "generator_terms": _generator_terms_from_parameterization(adapt["parameterization"]),
            "parameterization": copy.deepcopy(adapt["parameterization"]),
            "theta_runtime": list(adapt["optimal_point"]),
            "theta_logical": list(adapt["logical_optimal_point"]),
            "reference_state": copy.deepcopy(fixture["ansatz_input_state"]),
            "prepared_state": copy.deepcopy(fixture["ansatz_input_state"]),
            "qpu_preparable_in_principle": True,
            "matches_scaffold_runtime_contract": False,
            "promotion_status": "candidate_passed_thresholds",
        },
        "fit_summary": {
            "passes": {"all_thresholds": True},
            "fidelity": 1.0,
            "infidelity": 0.0,
        },
        "visibility": {
            "controller_visible_payload_refs": [],
            "potentially_promotable_payload_refs": ["ansatz_payload"],
            "diagnostic_only_payload_refs": ["qse_ritz_diagnostics", "target_state_diagnostics", "fit_summary"],
            "forbidden_to_controller_refs": [
                "qse_ritz_diagnostics.basis_coefficients",
                "target_state_diagnostics.amplitudes_qn_to_q0",
            ],
        },
    }
    out_path = tmp_path / "synthetic_spin_boson_qse_root_refit.json"
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    return out_path


def _synthetic_hh_runtime_template(tmp_path: Path) -> Path:
    payload = {
        "settings": {
            "problem": "hh",
            "L": 1,
            "t": 1.0,
            "u": 0.25,
            "dv": 0.0,
            "omega0": 1.0,
            "g_ep": 0.25,
            "n_ph_max": 1,
            "boson_encoding": "binary",
            "ordering": "blocked",
            "boundary": "open",
            "include_zero_point": True,
            "sector_n_up": 1,
            "sector_n_dn": 0,
            "adapt_pool": "full_meta",
        },
        "adapt_vqe": {"pool_type": "full_meta"},
    }
    path = tmp_path / "synthetic_hh_runtime_template.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _synthetic_hh_qse_root_refit(tmp_path: Path) -> Path:
    pauli_term = {
        "pauli_exyz": "zee",
        "coeff_re": 1.0,
        "coeff_im": 0.0,
        "nq": 3,
    }
    parameterization = {
        "mode": "per_pauli_term_v1",
        "term_order": "sorted",
        "ignore_identity": True,
        "coefficient_tolerance": 1.0e-12,
        "logical_operator_count": 1,
        "runtime_parameter_count": 1,
        "blocks": [
            {
                "candidate_label": "source_hh_boson_z",
                "logical_index": 0,
                "runtime_start": 0,
                "runtime_count": 1,
                "runtime_terms_exyz": [dict(pauli_term)],
            }
        ],
    }
    reference_state = {
        "source": "synthetic_hh_hf",
        "nq_total": 3,
        "amplitudes_qn_to_q0": {"001": {"re": 1.0, "im": 0.0}},
        "handoff_state_kind": "reference_state",
    }
    payload = {
        "schema_version": "qse_root_refit_v1",
        "pipeline": "qse_root_refit",
        "generated_utc": "2026-08-02T00:00:00Z",
        "backend": "offline_statevector",
        "uses_qiskit": False,
        "controller_boundary": {
            "controller_usable": False,
            "ansatz_payload_potentially_promotable": True,
        },
        "qse_ritz_diagnostics": {"state_index": 0, "forbidden_to_controller": True},
        "ansatz_payload": {
            "ansatz_schema": "pauli_rotation_ansatz_v1",
            "parameterization_mode": "per_pauli_term",
            "operator_basis_source": "synthetic_hh_compact_refit",
            "selected_operator_labels": ["source_hh_boson_z"],
            "generator_terms": [
                {
                    "logical_index": 0,
                    "label": "source_hh_boson_z",
                    "execution_mode": "termwise_product",
                    "terms": [dict(pauli_term)],
                }
            ],
            "parameterization": parameterization,
            "theta_runtime": [0.0],
            "theta_logical": [0.0],
            "reference_state": copy.deepcopy(reference_state),
            "prepared_state": {
                **copy.deepcopy(reference_state),
                "handoff_state_kind": "prepared_state",
            },
            "qpu_preparable_in_principle": True,
            "matches_scaffold_runtime_contract": False,
            "promotion_status": "candidate_passed_thresholds",
        },
        "fit_summary": {
            "passes": {"all_thresholds": True},
            "fidelity": 1.0,
            "infidelity": 0.0,
        },
        "visibility": {
            "controller_visible_payload_refs": [],
            "potentially_promotable_payload_refs": ["ansatz_payload"],
            "diagnostic_only_payload_refs": ["qse_ritz_diagnostics", "fit_summary"],
            "forbidden_to_controller_refs": ["qse_ritz_diagnostics"],
        },
    }
    path = tmp_path / "synthetic_hh_qse_root_refit.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_positive_hh_runtime_loader_validation_uses_locked_fixed_scaffold(tmp_path: Path) -> None:
    source_path = _synthetic_hh_qse_root_refit(tmp_path)
    template_path = _synthetic_hh_runtime_template(tmp_path)
    output_path = tmp_path / "synthetic_hh_promoted.json"

    artifact = promote_qse_root_refit(
        QSERuntimePromotionConfig(
            qse_root_refit_json=source_path,
            output_json=output_path,
            runtime_template_json=template_path,
            require_runtime_contract=True,
        )
    )

    assert artifact["runtime_contract"]["status"] == "validated"
    assert artifact["runtime_contract"]["loader_mode"] == "fixed_scaffold"
    assert artifact["runtime_contract"]["problem_key"] == "hh"
    assert artifact["runtime_contract"]["prepared_state_reconstruction_error"] <= 1.0e-10
    runtime_payload = artifact["runtime_payload"]
    assert runtime_payload["settings"]["adapt_pool"] == "fixed_scaffold_locked"
    adapt_vqe = runtime_payload["adapt_vqe"]
    assert adapt_vqe["pool_type"] == "fixed_scaffold_locked"
    assert adapt_vqe["structure_locked"] is True
    fixed = adapt_vqe["fixed_scaffold_metadata"]
    assert fixed["route_family"] == "locked_imported_scaffold_v1"
    assert fixed["subject_kind"] == "qse_excited_state_refit_v1"
    assert fixed["operator_count"] == 1
    assert fixed["runtime_term_count"] == 1
    assert fixed["source_order_runtime_indices"] == [0]
    assert fixed["runtime_term_labels_exyz"] == ["zee"]
    assert fixed["source_pool_type"] == "full_meta"

    from pipelines.scaffold.runtime_loader import load_scaffold_runtime_input_from_payload

    runtime_input = load_scaffold_runtime_input_from_payload(
        runtime_payload,
        artifact_json=output_path,
    )
    assert runtime_input.provenance["loader_mode"] == "fixed_scaffold"
    assert runtime_input.structure_locked is True
    assert runtime_input.can_structural_edit is False


def test_forbidden_nested_ansatz_fields_are_scrubbed_from_validated_payload(tmp_path: Path) -> None:
    source_path = _synthetic_spin_boson_qse_root_refit(tmp_path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    payload["ansatz_payload"]["parameterization"]["blocks"][0]["basis_coefficients"] = [
        {"basis_index": 0, "re": 1.0, "im": 0.0}
    ]
    payload["ansatz_payload"]["parameterization"]["blocks"][0]["runtime_terms_exyz"][0][
        "target_state_diagnostics"
    ] = {"poison_if_copied": True}
    payload["ansatz_payload"]["generator_terms"][0]["terms"][0]["fit_summary"] = {
        "poison_if_copied": True
    }
    payload["ansatz_payload"]["generator_terms"][0]["qse_ritz_diagnostics"] = {
        "poison_if_copied": True
    }
    poisoned_path = tmp_path / "poisoned_spin_boson_qse_root_refit.json"
    poisoned_path.write_text(json.dumps(payload), encoding="utf-8")

    artifact = promote_qse_root_refit(
        QSERuntimePromotionConfig(
            qse_root_refit_json=poisoned_path,
            output_json=tmp_path / "poisoned_spin_boson_promoted.json",
            runtime_template_json=_spin_boson_fixture_path(),
            require_runtime_contract=True,
        )
    )

    assert artifact["runtime_contract"]["status"] == "validated"
    controller_payload_json = json.dumps(
        {
            "sanitized_ansatz": artifact["sanitized_ansatz"],
            "runtime_payload": artifact["runtime_payload"],
        },
        sort_keys=True,
    )
    assert "basis_coefficients" not in controller_payload_json
    assert "target_state_diagnostics" not in controller_payload_json
    assert "fit_summary" not in controller_payload_json
    assert "qse_ritz_diagnostics" not in controller_payload_json
    assert "poison_if_copied" not in controller_payload_json


def test_runtime_template_nested_forbidden_settings_fail_closed(tmp_path: Path) -> None:
    source_path = _synthetic_spin_boson_qse_root_refit(tmp_path)
    template = json.loads(_spin_boson_fixture_path().read_text(encoding="utf-8"))
    template["settings"]["molecular_problem_json"] = {
        "metadata": {"exact_gs_energy": -1.0, "poison_if_copied": True}
    }
    template_path = tmp_path / "poisoned_runtime_template.json"
    template_path.write_text(json.dumps(template), encoding="utf-8")
    out_path = tmp_path / "poisoned_template_promoted.json"

    with pytest.raises(QSERuntimePromotionError, match="forbidden marker"):
        promote_qse_root_refit(
            QSERuntimePromotionConfig(
                qse_root_refit_json=source_path,
                output_json=out_path,
                runtime_template_json=template_path,
                require_runtime_contract=True,
            )
        )
    assert not out_path.exists()


def test_positive_runtime_loader_validation_with_spin_boson_template(tmp_path: Path) -> None:
    source_path = _synthetic_spin_boson_qse_root_refit(tmp_path)
    fixture_path = _spin_boson_fixture_path()

    artifact = promote_qse_root_refit(
        QSERuntimePromotionConfig(
            qse_root_refit_json=source_path,
            output_json=tmp_path / "spin_boson_promoted.json",
            runtime_template_json=fixture_path,
            require_runtime_contract=True,
        )
    )

    assert artifact["runtime_contract"]["status"] == "validated"
    assert artifact["runtime_contract"]["problem_key"] == "spin_boson"
    assert artifact["runtime_contract"]["prepared_state_reconstruction_error"] <= 1.0e-10
    assert artifact["controller_boundary"]["controller_usable"] is True
    assert artifact["controller_boundary"]["matches_scaffold_runtime_contract"] is True
    assert artifact["visibility"]["controller_visible_payload_refs"] == ["runtime_payload"]

    runtime_payload = artifact["runtime_payload"]
    assert runtime_payload is not None
    runtime_json = json.dumps(runtime_payload, sort_keys=True)
    assert "poison_if_copied" not in runtime_json
    assert "basis_coefficients" not in runtime_json
    assert "qse_ritz_diagnostics" not in runtime_json
    assert "target_state_diagnostics" not in runtime_json
    assert "exact" not in runtime_payload
    assert "ground_state" not in runtime_payload
    assert "exact_gs_energy" not in runtime_payload["adapt_vqe"]

    from pipelines.scaffold.runtime_loader import load_scaffold_runtime_input_from_payload

    runtime_input = load_scaffold_runtime_input_from_payload(
        runtime_payload,
        artifact_json=tmp_path / "spin_boson_promoted.json",
    )
    assert runtime_input.resolved_problem.family_key == "spin_boson"
    assert runtime_input.structure_locked is True
    assert runtime_input.can_structural_edit is False
    assert runtime_input.exact_energy is None


def test_cli_runtime_template_validation_emits_controller_usable_payload(tmp_path: Path) -> None:
    source_path = _synthetic_spin_boson_qse_root_refit(tmp_path)
    out_path = tmp_path / "spin_boson_cli_promoted.json"

    rc = qse_runtime_promotion_main(
        [
            "--qse-root-refit-json",
            str(source_path),
            "--runtime-template-json",
            str(_spin_boson_fixture_path()),
            "--require-runtime-contract",
            "--output-json",
            str(out_path),
        ]
    )

    assert rc == 0
    assert out_path.exists()
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    assert artifact["schema_version"] == "qse_runtime_promoted_ansatz_v1"
    assert artifact["runtime_contract"]["status"] == "validated"
    assert artifact["runtime_contract"]["controller_usable"] is True
    assert artifact["runtime_contract"]["problem_key"] == "spin_boson"
    assert artifact["runtime_contract"]["prepared_state_reconstruction_error"] <= 1.0e-10
    assert artifact["controller_boundary"]["controller_usable"] is True
    assert artifact["controller_boundary"]["matches_scaffold_runtime_contract"] is True
    assert artifact["controller_boundary"]["feeds_controller_decisions"] is False
    assert artifact["controller_boundary"]["realtime_wiring"] is False
    assert artifact["controller_boundary"]["live_route_executed"] is False
    assert artifact["sanitized_ansatz"]["matches_scaffold_runtime_contract"] is True
    assert artifact["visibility"]["controller_visible_payload_refs"] == ["runtime_payload"]

    runtime_payload = artifact["runtime_payload"]
    assert runtime_payload is not None
    runtime_json = json.dumps(runtime_payload, sort_keys=True)
    for forbidden_marker in (
        "basis_coefficients",
        "qse_ritz_diagnostics",
        "target_state_diagnostics",
        "fit_summary",
        "poison_if_copied",
        "exact",
        "ground_state",
        "exact_gs_energy",
    ):
        assert forbidden_marker not in runtime_json
    assert runtime_payload["settings"]["problem"] == "spin_boson"
    assert "exact" not in runtime_payload
    assert "ground_state" not in runtime_payload
    assert "exact_gs_energy" not in runtime_payload["adapt_vqe"]

    from pipelines.scaffold.runtime_loader import load_scaffold_runtime_input_from_payload

    runtime_input = load_scaffold_runtime_input_from_payload(
        runtime_payload,
        artifact_json=out_path,
    )
    assert runtime_input.resolved_problem.family_key == "spin_boson"
    assert runtime_input.structure_locked is True
    assert runtime_input.can_structural_edit is False
    assert runtime_input.exact_energy is None
