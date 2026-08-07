from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipelines.exact_bench.hh_noise_robustness_seq_report as report


class _LayoutStub:
    logical_parameter_count = 14
    runtime_parameter_count = 25


def test_imported_ansatz_input_state_audit_unavailable_without_ansatz_input_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        report,
        "_load_imported_artifact_context",
        lambda artifact_json: {
            "path": Path(artifact_json),
            "source_kind": "direct_adapt_artifact",
            "ansatz_input_state_meta": {
                "available": False,
                "reason": "missing_ansatz_input_state_provenance",
                "error": None,
                "source": None,
                "handoff_state_kind": None,
            },
        },
    )

    payload = report._run_imported_ansatz_input_state_audit(
        artifact_json="artifacts/json/example.json",
        noise_mode="backend_scheduled",
        shots=4096,
        seed=7,
        oracle_repeats=8,
        oracle_aggregate="mean",
        mitigation_config={"mode": "none", "zne_scales": [], "dd_sequence": None, "local_readout_strategy": None},
        symmetry_mitigation_config={"mode": "off"},
        backend_name="FakeGuadalupeV2",
        use_fake_backend=True,
        allow_aer_fallback=True,
        omp_shm_workaround=True,
    )

    assert payload["success"] is False
    assert payload["available"] is False
    assert payload["reason"] == "missing_ansatz_input_state_provenance"
    assert payload["reference_state_embedded"] is False


def test_imported_ansatz_input_state_audit_reports_stateprep_metadata_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_state: dict[str, object] = {}

    monkeypatch.setattr(
        report,
        "_load_imported_artifact_context",
        lambda artifact_json: {
            "path": Path(artifact_json),
            "source_kind": "handoff_bundle",
            "settings": {"L": 2, "ordering": "blocked"},
            "ordered_labels_exyz": ["ze"],
            "static_coeff_map_exyz": {"ze": 1.0 + 0.0j},
            "num_qubits": 2,
            "ansatz_input_state": [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            "layout": _LayoutStub(),
            "saved_energy": 0.1,
            "exact_energy": 0.05,
            "ansatz_input_state_meta": {
                "available": True,
                "reason": None,
                "error": None,
                "source": "warm_start_hva",
                "handoff_state_kind": "prepared_state",
            },
        },
    )

    def _fake_append_reference_state(circuit, state):
        captured_state["num_qubits"] = circuit.num_qubits
        captured_state["state"] = tuple(state.tolist())

    def _fake_core(**kwargs):
        assert kwargs["seed_transpiler"] == 5
        assert kwargs["transpile_optimization_level"] == 2
        assert kwargs["compile_request_source"] == "fixed_scaffold_runtime_transpile_cli"
        return {
            "success": True,
            "final_observables": {
                "energy_static": {
                    "ideal_mean": 0.125,
                    "noisy_mean": 0.2,
                    "delta_mean": 0.075,
                }
            },
            "audit_source": dict(kwargs.get("audit_source", {})),
            "compile_control": {
                "backend_name": None,
                "seed_transpiler": 5,
                "transpile_optimization_level": 2,
                "source": "fixed_scaffold_runtime_transpile_cli",
            },
            "compile_observation": {
                "available": True,
                "requested": {
                    "backend_name": None,
                    "seed_transpiler": 5,
                    "transpile_optimization_level": 2,
                    "source": "fixed_scaffold_runtime_transpile_cli",
                },
                "observed": {
                    "backend_name": None,
                    "seed_transpiler": 5,
                    "transpile_optimization_level": 2,
                },
                "matches_requested": True,
                "mismatch_fields": [],
                "reason": None,
            },
        }

    monkeypatch.setattr(report, "_append_reference_state", _fake_append_reference_state)
    monkeypatch.setattr(report, "_run_static_observable_audit_core", _fake_core)

    payload = report._run_imported_ansatz_input_state_audit(
        artifact_json="artifacts/json/example.json",
        noise_mode="ideal",
        shots=4096,
        seed=7,
        oracle_repeats=8,
        oracle_aggregate="mean",
        mitigation_config={"mode": "none", "zne_scales": [], "dd_sequence": None, "local_readout_strategy": None},
        symmetry_mitigation_config={"mode": "off"},
        backend_name=None,
        use_fake_backend=False,
        allow_aer_fallback=True,
        omp_shm_workaround=True,
        seed_transpiler=5,
        transpile_optimization_level=2,
        compile_request_source="fixed_scaffold_runtime_transpile_cli",
    )

    assert captured_state["num_qubits"] == 2
    assert captured_state["state"][0] == pytest.approx(1.0 + 0.0j)
    assert payload["success"] is True
    assert payload["reference_state_embedded"] is True
    assert payload["audit_source"]["kind"] == "imported_ansatz_input_state"
    assert payload["audit_source"]["includes_ansatz_stateprep_noise"] is False
    assert payload["ansatz_input_state_source"] == "warm_start_hva"
    assert payload["ansatz_input_state_kind"] == "prepared_state"
    assert payload["ansatz_input_state_reference"]["artifact_full_circuit_saved_energy"] == pytest.approx(0.1)
    assert payload["ansatz_input_state_reference"]["ideal_input_state_energy"] == pytest.approx(0.125)
    assert payload["compile_control"]["seed_transpiler"] == 5
    assert payload["compile_observation"]["matches_requested"] is True


def test_imported_full_circuit_audit_unavailable_without_ansatz_input_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        report,
        "_load_imported_artifact_context",
        lambda artifact_json: {
            "path": Path(artifact_json),
            "source_kind": "direct_adapt_artifact",
            "ansatz_input_state_meta": {
                "available": False,
                "reason": "missing_ansatz_input_state_provenance",
                "error": None,
                "source": None,
                "handoff_state_kind": None,
            },
        },
    )

    payload = report._run_imported_full_circuit_audit(
        artifact_json="artifacts/json/example.json",
        noise_mode="backend_scheduled",
        shots=4096,
        seed=7,
        oracle_repeats=8,
        oracle_aggregate="mean",
        mitigation_config={"mode": "none", "zne_scales": [], "dd_sequence": None, "local_readout_strategy": None},
        symmetry_mitigation_config={"mode": "off"},
        backend_name="FakeGuadalupeV2",
        use_fake_backend=True,
        allow_aer_fallback=True,
        omp_shm_workaround=True,
    )

    assert payload["success"] is False
    assert payload["available"] is False
    assert payload["reason"] == "missing_ansatz_input_state_provenance"
    assert payload["reference_state_embedded"] is False


def test_imported_full_circuit_audit_reports_ansatz_input_metadata_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        report,
        "_load_imported_artifact_context",
        lambda artifact_json: {
            "path": Path(artifact_json),
            "source_kind": "handoff_bundle",
            "settings": {"L": 2, "ordering": "blocked"},
            "ordered_labels_exyz": ["ze"],
            "static_coeff_map_exyz": {"ze": 1.0 + 0.0j},
            "circuit": object(),
            "layout": _LayoutStub(),
            "saved_energy": 0.1,
            "exact_energy": 0.05,
            "ansatz_input_state_meta": {
                "available": True,
                "reason": None,
                "error": None,
                "source": "warm_start_hva",
                "handoff_state_kind": "prepared_state",
            },
        },
    )

    def _fake_core(**kwargs):
        assert kwargs["seed_transpiler"] == 5
        assert kwargs["transpile_optimization_level"] == 2
        assert kwargs["compile_request_source"] == "fixed_scaffold_runtime_transpile_cli"
        return {
            "success": True,
            "final_observables": {
                "energy_static": {
                    "ideal_mean": 0.125,
                    "noisy_mean": 0.2,
                    "delta_mean": 0.075,
                }
            },
            "audit_source": dict(kwargs.get("audit_source", {})),
            "compile_control": {
                "backend_name": None,
                "seed_transpiler": 5,
                "transpile_optimization_level": 2,
                "source": "fixed_scaffold_runtime_transpile_cli",
            },
            "compile_observation": {
                "available": True,
                "requested": {
                    "backend_name": None,
                    "seed_transpiler": 5,
                    "transpile_optimization_level": 2,
                    "source": "fixed_scaffold_runtime_transpile_cli",
                },
                "observed": {
                    "backend_name": None,
                    "seed_transpiler": 5,
                    "transpile_optimization_level": 2,
                },
                "matches_requested": True,
                "mismatch_fields": [],
                "reason": None,
            },
        }

    monkeypatch.setattr(report, "_run_static_observable_audit_core", _fake_core)

    payload = report._run_imported_full_circuit_audit(
        artifact_json="artifacts/json/example.json",
        noise_mode="ideal",
        shots=4096,
        seed=7,
        oracle_repeats=8,
        oracle_aggregate="mean",
        mitigation_config={"mode": "none", "zne_scales": [], "dd_sequence": None, "local_readout_strategy": None},
        symmetry_mitigation_config={"mode": "off"},
        backend_name=None,
        use_fake_backend=False,
        allow_aer_fallback=True,
        omp_shm_workaround=True,
        seed_transpiler=5,
        transpile_optimization_level=2,
        compile_request_source="fixed_scaffold_runtime_transpile_cli",
    )

    assert payload["success"] is True
    assert payload["audit_source"]["reference_state_embedded"] is True
    assert payload["audit_source"]["ansatz_input_state_source"] == "warm_start_hva"
    assert payload["audit_source"]["ansatz_input_state_kind"] == "prepared_state"
    assert payload["full_circuit_reference"]["saved_artifact_energy"] == pytest.approx(0.1)
    assert payload["full_circuit_reference"]["ideal_circuit_energy"] == pytest.approx(0.125)
    assert payload["compile_control"]["seed_transpiler"] == 5
    assert payload["compile_observation"]["matches_requested"] is True
