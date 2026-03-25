from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_ibm_runtime")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.adapt_circuit_cost import (
    CompileScoutConfig,
    _build_ansatz_circuit,
    _normalize_adapt_payload,
    _resolve_ansatz_input_state_from_payload,
    _resolve_runtime_layout_and_theta,
    _resolve_total_qubits,
    reconstruct_imported_adapt_circuit,
    resolve_compile_scout_config,
    run_compile_scout,
)
from pipelines.hardcoded.imported_artifact_resolution import ImportedArtifactResolution
from src.quantum.ansatz_parameterization import build_parameter_layout, serialize_layout
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, PauliPolynomial, PauliTerm


def _legacy_scaffold_terms() -> list[AnsatzTerm]:
    return [
        AnsatzTerm(
            label="g0",
            polynomial=PauliPolynomial(
                "JW",
                [PauliTerm(2, ps="xx", pc=1.0), PauliTerm(2, ps="zz", pc=0.5)],
            ),
        ),
        AnsatzTerm(
            label="g1",
            polynomial=PauliPolynomial("JW", [PauliTerm(2, ps="xy", pc=1.0)]),
        ),
    ]


def test_build_ansatz_circuit_uses_direct_per_pauli_rotations() -> None:
    terms = [
        AnsatzTerm(label="g0", polynomial=PauliPolynomial("JW", [PauliTerm(2, ps="xx", pc=1.0)])),
        AnsatzTerm(label="g1", polynomial=PauliPolynomial("JW", [PauliTerm(2, ps="ez", pc=1.0)])),
    ]
    layout = build_parameter_layout(terms)
    qc = _build_ansatz_circuit(layout, np.array([0.2, -0.3], dtype=float), 2)
    counts = qc.count_ops()

    assert int(counts.get("cx", 0)) == 2
    assert int(counts.get("rz", 0)) == 2
    assert int(counts.get("h", 0)) == 4


def test_build_ansatz_circuit_embeds_generic_ansatz_input_state() -> None:
    layout = build_parameter_layout([])
    ref_state = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    qc = _build_ansatz_circuit(layout, np.array([], dtype=float), 1, ref_state=ref_state)
    counts = qc.count_ops()

    assert int(counts.get("initialize", 0)) == 1


def test_resolve_runtime_layout_and_theta_expands_legacy_logical_theta() -> None:
    scaffold_ops = _legacy_scaffold_terms()
    layout, theta_runtime = _resolve_runtime_layout_and_theta(
        {"adapt_vqe": {"optimal_point": [0.4, -0.2]}},
        scaffold_ops,
    )

    assert int(layout.logical_parameter_count) == 2
    assert int(layout.runtime_parameter_count) == 3
    assert np.allclose(theta_runtime, [0.4, 0.4, -0.2])


def test_resolve_runtime_layout_and_theta_accepts_serialized_parameterization() -> None:
    scaffold_ops = _legacy_scaffold_terms()
    layout = build_parameter_layout(scaffold_ops)
    theta_runtime = np.array([0.1, 0.15, -0.2], dtype=float)
    restored_layout, restored_theta = _resolve_runtime_layout_and_theta(
        {
            "adapt_vqe": {
                "optimal_point": theta_runtime.tolist(),
                "parameterization": serialize_layout(layout),
            }
        },
        scaffold_ops,
    )

    assert int(restored_layout.logical_parameter_count) == 2
    assert int(restored_layout.runtime_parameter_count) == 3
    assert np.allclose(restored_theta, theta_runtime)


def test_resolve_runtime_layout_and_theta_preserves_serialized_block_order() -> None:
    terms = [
        AnsatzTerm(
            label="gB",
            polynomial=PauliPolynomial("JW", [PauliTerm(2, ps="xy", pc=1.0)]),
        ),
        AnsatzTerm(
            label="gA",
            polynomial=PauliPolynomial("JW", [PauliTerm(2, ps="zz", pc=1.0)]),
        ),
    ]
    layout = build_parameter_layout(terms, sort_terms=False)
    theta_runtime = np.array([0.3, -0.1], dtype=float)

    restored_layout, restored_theta = _resolve_runtime_layout_and_theta(
        {
            "adapt_vqe": {
                "optimal_point": theta_runtime.tolist(),
                "parameterization": serialize_layout(layout),
            }
        },
        list(reversed(terms)),
    )

    assert [block.candidate_label for block in restored_layout.blocks] == ["gB", "gA"]
    assert np.allclose(restored_theta, theta_runtime)


def test_normalize_adapt_payload_accepts_raw_top_level_adapt_json_shape() -> None:
    raw = {
        "settings": {"L": 2, "n_ph_max": 1, "boson_encoding": "binary"},
        "energy": -1.2,
        "exact_gs_energy": -1.3,
        "operators": ["g0", "g1"],
        "optimal_point": [0.1, 0.2],
        "ansatz_depth": 2,
        "num_parameters": 2,
        "pool_type": "pareto_lean_l2",
    }
    normalized = _normalize_adapt_payload(raw)

    assert normalized["settings"]["L"] == 2
    assert normalized["adapt_vqe"]["operators"] == ["g0", "g1"]
    assert normalized["adapt_vqe"]["optimal_point"] == [0.1, 0.2]
    assert normalized["adapt_vqe"]["pool_type"] == "pareto_lean_l2"


def test_resolve_total_qubits_uses_layout_nq_or_encoding_fallback() -> None:
    layout = build_parameter_layout(_legacy_scaffold_terms())
    assert _resolve_total_qubits({"L": 2, "n_ph_max": 1, "boson_encoding": "binary"}, layout) == 2

    empty_layout = build_parameter_layout([])
    assert _resolve_total_qubits({"L": 2, "n_ph_max": 1, "boson_encoding": "unary"}, empty_layout) == 8


def test_resolve_ansatz_input_state_from_payload_uses_top_level_provenance_only() -> None:
    state, meta = _resolve_ansatz_input_state_from_payload(
        {
            "ansatz_input_state": {
                "source": "warm_start_hva",
                "handoff_state_kind": "prepared_state",
                "nq_total": 1,
                "amplitudes_qn_to_q0": {
                    "0": {"re": 1.0, "im": 0.0},
                },
            },
            "initial_state": {
                "source": "adapt_vqe",
                "handoff_state_kind": "prepared_state",
                "nq_total": 1,
                "amplitudes_qn_to_q0": {
                    "1": {"re": 1.0, "im": 0.0},
                },
            },
        }
    )

    assert meta["available"] is True
    assert meta["source"] == "warm_start_hva"
    assert meta["handoff_state_kind"] == "prepared_state"
    assert np.allclose(state, [1.0, 0.0])


def test_reconstruct_imported_adapt_circuit_requires_ansatz_input_state_for_embedding() -> None:
    payload = {
        "settings": {
            "L": 1,
            "t": 1.0,
            "u": 4.0,
            "dv": 0.0,
            "omega0": 1.0,
            "g_ep": 0.5,
            "n_ph_max": 1,
            "boson_encoding": "binary",
            "ordering": "blocked",
            "boundary": "open",
        },
        "adapt_vqe": {
            "operators": ["g0"],
            "optimal_point": [0.0],
        },
        "initial_state": {
            "source": "adapt_vqe",
            "handoff_state_kind": "prepared_state",
            "nq_total": 3,
            "amplitudes_qn_to_q0": {"001": {"re": 1.0, "im": 0.0}},
        },
        "ansatz_input_state": {
            "source": "hf",
            "handoff_state_kind": "reference_state",
            "nq_total": 3,
            "amplitudes_qn_to_q0": {"100": {"re": 1.0, "im": 0.0}},
        },
        "continuation": {
            "selected_generator_metadata": [
                {
                    "candidate_label": "g0",
                    "compile_metadata": {
                        "serialized_terms_exyz": [
                            {"pauli_exyz": "eee", "coeff_re": 1.0, "coeff_im": 0.0, "nq": 3},
                        ]
                    },
                }
            ]
        },
    }

    bundle = reconstruct_imported_adapt_circuit(payload)
    assert bundle["ansatz_input_state_meta"]["available"] is True
    assert bundle["ansatz_input_state_meta"]["source"] == "hf"
    assert bundle["ansatz_input_state"] is not None


def _fake_compile_bundle() -> dict[str, object]:
    qc = QuantumCircuit(2)
    qc.h(0)
    h_poly = PauliPolynomial("JW", [PauliTerm(2, ps="zz", pc=1.0)])
    return {
        "payload": {"exact": {"E_exact_sector": -1.1}},
        "adapt_vqe": {"operators": ["g0"], "energy": -1.0, "exact_gs_energy": -1.1},
        "settings": {"L": 1},
        "h_poly": h_poly,
        "layout": SimpleNamespace(logical_parameter_count=1, runtime_parameter_count=1),
        "theta_runtime": np.array([0.0], dtype=float),
        "num_qubits": 2,
        "ansatz_input_state": np.array([1.0, 0.0, 0.0, 0.0], dtype=complex),
        "ansatz_input_state_meta": {
            "available": True,
            "source": "hf",
            "handoff_state_kind": "reference_state",
        },
        "circuit": qc,
    }


def test_resolve_compile_scout_config_defaults_to_imported_source(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = ImportedArtifactResolution(
        mode="imported_artifact",
        requested_json=Path("artifacts/json/default.json"),
        resolved_json=Path("artifacts/json/default.json"),
        source_kind="direct_payload",
        default_subject=True,
    )
    monkeypatch.setattr(
        "pipelines.hardcoded.adapt_circuit_cost.resolve_imported_artifact_path",
        lambda requested_json, require_default_import_source: expected,
    )

    cfg = resolve_compile_scout_config(
        SimpleNamespace(
            artifact_json_flag=None,
            artifact_json=None,
            backend_name="ibm_boston",
            legacy_backend_name=None,
            optimization_level=None,
            legacy_opt_level=None,
            candidate_backends="FakeGuadalupeV2,FakeManilaV2",
            sweep_backends=False,
            seed_transpiler=11,
            output_json=None,
        )
    )

    assert cfg.source == expected
    assert cfg.requested_backend_name == "ibm_boston"
    assert cfg.candidate_backends == ("FakeGuadalupeV2", "FakeManilaV2")
    assert cfg.sweep_backends is False
    assert cfg.seed_transpiler == 11
    assert cfg.optimization_level == 1


def test_run_compile_scout_falls_back_to_sweep_when_requested_backend_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("pipelines.hardcoded.adapt_circuit_cost._load_adapt_result", lambda path: {})
    monkeypatch.setattr(
        "pipelines.hardcoded.adapt_circuit_cost.reconstruct_imported_adapt_circuit",
        lambda payload: _fake_compile_bundle(),
    )
    monkeypatch.setattr(
        "pipelines.hardcoded.adapt_circuit_cost.list_local_fake_backend_names",
        lambda: ("FakeAlphaV2", "FakeBetaV2"),
    )

    class _Backend:
        def __init__(self, name: str) -> None:
            self.name = name
            self.num_qubits = 5

    def _fake_load_backend(name: str | None) -> tuple[object, str]:
        if str(name) == "ibm_boston":
            raise ValueError("Unknown fake backend 'FakeIbmBostonV2'.")
        return _Backend(str(name)), str(name)

    def _fake_compile(qc: QuantumCircuit, backend: object, *, seed_transpiler: int, optimization_level: int) -> dict[str, object]:
        compiled = QuantumCircuit(2)
        if getattr(backend, "name", "") == "FakeAlphaV2":
            compiled.cx(0, 1)
        return {
            "compiled": compiled,
            "logical_to_physical": (0, 1),
            "compiled_num_qubits": 2,
        }

    monkeypatch.setattr("pipelines.hardcoded.adapt_circuit_cost._load_fake_backend", _fake_load_backend)
    monkeypatch.setattr("pipelines.hardcoded.adapt_circuit_cost.compile_circuit_for_local_backend", _fake_compile)

    cfg = CompileScoutConfig(
        source=ImportedArtifactResolution(
            mode="imported_artifact",
            requested_json=tmp_path / "lean.json",
            resolved_json=tmp_path / "lean.json",
            source_kind="direct_payload",
            default_subject=False,
        ),
        requested_backend_name="ibm_boston",
        candidate_backends=("FakeAlphaV2", "FakeBetaV2"),
        sweep_backends=False,
        seed_transpiler=7,
        optimization_level=1,
        output_json=tmp_path / "compile_scout.json",
    )

    payload = run_compile_scout(cfg)

    assert payload["requested_backend"]["supported_locally"] is False
    assert payload["requested_backend"]["fallback_to_sweep"] is True
    assert payload["selected_backend"]["transpile_backend"] == "FakeBetaV2"
    assert Path(payload["artifacts"]["output_json"]).exists()


def test_run_compile_scout_ranks_by_2q_then_depth_then_size(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("pipelines.hardcoded.adapt_circuit_cost._load_adapt_result", lambda path: {})
    monkeypatch.setattr(
        "pipelines.hardcoded.adapt_circuit_cost.reconstruct_imported_adapt_circuit",
        lambda payload: _fake_compile_bundle(),
    )

    class _Backend:
        def __init__(self, name: str) -> None:
            self.name = name
            self.num_qubits = 5

    monkeypatch.setattr(
        "pipelines.hardcoded.adapt_circuit_cost._load_fake_backend",
        lambda name: (_Backend(str(name)), str(name)),
    )

    def _fake_compile(qc: QuantumCircuit, backend: object, *, seed_transpiler: int, optimization_level: int) -> dict[str, object]:
        compiled = QuantumCircuit(2)
        if getattr(backend, "name", "") == "FakeAlphaV2":
            compiled.cx(0, 1)
            compiled.cx(0, 1)
        elif getattr(backend, "name", "") == "FakeBetaV2":
            compiled.cx(0, 1)
            compiled.h(0)
            compiled.h(1)
        else:
            compiled.cx(0, 1)
            compiled.h(0)
        return {
            "compiled": compiled,
            "logical_to_physical": (0, 1),
            "compiled_num_qubits": 2,
        }

    monkeypatch.setattr("pipelines.hardcoded.adapt_circuit_cost.compile_circuit_for_local_backend", _fake_compile)

    cfg = CompileScoutConfig(
        source=ImportedArtifactResolution(
            mode="imported_artifact",
            requested_json=tmp_path / "lean.json",
            resolved_json=tmp_path / "lean.json",
            source_kind="direct_payload",
            default_subject=False,
        ),
        requested_backend_name=None,
        candidate_backends=("FakeAlphaV2", "FakeBetaV2", "FakeGammaV2"),
        sweep_backends=True,
        seed_transpiler=7,
        optimization_level=1,
        output_json=tmp_path / "compile_rank.json",
    )

    payload = run_compile_scout(cfg)

    assert payload["selected_backend"]["transpile_backend"] == "FakeGammaV2"
    assert payload["selected_backend"]["compiled_count_2q"] == 1
    assert payload["selected_backend"]["compiled_depth"] <= payload["rows"][1]["compiled_depth"]
