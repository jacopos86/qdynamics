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

import pipelines.scaffold.adapt_circuit_cost as hardcoded_compile_scout_mod
import pipelines.scaffold.adapt_circuit_cost as scaffold_compile_scout_mod

from pipelines.scaffold.adapt_circuit_cost import (
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
from pipelines.scaffold.imported_artifact_resolution import ImportedArtifactResolution
from src.quantum.ansatz_parameterization import build_parameter_layout, serialize_layout
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, PauliPolynomial, PauliTerm
from pipelines.exact_bench.table_i_qiskit_resource_compile import (
    compile_table_i_ansatz_terms,
    compile_table_i_pauli_label_groups,
)
from pipelines.qiskit_backend_tools import compiled_gate_stats


def test_compiled_gate_stats_reports_one_qubit_count_without_noops() -> None:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.sx(1)
    qc.rz(0.25, 0)
    qc.id(0)
    qc.barrier(0, 1)
    qc.cx(0, 1)

    stats = compiled_gate_stats(qc)

    assert stats["compiled_count_1q"] == 3
    assert stats["compiled_count_2q"] == 1
    assert stats["compiled_count_1q_semantics"] == "post_transpile_one_qubit_quantum_ops_excluding_barrier_delay_id_measure_reset"


def test_table_i_pauli_compile_payload_exposes_one_qubit_total() -> None:
    payload = compile_table_i_pauli_label_groups(
        pauli_label_groups=(("xx",),),
        num_qubits=2,
        reference_state=None,
        source_kind="unit_test_prefix_resource",
    )

    assert payload["compiled_circuit_stats_status"] == "ok"
    assert "compiled_count_1q_total" in payload
    assert payload["compiled_count_1q_total"] >= 0
    assert payload["compiled_count_1q_semantics"] == "post_transpile_one_qubit_quantum_ops_excluding_barrier_delay_id_measure_reset"
    assert payload["qiskit_basis_work_status"] == "ok"
    assert payload["qiskit_pretranspile_basis_change_1q_total"] == 4
    assert payload["qiskit_pretranspile_pauli_rotation_rz_total"] == 1
    assert payload["qiskit_pretranspile_pauli_1q_work_total"] == 5
    assert payload["qiskit_pretranspile_pauli_1q_work_components"] == {
        "h": 4,
        "s": 0,
        "sdg": 0,
        "rz": 1,
    }


def test_table_i_qiskit_basis_work_excludes_reference_state_preparation() -> None:
    reference_state = np.array([1.0, 1.0, 0.0, 0.0], dtype=complex) / np.sqrt(2.0)
    payload = compile_table_i_pauli_label_groups(
        pauli_label_groups=(("yy",),),
        num_qubits=2,
        reference_state=reference_state,
        source_kind="unit_test_prefix_resource",
    )

    assert payload["qiskit_basis_work_status"] == "ok"
    assert payload["qiskit_pretranspile_basis_change_1q_total"] == 8
    assert payload["qiskit_pretranspile_pauli_rotation_rz_total"] == 1
    assert payload["qiskit_pretranspile_pauli_1q_work_total"] == 9


def test_table_i_qiskit_basis_work_fails_closed_for_exact_unitary_block() -> None:
    grouped = AnsatzTerm(
        label="noncommuting_group",
        polynomial=PauliPolynomial(
            "JW",
            [
                PauliTerm(1, ps="x", pc=1.0),
                PauliTerm(1, ps="z", pc=1.0),
            ],
        ),
        execution_mode="grouped_exact",
    )

    payload = compile_table_i_ansatz_terms(
        ops=(grouped,),
        num_qubits=1,
        reference_state=None,
        source_kind="unit_test_prefix_resource",
    )

    assert (
        payload["qiskit_basis_work_status"]
        == "unavailable_noncommuting_grouped_exact_synthesis"
    )
    assert payload["qiskit_pretranspile_basis_change_1q_total"] is None
    assert payload["qiskit_pretranspile_pauli_1q_work_total"] is None
    assert payload["qiskit_basis_work_non_attributable_operator_count"] == 1


def test_hardcoded_compile_scout_wrapper_aliases_scaffold_owner() -> None:
    assert hardcoded_compile_scout_mod is scaffold_compile_scout_mod
    for name in (
        "CompileScoutConfig",
        "resolve_compile_scout_config",
        "reconstruct_imported_adapt_circuit",
        "run_compile_scout",
    ):
        assert getattr(hardcoded_compile_scout_mod, name) is getattr(scaffold_compile_scout_mod, name)


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


def test_build_ansatz_circuit_y_basis_round_trip_counts_sdg_h_and_h_s() -> None:
    terms = [
        AnsatzTerm(label="g0", polynomial=PauliPolynomial("JW", [PauliTerm(2, ps="yy", pc=1.0)])),
    ]
    layout = build_parameter_layout(terms)
    qc = _build_ansatz_circuit(layout, np.array([0.2], dtype=float), 2)
    counts = qc.count_ops()

    assert int(counts.get("cx", 0)) == 2
    assert int(counts.get("rz", 0)) == 1
    assert int(counts.get("h", 0)) == 4
    assert int(counts.get("sdg", 0)) == 2
    assert int(counts.get("s", 0)) == 2


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


def test_resolve_runtime_layout_and_theta_repairs_stale_pre_prune_layout() -> None:
    terms = [
        AnsatzTerm(
            label="g0",
            polynomial=PauliPolynomial(
                "JW",
                [PauliTerm(2, ps="xx", pc=1.0), PauliTerm(2, ps="zz", pc=0.5)],
            ),
        ),
        AnsatzTerm(
            label="removed",
            polynomial=PauliPolynomial("JW", [PauliTerm(2, ps="xy", pc=1.0)]),
        ),
        AnsatzTerm(
            label="g2",
            polynomial=PauliPolynomial("JW", [PauliTerm(2, ps="yy", pc=1.0)]),
        ),
    ]
    stale_layout = build_parameter_layout(terms)

    restored_layout, restored_theta = _resolve_runtime_layout_and_theta(
        {
            "adapt_vqe": {
                "operators": ["g0", "g2"],
                "optimal_point": [0.1, 0.1, -0.2],
                "logical_optimal_point": [0.1, -0.2],
                "parameterization": serialize_layout(stale_layout),
            }
        },
        (),
    )

    assert [block.candidate_label for block in restored_layout.blocks] == ["g0", "g2"]
    assert [block.runtime_start for block in restored_layout.blocks] == [0, 2]
    assert int(restored_layout.runtime_parameter_count) == 3
    assert np.allclose(restored_theta, [0.1, 0.1, -0.2])


def test_resolve_runtime_layout_and_theta_does_not_guess_non_subsequence_layout() -> None:
    stale_layout = build_parameter_layout(_legacy_scaffold_terms())

    with pytest.raises(ValueError, match="Runtime theta length"):
        _resolve_runtime_layout_and_theta(
            {
                "adapt_vqe": {
                    "operators": ["unknown"],
                    "optimal_point": [0.2],
                    "logical_optimal_point": [0.2],
                    "parameterization": serialize_layout(stale_layout),
                }
            },
            (),
        )


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


def test_reconstruct_imported_adapt_circuit_accepts_child_set_metadata_only() -> None:
    label = "g_parent::child_set[0]"
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
            "adapt_pool": "full_meta",
        },
        "adapt_vqe": {
            "operators": [label],
            "optimal_point": [0.3],
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
                    "candidate_label": label,
                    "compile_metadata": {
                        "serialized_terms_exyz": [
                            {"pauli_exyz": "xee", "coeff_re": 1.0, "coeff_im": 0.0, "nq": 3},
                            {"pauli_exyz": "zee", "coeff_re": 0.25, "coeff_im": 0.0, "nq": 3},
                        ]
                    },
                }
            ]
        },
    }

    bundle = reconstruct_imported_adapt_circuit(payload)

    assert int(bundle["layout"].logical_parameter_count) == 1
    assert bundle["layout"].blocks[0].candidate_label == label
    assert int(bundle["num_qubits"]) == 3
    assert np.allclose(bundle["theta_runtime"], [0.3])


def test_reconstruct_imported_adapt_circuit_prefers_runtime_loader_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = build_parameter_layout(_legacy_scaffold_terms())
    runtime_input = SimpleNamespace(
        psi_ref=np.array([1.0, 0.0, 0.0, 0.0], dtype=complex),
        base_layout=layout,
        theta_runtime=np.array([0.1, 0.1, -0.2], dtype=float),
        resolved_problem=SimpleNamespace(layout=SimpleNamespace(total_qubits=2)),
        h_poly=PauliPolynomial("JW", [PauliTerm(2, ps="zz", pc=1.0)]),
    )
    payload = {
        "settings": {"L": 1, "problem": "hh"},
        "adapt_vqe": {"operators": ["g0", "g1"], "optimal_point": [0.1, -0.2]},
        "ansatz_input_state": {
            "source": "hf",
            "handoff_state_kind": "reference_state",
            "nq_total": 2,
            "amplitudes_qn_to_q0": {"00": {"re": 1.0, "im": 0.0}},
        },
    }

    monkeypatch.setattr(
        "pipelines.scaffold.adapt_circuit_cost.load_scaffold_runtime_input_from_payload",
        lambda data: runtime_input,
    )

    bundle = reconstruct_imported_adapt_circuit(payload)

    assert bundle["reconstruction_source"] == "runtime_loader"
    assert bundle["layout"] is layout
    assert np.allclose(bundle["theta_runtime"], runtime_input.theta_runtime)
    assert int(bundle["num_qubits"]) == 2


def test_reconstruct_imported_adapt_circuit_falls_back_when_runtime_loader_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "settings": {"L": 1, "problem": "hh"},
        "adapt_vqe": {"operators": ["g0"], "optimal_point": [0.0]},
        "ansatz_input_state": {
            "source": "hf",
            "handoff_state_kind": "reference_state",
            "nq_total": 1,
            "amplitudes_qn_to_q0": {"0": {"re": 1.0, "im": 0.0}},
        },
    }

    monkeypatch.setattr(
        "pipelines.scaffold.adapt_circuit_cost.load_scaffold_runtime_input_from_payload",
        lambda data: (_ for _ in ()).throw(ValueError("boom")),
    )
    monkeypatch.setattr(
        hardcoded_compile_scout_mod,
        "_reconstruct_imported_adapt_circuit_legacy",
        lambda data: {"payload": data, "reconstruction_source": "legacy"},
    )

    bundle = reconstruct_imported_adapt_circuit(payload)

    assert bundle["reconstruction_source"] == "legacy_fallback"
    assert "ValueError: boom" in bundle["runtime_loader_warning"]


def test_reconstruct_imported_adapt_circuit_forbids_hh_fallback_for_h2o(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "settings": {
            "L": 6,
            "problem": "molecular_vibronic_h2o_linear_fd",
        },
        "adapt_vqe": {"operators": [], "optimal_point": []},
        "ansatz_input_state": {
            "source": "hf",
            "handoff_state_kind": "reference_state",
            "nq_total": 1,
            "amplitudes_qn_to_q0": {"0": {"re": 1.0, "im": 0.0}},
        },
    }
    monkeypatch.setattr(
        "pipelines.scaffold.adapt_circuit_cost.load_scaffold_runtime_input_from_payload",
        lambda data: (_ for _ in ()).throw(ValueError("fixture path missing")),
    )

    with pytest.raises(ValueError, match="legacy Hamiltonian fallback is forbidden"):
        reconstruct_imported_adapt_circuit(payload)


def test_reconstruct_imported_adapt_circuit_applies_fixture_override_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}
    layout = build_parameter_layout([])
    runtime_input = SimpleNamespace(
        psi_ref=np.array([1.0, 0.0], dtype=complex),
        base_layout=layout,
        theta_runtime=np.array([], dtype=float),
        resolved_problem=SimpleNamespace(layout=SimpleNamespace(total_qubits=1)),
        h_poly=PauliPolynomial("JW", [PauliTerm(1, ps="z", pc=1.0)]),
    )

    def _loader(data: dict[str, object]) -> SimpleNamespace:
        seen.update(data["settings"])
        return runtime_input

    monkeypatch.setattr(
        "pipelines.scaffold.adapt_circuit_cost.load_scaffold_runtime_input_from_payload",
        _loader,
    )
    payload = {
        "settings": {"problem": "molecular_vibronic_h2o_linear_fd"},
        "adapt_vqe": {"operators": [], "optimal_point": []},
        "ansatz_input_state": {
            "source": "hf",
            "handoff_state_kind": "reference_state",
            "nq_total": 1,
            "amplitudes_qn_to_q0": {"0": {"re": 1.0, "im": 0.0}},
        },
    }

    bundle = reconstruct_imported_adapt_circuit(
        payload,
        settings_overrides={
            "molecular_vibronic_h2o_linear_fd_fixture_json": "fixture.json"
        },
    )

    assert bundle["reconstruction_source"] == "runtime_loader"
    assert seen["molecular_vibronic_h2o_linear_fd_fixture_json"] == "fixture.json"


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
        "pipelines.scaffold.adapt_circuit_cost.resolve_imported_artifact_path",
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
    monkeypatch.setattr("pipelines.scaffold.adapt_circuit_cost._load_adapt_result", lambda path: {})
    monkeypatch.setattr(
        "pipelines.scaffold.adapt_circuit_cost.reconstruct_imported_adapt_circuit",
        lambda payload: _fake_compile_bundle(),
    )
    monkeypatch.setattr(
        "pipelines.scaffold.adapt_circuit_cost.list_local_fake_backend_names",
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

    monkeypatch.setattr("pipelines.scaffold.adapt_circuit_cost._load_fake_backend", _fake_load_backend)
    monkeypatch.setattr("pipelines.scaffold.adapt_circuit_cost.compile_circuit_for_local_backend", _fake_compile)

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
    monkeypatch.setattr("pipelines.scaffold.adapt_circuit_cost._load_adapt_result", lambda path: {})
    monkeypatch.setattr(
        "pipelines.scaffold.adapt_circuit_cost.reconstruct_imported_adapt_circuit",
        lambda payload: _fake_compile_bundle(),
    )

    class _Backend:
        def __init__(self, name: str) -> None:
            self.name = name
            self.num_qubits = 5

    monkeypatch.setattr(
        "pipelines.scaffold.adapt_circuit_cost._load_fake_backend",
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

    monkeypatch.setattr("pipelines.scaffold.adapt_circuit_cost.compile_circuit_for_local_backend", _fake_compile)

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
