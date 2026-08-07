from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.scaffold.runtime_contract import CandidatePoolSource
from pipelines.time_dynamics.benchmarks import registry as benchmark_registry
from pipelines.time_dynamics.diagnostics import avqds_results_report as report_mod
from pipelines.time_dynamics.benchmarks.avqds_tetris import (
    TetrisCandidateScore,
    TetrisPoolAtom,
    select_avqds_method1_candidate,
    select_tetris_method3_layer,
    solve_avqds_projective_geometry,
)
from pipelines.time_dynamics.benchmarks.common import (
    _build_layout_for_terms,
    _compiled_executor_for_terms,
)
from pipelines.time_dynamics.tables import generic_dynamics_rows as rows_mod
from pipelines.time_dynamics.tables.dynamics_benchmark_contract import DynamicsBenchmarkCase
from src.quantum.ansatz_parameterization import build_parameter_layout
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, hamiltonian_matrix


def _poly(pauli: str, coefficient: float = 1.0) -> PauliPolynomial:
    return PauliPolynomial(
        "JW",
        [PauliTerm(len(pauli), ps=str(pauli), pc=float(coefficient))],
    )


def _term(label: str, pauli: str) -> AnsatzTerm:
    return AnsatzTerm(label=str(label), polynomial=_poly(pauli))


def _atom(index: int, pauli: str) -> TetrisPoolAtom:
    return TetrisPoolAtom(
        pool_index=int(index),
        pauli_exyz=str(pauli),
        qubit_support=tuple(i for i, letter in enumerate(pauli) if letter != "e"),
        source_labels=("unit_pool",),
        nq=len(pauli),
    )


def _score(atom: TetrisPoolAtom, gain: float) -> TetrisCandidateScore:
    return TetrisCandidateScore(
        atom=atom,
        distance_sq=float(1.0 - gain),
        distance_sq_gain=float(gain),
        retained_rank=1,
        parameter_count=1,
    )


def test_tetris_method3_packs_disjoint_generators() -> None:
    scores = (
        _score(_atom(0, "xe"), 0.8),
        _score(_atom(1, "ex"), 0.7),
        _score(_atom(2, "xx"), 0.6),
    )

    selected = select_tetris_method3_layer(
        scores,
        min_distance_sq_gain=0.0,
    )

    assert [item.atom.pauli_exyz for item in selected] == ["xe", "ex"]


def test_tetris_singleton_layer_limit_is_original_avqds_method1() -> None:
    scores = (
        _score(_atom(0, "xe"), 0.8),
        _score(_atom(1, "ex"), 0.7),
        _score(_atom(2, "xx"), 0.6),
    )

    tetris_singleton = select_tetris_method3_layer(
        scores,
        min_distance_sq_gain=0.0,
        max_layer_width=1,
    )
    method1 = select_avqds_method1_candidate(
        scores,
        min_distance_sq_gain=0.0,
    )

    assert tetris_singleton == method1
    assert [item.atom.pauli_exyz for item in tetris_singleton] == ["xe"]


def test_projective_geometry_recovers_exact_x_velocity() -> None:
    terms = (_term("z_seed", "z"), _term("x_candidate", "x"))
    layout = _build_layout_for_terms(terms, reference_layout=build_parameter_layout(terms))
    executor = _compiled_executor_for_terms(terms, layout)
    theta = np.zeros(layout.runtime_parameter_count, dtype=float)
    psi_ref = np.asarray([1.0, 0.0], dtype=complex)
    hmat = np.asarray(hamiltonian_matrix(_poly("x")), dtype=complex)

    geometry = solve_avqds_projective_geometry(
        executor=executor,
        psi_ref=psi_ref,
        theta_runtime=theta,
        hmat=hmat,
        eigenvalue_cutoff=1.0e-6,
    )

    assert geometry.distance_sq == pytest.approx(0.0, abs=1.0e-12)
    assert geometry.theta_dot[-1] == pytest.approx(1.0, abs=1.0e-12)
    assert geometry.retained_rank == 1


def _runtime_input() -> SimpleNamespace:
    h_poly = _poly("x", 0.7)
    selected = (_term("z_seed", "z"),)
    layout = build_parameter_layout(selected)
    psi_ref = np.asarray([1.0, 0.0], dtype=complex)
    return SimpleNamespace(
        h_poly=h_poly,
        psi_ref=psi_ref,
        psi_initial=psi_ref.copy(),
        selected_terms=selected,
        candidate_pool_terms=selected + (_term("x_candidate", "x"),),
        candidate_pool_source=CandidatePoolSource(
            source_kind="resolved_pool",
            pool_key="unit_pool",
            completeness="complete",
        ),
        base_layout=layout,
        theta_runtime=np.zeros(layout.runtime_parameter_count, dtype=float),
        structure_locked=False,
        provenance={"artifact_json": "unit_seed.json"},
        resolved_problem=SimpleNamespace(
            family_key="hubbard",
            hamiltonian=h_poly,
            request=SimpleNamespace(num_sites=1, ordering="site_major"),
        ),
    )


def test_avqds_tetris_dispatch_emits_published_method3_row(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "unit_seed.json"
    artifact.write_text("{}", encoding="utf-8")
    case = DynamicsBenchmarkCase(
        case_id="unit_hubbard_tetris",
        family="hubbard",
        table_class="fermionic_lattice",
        artifact_json=str(artifact),
        t_final=0.1,
        num_times=2,
    )
    monkeypatch.setattr(
        rows_mod,
        "load_scaffold_runtime_input",
        lambda *args, **kwargs: _runtime_input(),
    )

    row = rows_mod.run_generic_dynamics_row(
        case=case,
        algorithm_id="dyn_avqds_tetris",
        output_dir=tmp_path / "avqds_tetris",
    ).to_dict()

    assert row["status"] == "completed"
    assert row["algorithm_id"] == "dyn_avqds_tetris"
    assert row["method_label"] == "AVQDS(T) TETRIS dynamics"
    assert row["table_fields"]["table_status_label"] == "AVQDS(T) Method-3 TETRIS"
    assert row["qpu_faithful"] is True
    assert row["exact_assisted"] is False
    assert row["metrics"]["method_kind"] == "avqds_tetris"
    assert row["metrics"]["tetris_layer_count"] == 1
    assert row["metrics"]["tetris_generators_added_total"] == 1
    assert row["metrics"]["unsupported_checkpoint_count"] == 0
    assert row["metrics"]["append_scoring_uses_exact_reference"] is False
    assert row["metrics"]["avqds_tetris_correctness_passed"] is True
    assert row["resources"]["measurement_model"] == "ideal_expectation_primitives_no_finite_shots"
    assert row["resources"]["shots_total"] is None
    assert row["provenance"]["runner_module"] == (
        "pipelines.time_dynamics.benchmarks.avqds_tetris"
    )
    raw = json.loads(
        (tmp_path / "avqds_tetris" / "raw_payload.json").read_text(encoding="utf-8")
    )
    assert raw["provenance"]["literature_method"] == "AVQDS(T), Method 3 TETRIS"
    assert raw["tetris_layer_events"][0]["count"] == 1
    assert raw["tetris_layer_events"][0]["pauli_terms"] == ["x"]
    assert raw["avqds_tetris_correctness"]["passed"] is True
    assert (tmp_path / "avqds_tetris" / "avqds_tetris_correctness.json").exists()


def test_avqds_tetris_uses_and_reconstructs_shared_redundancy_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "unit_redundant_seed.json"
    artifact.write_text("{}", encoding="utf-8")
    case = DynamicsBenchmarkCase(
        case_id="unit_hubbard_tetris_redundancy",
        family="hubbard",
        table_class="fermionic_lattice",
        artifact_json=str(artifact),
        t_final=0.1,
        num_times=2,
        metadata={
            "diagnostic_redundancy_layer_count": 2,
            "diagnostic_redundancy_pool_profile": "hamiltonian_drive_pauli",
        },
    )
    monkeypatch.setattr(
        rows_mod,
        "load_scaffold_runtime_input",
        lambda *args, **kwargs: _runtime_input(),
    )

    row = rows_mod.run_generic_dynamics_row(
        case=case,
        algorithm_id="dyn_avqds_tetris",
        output_dir=tmp_path / "avqds_tetris_redundancy",
    ).to_dict()
    raw = json.loads(
        (tmp_path / "avqds_tetris_redundancy" / "raw_payload.json").read_text(
            encoding="utf-8"
        )
    )

    receipt = raw["diagnostic_redundancy_stress"]
    assert row["status"] == "completed"
    assert receipt["prepared_state_parity_passed"] is True
    assert receipt["layer_count"] == 2
    assert receipt["pool_atom_count"] == 1
    assert receipt["appended_coordinate_count"] == 2
    assert raw["trajectory"][0]["runtime_parameter_count"] == 3

    monkeypatch.setattr(
        report_mod,
        "load_scaffold_runtime_input",
        lambda *args, **kwargs: _runtime_input(),
    )
    reconstructed = report_mod.reconstruct_terminal_avqds(raw)
    assert reconstructed.parity["passed"] is True
    assert reconstructed.layout.runtime_parameter_count == 3
    assert reconstructed.diagnostic_redundancy_stress["applied"] is True


def test_registry_distinguishes_tetris_from_pf_target_diagnostic() -> None:
    assert benchmark_registry.runner_module_for_algorithm("dyn_avqds_tetris") == (
        "pipelines.time_dynamics.benchmarks.avqds_tetris"
    )
    assert benchmark_registry.runner_module_for_algorithm("dyn_avqds_t") == (
        "pipelines.time_dynamics.benchmarks.legacy_native"
    )
