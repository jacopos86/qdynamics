"""Focused coverage for generic-static comparator scaling support."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chtc.phase3_optuna import run_paper_i_hh_spsa_budget_ladder_cell as cell_runner
from pipelines.exact_bench import generic_static_adapt_variants as variants
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm


class _TinyLayout:
    total_qubits = 2
    fermion_qubits = 2

    def block(self, name: str):  # noqa: ANN201 - intentionally tiny test double
        if name == "fermion":
            return SimpleNamespace(start_qubit=0, stop_qubit=2)
        return None


def _hubbard_spec() -> SimpleNamespace:
    return SimpleNamespace(
        benchmark_id="hubbard_L2",
        family="hubbard",
        base_pipeline_args=("--problem", "hubbard", "--L", "2"),
        split="paper_i_scaling_matrix_20260710_v1",
        tags=(),
        features=None,
    )


def _hubbard_context() -> SimpleNamespace:
    hamiltonian = PauliPolynomial(
        "JW",
        [
            PauliTerm(2, ps="xx", pc=0.5),
            PauliTerm(2, ps="yy", pc=0.5),
        ],
    )
    return SimpleNamespace(
        request=SimpleNamespace(
            problem_key="hubbard",
            num_sites=2,
            t=1.0,
            u=0.25,
            dv=0.0,
            omega0=1.0,
            g_ep=0.0,
            n_ph_max=0,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
            n_fermions=2,
        ),
        layout=_TinyLayout(),
        hamiltonian=hamiltonian,
        reference_state=SimpleNamespace(build_state=lambda: np.eye(4, dtype=complex)[1]),
        exact_target=SimpleNamespace(resolve_energy=lambda ai_log=None: -1.0),
        sector=SimpleNamespace(constraints=()),
    )


def _sector_probability(_context, _psi):  # noqa: ANN001, ANN202 - test double
    return {
        "sector_probability": 1.0,
        "sector_leak_probability": 0.0,
        "sector_leak_flag": False,
        "sector_leak_threshold": 1e-8,
        "boson_legal_probability_min": None,
        "boson_illegal_probability_max": None,
        "boson_truncation_leak_flag": False,
        "boson_subspace_diagnostics": None,
        "truncation_constraints_evaluated": [],
    }


def test_exact_fidelity_env_cap_skips_dense_diagonalization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GENERIC_STATIC_TABLE_EXACT_FIDELITY_MAX_QUBITS", "10")
    cap = variants._exact_fidelity_max_qubits_from_env()

    def _dense_action_must_not_run(*_args, **_kwargs):  # noqa: ANN202
        raise AssertionError("dense Hamiltonian construction ran above the configured qubit cap")

    monkeypatch.setattr(variants, "apply_compiled_polynomial", _dense_action_must_not_run)
    fields = variants._dense_exact_state_fidelity_for_selected(
        context=SimpleNamespace(),
        selected=(),
        theta=np.asarray([], dtype=float),
        psi_ref=np.asarray([1.0], dtype=complex),
        h_compiled=SimpleNamespace(nq=12),
        pauli_action_cache={},
        exact_energy=None,
        max_qubits=cap,
    )

    assert cap == 10
    assert fields == {
        "infidelity_exact": None,
        "exact_state_fidelity": None,
        "infidelity_status": "not_available_dense_diagonalization_qubit_cap",
        "exact_state_fidelity_source": "dense_diagonalization_skipped",
        "exact_state_fidelity_qubit_cap": 10,
        "exact_state_fidelity_s_alg_charged": False,
        "ground_space_fidelity": {
            "schema": "ground_space_projector_fidelity_v1",
            "status": "not_available_dense_diagonalization_qubit_cap",
            "usage_scope": "post_run_reporting_only",
            "controller_decision_eligible": False,
            "optimizer_input_eligible": False,
            "stopping_input_eligible": False,
            "s_alg_charged": False,
            "qubit_count": 12,
            "qubit_cap": 10,
        },
    }


def test_append_geo_cell_env_forwards_exact_fidelity_qubit_cap(tmp_path: Path) -> None:
    row = {
        "method_key": "geo",
        "suite_profile": "paper_i_scaling_matrix_20260710_v1",
        "adapt_optimizer_kind": "powell",
        "max_depth": "50",
        "budget": "200",
        "same_cutoff_exact_gs_energy": "-1.0",
        "exact_reference_energy": "-1.0",
        "exact_reference_n_ph_max": "2",
        "exact_fidelity_max_qubits": "10",
    }

    env = cell_runner.append_geo_env(row, tmp_path / "out")

    assert env["GENERIC_STATIC_TABLE_EXACT_FIDELITY_MAX_QUBITS"] == "10"


@pytest.mark.parametrize(
    "algorithm_id",
    ["static_geo_adapt_vqe", "static_full_meta_append_adapt_vqe"],
)
def test_non_hh_comparators_emit_replay_shaped_runtime_seed_sidecar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    algorithm_id: str,
) -> None:
    def _fake_minimize(objective, x0, method=None, options=None):  # noqa: ANN001, ANN003, ANN201
        del method, options
        x = np.asarray(x0, dtype=float).reshape(-1) + 0.1
        return SimpleNamespace(x=x, fun=float(objective(x)), nfev=2, nit=1, success=True, message="ok")

    monkeypatch.setattr(variants, "has_scipy_minimize_support", lambda: True)
    monkeypatch.setattr(variants, "_import_scipy_minimize", lambda: _fake_minimize)
    monkeypatch.setattr(variants, "_spec_by_case_id", lambda family, case_id, algorithm_id: _hubbard_spec())
    monkeypatch.setattr(variants, "_resolve_context_from_spec", lambda spec: _hubbard_context())
    monkeypatch.setattr(
        variants,
        "build_full_meta_candidate_pool",
        lambda context, *, max_terms=variants._POOL_TERM_CAP: variants.build_pairwise_qubit_excitation_pool(
            context.layout.total_qubits,
            max_terms=max_terms,
        ),
    )
    monkeypatch.setattr(variants, "sector_probability", _sector_probability)

    output_dir = tmp_path / algorithm_id
    payload = variants.run_generic_static_adapt_variant_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id=algorithm_id,
        output_dir=output_dir,
        max_adapt_iterations=1,
        optimizer_maxiter=5,
        gradient_threshold=0.0,
        adapt_optimizer_kind="powell",
        optimizer_overlay_source="test",
    )

    runtime_seed_path = output_dir / "runtime_seed.json"
    assert runtime_seed_path.exists()
    assert payload["runtime_seed_json"] == str(runtime_seed_path)
    runtime_seed = json.loads(runtime_seed_path.read_text(encoding="utf-8"))
    assert runtime_seed["family"] == "hubbard"
    assert runtime_seed["settings"]["problem"] == "hubbard"
    assert runtime_seed["algorithm_id"] == algorithm_id
    assert runtime_seed["adapt_vqe"]["algorithm_id"] == algorithm_id
    assert runtime_seed["adapt_vqe"]["operators"]
    assert runtime_seed["adapt_vqe"]["selected_generator_semantics_sha256"]
    assert runtime_seed["ansatz_input_state"]["handoff_state_kind"] == "reference_state"
    assert runtime_seed["initial_state"]["handoff_state_kind"] == "prepared_state"
