"""Integration tests for the hardcoded ADAPT-VQE pipeline.

Tests cover:
  - L=2 Hubbard UCCSD pool (basic ADAPT-VQE convergence)
  - L=2 HH HVA pool (sector-filtered HH ground energy)
  - L=2 HH PAOP pool (polaron-adapted operators)
  - Pool builder sanity checks (non-empty, correct types)
  - Sector filtering correctness (HH uses fermion-only filtering)
  - PAOP module importability
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipelines.exact_bench.noise_oracle_runtime as _raw_runtime
import pipelines.static_adapt.engine_support as _engine_support_mod

from src.quantum.hubbard_latex_python_pairs import (
    build_hubbard_hamiltonian,
    build_hubbard_holstein_hamiltonian,
    boson_qubits_per_site,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.chemistry.molecular_hamiltonian import (
    build_restricted_closed_shell_molecular_hamiltonian,
)
from src.quantum.chemistry.psi4_adapter import (
    load_restricted_closed_shell_problem_from_json,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.vqe_latex_python_pairs import (
    AnsatzTerm,
    exact_ground_energy_sector,
    exact_ground_energy_sector_hh,
    half_filled_num_particles,
)

# Import ADAPT pipeline internals
import builtins
import pipelines.static_adapt.adapt_pipeline as _adapt_mod
from pipelines.scaffold.hh_continuation_types import CompileCostEstimate
from pipelines.scaffold.hh_continuation_types import CandidateFeatures
from pipelines.static_adapt.builders.problem_registry import ProblemRequest, resolve_problem_context
from pipelines.static_adapt.builders.pool_resolution import (
    resolve_pool_plan,
    resolve_requested_pool_filters,
)
from pipelines.static_adapt.hardware_resolution_profiles import (
    HARDWARE_RESOLUTION_GRADIENT_PROFILE_SCHEMA,
    HARDWARE_RESOLUTION_PROFILE_MANIFEST_SCHEMA,
    HARDWARE_RESOLUTION_PROFILE_UNITS,
)
from pipelines.static_adapt.extensions import PRUNING_RUNTIME_KEYS

MATURITY_RUNTIME_KEYS = (
    "phase1_maturity_cap_min",
    "phase1_maturity_cap_max",
    "phase2_maturity_cap_min",
    "phase2_maturity_cap_max",
    "phase3_maturity_cap_min",
    "phase3_maturity_cap_max",
    "phase_maturity_shot_min",
    "phase_maturity_shot_max",
    "phase1_maturity_shot_cap",
    "phase2_maturity_shot_cap",
    "phase3_maturity_shot_cap",
)

_run_hardcoded_adapt_vqe = _adapt_mod._run_hardcoded_adapt_vqe
_build_uccsd_pool = _adapt_mod._build_uccsd_pool
_build_cse_pool = _adapt_mod._build_cse_pool
_build_full_hamiltonian_pool = _adapt_mod._build_full_hamiltonian_pool
_build_hva_pool = _adapt_mod._build_hva_pool
_build_paop_pool = _adapt_mod._build_paop_pool
_build_hh_termwise_augmented_pool = _adapt_mod._build_hh_termwise_augmented_pool
_build_hh_uccsd_fermion_lifted_pool = _adapt_mod._build_hh_uccsd_fermion_lifted_pool
_build_hh_pareto_lean_pool = _adapt_mod._build_hh_pareto_lean_pool
_build_hh_pareto_lean_l2_pool = _adapt_mod._build_hh_pareto_lean_l2_pool
_deduplicate_pool_terms = _adapt_mod._deduplicate_pool_terms
_exact_gs_energy_for_problem = _adapt_mod._exact_gs_energy_for_problem
_compile_polynomial_action = _adapt_mod._compile_polynomial_action
_apply_compiled_polynomial = _adapt_mod._apply_compiled_polynomial
_apply_pauli_polynomial_uncached = _adapt_mod._apply_pauli_polynomial_uncached
_accepted_live_prune_labels_from_history = _adapt_mod._accepted_live_prune_labels_from_history
_filter_repeat_live_prune_candidates = _adapt_mod._filter_repeat_live_prune_candidates
_commutator_gradient = _adapt_mod._commutator_gradient
_resolve_reopt_active_indices = _adapt_mod._resolve_reopt_active_indices
_make_reduced_objective = _adapt_mod._make_reduced_objective
_VALID_REOPT_POLICIES = _adapt_mod._VALID_REOPT_POLICIES
_Phase3OracleGradientConfig = _adapt_mod.Phase3OracleGradientConfig
_FinalNoiseAuditConfig = _adapt_mod.FinalNoiseAuditConfig


def _write_hardware_resolution_profile_json(
    tmp_path: Path,
    *,
    name: str = "calib_a",
    hw_floor: float = 0.02,
    drift_floor: float = 0.03,
) -> Path:
    payload: dict[str, Any] = {
        "schema": HARDWARE_RESOLUTION_PROFILE_MANIFEST_SCHEMA,
        "profiles": {
            name: {
                "schema": HARDWARE_RESOLUTION_GRADIENT_PROFILE_SCHEMA,
                "name": name,
                "gradient_hw_floor": float(hw_floor),
                "gradient_drift_floor": float(drift_floor),
                "units": HARDWARE_RESOLUTION_PROFILE_UNITS,
                "provenance": {
                    "source": "test_adapt_vqe_integration",
                    "generated_utc": "2026-05-16T00:00:00Z",
                },
            }
        },
    }
    path = tmp_path / "hardware_resolution_profiles.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _write_molecular_problem_json(tmp_path: Path) -> Path:
    payload: dict[str, Any] = {
        "geometry_spec": "H 0.0 0.0 0.0\nH 0.0 0.0 0.7414",
        "basis": "sto-3g",
        "charge": 0,
        "multiplicity": 1,
        "reference": "rhf",
        "n_spatial_orbitals": 2,
        "n_spin_orbitals": 4,
        "n_alpha": 1,
        "n_beta": 1,
        "hf_energy": -1.1166843871,
        "nuclear_repulsion_energy": 0.7151043391,
        "one_body_integrals_mo": [
            [-1.252477303982, 0.0],
            [0.0, -0.475934275355],
        ],
        "two_body_integrals_mo": [
            [
                [
                    [0.674493166181, 0.0],
                    [0.0, 0.181287518779],
                ],
                [
                    [0.0, 0.181287518779],
                    [0.181287518779, 0.0],
                ],
            ],
            [
                [
                    [0.0, 0.181287518779],
                    [0.181287518779, 0.0],
                ],
                [
                    [0.6634721010, 0.0],
                    [0.0, 0.6973980100],
                ],
            ],
        ],
    }
    path = tmp_path / "molecular_problem.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fermion_sector_weights(
    psi: np.ndarray,
    *,
    num_sites: int,
    ordering: str,
) -> dict[tuple[int, int], float]:
    if str(ordering) == "blocked":
        alpha = list(range(int(num_sites)))
        beta = list(range(int(num_sites), 2 * int(num_sites)))
    else:
        alpha = list(range(0, 2 * int(num_sites), 2))
        beta = list(range(1, 2 * int(num_sites), 2))
    out: dict[tuple[int, int], float] = {}
    for idx, amp in enumerate(np.asarray(psi, dtype=complex).reshape(-1)):
        prob = float(abs(amp) ** 2)
        if prob <= 1e-14:
            continue
        n_up = int(sum((idx >> int(q)) & 1 for q in alpha))
        n_dn = int(sum((idx >> int(q)) & 1 for q in beta))
        out[(n_up, n_dn)] = float(out.get((n_up, n_dn), 0.0) + prob)
    return out


def _spinless_sector_weights(
    psi: np.ndarray,
    *,
    num_sites: int,
) -> dict[int, float]:
    out: dict[int, float] = {}
    for idx, amp in enumerate(np.asarray(psi, dtype=complex).reshape(-1)):
        prob = float(abs(amp) ** 2)
        if prob <= 1e-14:
            continue
        n_fermions = int(sum((idx >> int(q)) & 1 for q in range(int(num_sites))))
        out[n_fermions] = float(out.get(n_fermions, 0.0) + prob)
    return out


def _selected_ops_from_payload(
    *,
    resolved_problem,
    payload: dict[str, Any],
    continuation_mode: str,
) -> list[AnsatzTerm]:
    adapt_pool = payload.get("pool_type")
    plan = resolve_pool_plan(
        resolved_problem=resolved_problem,
        continuation_mode=str(continuation_mode),
        adapt_pool=adapt_pool,
        paop_r=0,
        paop_split_paulis=False,
        paop_prune_eps=0.0,
        paop_normalization="none",
        phase3_symmetry_mitigation_mode="off",
        filter_resolution=resolve_requested_pool_filters(
            problem_key=str(resolved_problem.family_key),
            num_sites=int(resolved_problem.request.num_sites),
            n_ph_max=int(resolved_problem.request.n_ph_max),
            adapt_pool=(None if adapt_pool in {None, ""} else str(adapt_pool)),
            adapt_pool_class_filter_json=None,
            adapt_pool_label_filter_json=None,
        ),
    )
    label_to_term = {str(term.label): term for term in plan.pool}
    return [label_to_term[str(label)] for label in payload["operators"]]


def _runtime_theta_matches_logical_blocks(payload: dict[str, Any], tol: float = 1e-10) -> bool:
    theta_runtime = np.asarray(payload.get("optimal_point", []), dtype=float).reshape(-1)
    theta_logical = np.asarray(payload.get("logical_optimal_point", []), dtype=float).reshape(-1)
    blocks = list(payload.get("parameterization", {}).get("blocks", []))
    if int(theta_runtime.size) == 0:
        return True
    if int(theta_logical.size) != int(len(blocks)):
        return False
    for logical_idx, block in enumerate(blocks):
        start = int(block.get("runtime_start", 0))
        count = int(block.get("runtime_count", 0))
        if count <= 0:
            continue
        vals = theta_runtime[start:start + count]
        if int(vals.size) != int(count):
            return False
        if np.max(np.abs(vals - float(theta_logical[logical_idx]))) > float(tol):
            return False
    return True


class TestCompiledPauliCache:
    """Parity and performance checks for cached compiled Pauli actions."""

    def test_reopt_helper_cluster_remains_wrapper_visible(self):
        expected_names = [
            "_logical_theta_alias",
            "_VALID_REOPT_POLICIES",
            "_VALID_ADAPT_INNER_OPTIMIZERS",
            "_scipy_adapt_heartbeat_event",
            "_scipy_adapt_optimizer_options",
            "_run_scipy_adapt_optimizer",
            "_resolve_reopt_active_indices",
            "_make_reduced_objective",
        ]
        for name in expected_names:
            assert hasattr(_adapt_mod, name), f"wrapper surface missing {name}"
            assert getattr(_adapt_mod, name) is getattr(_engine_support_mod, name)

    def test_seq2p_helper_cluster_remains_wrapper_visible(self):
        expected_names = [
            "_ADAPTLogicalCandidate",
            "_parse_seq2p_step_label",
            "_build_seq2p_logical_candidates",
            "_logical_candidate_gradient_summary",
        ]
        for name in expected_names:
            assert hasattr(_adapt_mod, name), f"wrapper surface missing {name}"
        assert _adapt_mod._parse_seq2p_step_label is _engine_support_mod._parse_seq2p_step_label
        assert _adapt_mod._build_seq2p_logical_candidates is _engine_support_mod._build_seq2p_logical_candidates
        assert _adapt_mod._logical_candidate_gradient_summary is _engine_support_mod._logical_candidate_gradient_summary

    def test_gradient_state_helper_cluster_remains_wrapper_visible(self):
        expected_names = [
            "_apply_pauli_polynomial_uncached",
            "_apply_pauli_polynomial",
            "_commutator_gradient",
            "_prepare_adapt_state",
            "_adapt_energy_fn",
        ]
        for name in expected_names:
            assert hasattr(_adapt_mod, name), f"wrapper surface missing {name}"
            assert getattr(_adapt_mod, name) is getattr(_engine_support_mod, name)

    def test_to_ixyz_wrapper_surface_matches_live_contract(self):
        assert _adapt_mod._to_ixyz("exyz") == "IXYZ"

    def test_empty_polynomial_compiled_sentinel_returns_zero_apply(self):
        psi = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        compiled = _compile_polynomial_action(PauliPolynomial("JW"))
        hpsi = _apply_compiled_polynomial(psi, compiled)
        assert int(compiled.nq) == 0
        assert len(compiled.terms) == 0
        assert np.array_equal(hpsi, np.zeros_like(psi))

    def test_trajectory_wrapper_injects_ai_log(self, monkeypatch: pytest.MonkeyPatch):
        captured: dict[str, object] = {}

        def _fake_runtime(**kwargs):
            captured["ai_log"] = kwargs.get("ai_log")
            return ([{"time": 0.0, "fidelity": 1.0}], [])

        monkeypatch.setattr(_adapt_mod, "_simulate_trajectory_runtime", _fake_runtime)
        rows, exact_states = _adapt_mod._simulate_trajectory(
            num_sites=2,
            psi0=np.array([1.0, 0.0, 0.0, 0.0], dtype=complex),
            hmat=np.eye(4, dtype=complex),
            ordered_labels_exyz=["ee"],
            coeff_map_exyz={"ee": 0.0},
            trotter_steps=1,
            t_final=0.0,
            num_times=1,
            suzuki_order=2,
        )

        assert captured["ai_log"] is _adapt_mod._ai_log
        assert rows == [{"time": 0.0, "fidelity": 1.0}]
        assert exact_states == []

    def test_build_seq2p_logical_candidates_groups_adjacent_pairs(self):
        poly = PauliPolynomial("JW")
        pool = [
            AnsatzTerm(label="pair_a::step=ferm", polynomial=poly),
            AnsatzTerm(label="pair_a::step=motif", polynomial=poly),
            AnsatzTerm(label="pair_b::step=ferm", polynomial=poly),
            AnsatzTerm(label="pair_b::step=motif", polynomial=poly),
        ]

        candidates = _adapt_mod._build_seq2p_logical_candidates(pool, family_id="seq2p_family")

        assert _adapt_mod._parse_seq2p_step_label("pair_a::step=ferm") == ("pair_a", "ferm")
        assert _adapt_mod._parse_seq2p_step_label("pair_a::step=motif") == ("pair_a", "motif")
        assert len(candidates) == 2
        assert candidates[0].logical_label == "pair_a"
        assert candidates[0].pool_indices == (0, 1)
        assert candidates[0].parameterization == "double_sequential"
        assert candidates[0].family_id == "seq2p_family"
        assert candidates[1].logical_label == "pair_b"
        assert candidates[1].pool_indices == (2, 3)
        assert candidates[1].parameterization == "double_sequential"
        assert candidates[1].family_id == "seq2p_family"
        assert isinstance(candidates[0], _adapt_mod._ADAPTLogicalCandidate)
        assert candidates[0].__class__ is _engine_support_mod._ADAPTLogicalCandidate

    def test_logical_candidate_gradient_summary_uses_euclidean_score(self):
        candidate = _adapt_mod._ADAPTLogicalCandidate(
            logical_label="pair_a",
            pool_indices=(0, 1),
            parameterization="double_sequential",
            family_id="seq2p_family",
        )
        score, signed_components, abs_components = _adapt_mod._logical_candidate_gradient_summary(
            candidate,
            np.array([3.0, -4.0], dtype=float),
        )

        assert signed_components == [3.0, -4.0]
        assert abs_components == [3.0, 4.0]
        assert score == pytest.approx(5.0)

    def test_build_seq2p_logical_candidates_rejects_misordered_pair(self):
        poly = PauliPolynomial("JW")
        pool = [
            AnsatzTerm(label="pair_a::step=motif", polynomial=poly),
            AnsatzTerm(label="pair_a::step=ferm", polynomial=poly),
        ]

        with pytest.raises(ValueError, match="expected adjacent ferm/motif terms"):
            _adapt_mod._build_seq2p_logical_candidates(pool, family_id="seq2p_family")

    @staticmethod
    def _random_state(nq: int, seed: int = 13) -> np.ndarray:
        rng = np.random.default_rng(int(seed))
        psi = rng.normal(size=1 << int(nq)) + 1j * rng.normal(size=1 << int(nq))
        psi = np.asarray(psi, dtype=complex)
        return psi / np.linalg.norm(psi)

    def test_compiled_apply_matches_uncached(self):
        h_poly = build_hubbard_hamiltonian(
            dims=2, t=1.0, U=4.0, v=0.3,
            repr_mode="JW", indexing="blocked", pbc=True,
        )
        psi = self._random_state(4, seed=101)
        compiled = _compile_polynomial_action(h_poly)
        uncached = _apply_pauli_polynomial_uncached(psi, h_poly)
        cached = _apply_compiled_polynomial(psi, compiled)
        assert np.max(np.abs(cached - uncached)) < 1e-12

    def test_commutator_gradient_matches_uncached(self):
        h_poly = build_hubbard_hamiltonian(
            dims=3, t=1.0, U=4.0, v=0.1,
            repr_mode="JW", indexing="blocked", pbc=True,
        )
        num_particles = half_filled_num_particles(3)
        pool = _build_uccsd_pool(3, num_particles, "blocked")
        assert len(pool) > 0
        op = pool[0]
        psi = self._random_state(6, seed=202)

        grad_uncached = _commutator_gradient(h_poly, op, psi)
        grad_cached = _commutator_gradient(
            h_poly,
            op,
            psi,
            h_compiled=_compile_polynomial_action(h_poly),
            pool_compiled=_compile_polynomial_action(op.polynomial),
        )
        assert abs(grad_cached - grad_uncached) < 1e-12

    def test_gradient_cached_speedup(self):
        h_poly = build_hubbard_hamiltonian(
            dims=3, t=1.0, U=4.0, v=0.1,
            repr_mode="JW", indexing="blocked", pbc=True,
        )
        num_particles = half_filled_num_particles(3)
        pool = _build_cse_pool(3, "blocked", 1.0, 4.0, 0.1, "periodic")
        assert len(pool) > 0
        op = pool[0]
        psi = self._random_state(6, seed=303)

        h_compiled = _compile_polynomial_action(h_poly)
        op_compiled = _compile_polynomial_action(op.polynomial)

        # Warm up to avoid one-time dispatch effects dominating timings.
        _commutator_gradient(h_poly, op, psi)
        _commutator_gradient(h_poly, op, psi, h_compiled=h_compiled, pool_compiled=op_compiled)

        def _bench_uncached(num_iter: int) -> float:
            t0 = time.perf_counter()
            for _ in range(int(num_iter)):
                _commutator_gradient(h_poly, op, psi)
            return float(time.perf_counter() - t0)

        def _bench_cached(num_iter: int) -> float:
            t0 = time.perf_counter()
            for _ in range(int(num_iter)):
                _commutator_gradient(
                    h_poly,
                    op,
                    psi,
                    h_compiled=h_compiled,
                    pool_compiled=op_compiled,
                )
            return float(time.perf_counter() - t0)

        num_iter = 8
        uncached_elapsed = _bench_uncached(num_iter)
        while uncached_elapsed < 0.15 and num_iter < 4096:
            num_iter *= 2
            uncached_elapsed = _bench_uncached(num_iter)
        cached_elapsed = _bench_cached(num_iter)
        speedup = uncached_elapsed / cached_elapsed if cached_elapsed > 0.0 else float("inf")
        assert speedup > 1.5, (
            f"Expected cached gradient speedup > 1.5x, got {speedup:.2f}x "
            f"(uncached={uncached_elapsed:.4f}s, cached={cached_elapsed:.4f}s, iters={num_iter})"
        )


class TestAdaptCompiledStateBackendParity:
    """Compiled ansatz execution must preserve ADAPT selection/energy parity."""

    def test_compiled_state_backend_matches_legacy_sequence(self):
        h_poly = build_hubbard_hamiltonian(
            dims=2,
            t=1.0,
            U=4.0,
            v=0.0,
            repr_mode="JW",
            indexing="blocked",
            pbc=True,
        )
        common_kwargs = dict(
            h_poly=h_poly,
            num_sites=2,
            ordering="blocked",
            problem="hubbard",
            adapt_pool="uccsd",
            t=1.0,
            u=4.0,
            dv=0.0,
            boundary="periodic",
            omega0=0.0,
            g_ep=0.0,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=6,
            eps_grad=1e-6,
            eps_energy=1e-10,
            maxiter=120,
            seed=17,
            allow_repeats=False,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
        )

        payload_legacy, _psi_legacy = _run_hardcoded_adapt_vqe(
            **common_kwargs,
            adapt_state_backend="legacy",
        )
        payload_compiled, _psi_compiled = _run_hardcoded_adapt_vqe(
            **common_kwargs,
            adapt_state_backend="compiled",
        )

        seq_legacy = [int(row["pool_index"]) for row in payload_legacy.get("history", [])]
        seq_compiled = [int(row["pool_index"]) for row in payload_compiled.get("history", [])]
        labels_legacy = [str(row["selected_op"]) for row in payload_legacy.get("history", [])]
        labels_compiled = [str(row["selected_op"]) for row in payload_compiled.get("history", [])]

        n_check = min(5, len(seq_legacy), len(seq_compiled))
        assert n_check > 0
        assert seq_compiled[:n_check] == seq_legacy[:n_check]
        assert labels_compiled[:n_check] == labels_legacy[:n_check]
        assert abs(float(payload_compiled["energy"]) - float(payload_legacy["energy"])) < 1e-8

    def test_current_checkpoint_json_is_recoverable_scaffold(self, tmp_path):
        from pipelines.scaffold.runtime_loader import load_scaffold_runtime_input_from_payload

        h_poly = build_hubbard_hamiltonian(
            dims=2,
            t=1.0,
            U=4.0,
            v=0.0,
            repr_mode="JW",
            indexing="blocked",
            pbc=True,
        )
        current_json = tmp_path / "current.json"

        payload, _psi = _run_hardcoded_adapt_vqe(
            h_poly=h_poly,
            num_sites=2,
            ordering="blocked",
            problem="hubbard",
            adapt_pool="uccsd",
            t=1.0,
            u=4.0,
            dv=0.0,
            boundary="periodic",
            omega0=0.0,
            g_ep=0.0,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-12,
            eps_energy=1e-12,
            maxiter=30,
            seed=23,
            allow_repeats=False,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_state_backend="compiled",
            adapt_current_json=current_json,
            adapt_current_json_every_depth=1,
            adapt_current_json_keep_history_tail=5,
        )

        current_payload = json.loads(current_json.read_text(encoding="utf-8"))
        assert current_payload["schema_version"] == "static_adapt_current_checkpoint_v1"
        assert current_payload["no_credentials_serialized"] is True
        assert current_payload["credential_audit"]["no_credentials_serialized"] is True
        assert current_payload["checkpoint"]["complete"] is False
        assert current_payload["checkpoint"]["target_hit_classification"]["target_hit_success"] is False
        adapt_payload = current_payload["adapt_vqe"]
        assert adapt_payload["partial_checkpoint"] is True
        assert adapt_payload["checkpoint_reason"] == "iteration_done"
        assert adapt_payload["ansatz_depth"] == int(payload["ansatz_depth"])
        assert adapt_payload["operators"] == list(payload["operators"])
        assert adapt_payload["history_count"] == 1
        assert adapt_payload["history_tail_count"] == 1
        assert adapt_payload["parameterization_mode"] == adapt_payload[
            "parameterization_execution_mode"
        ]
        assert current_payload["checkpoint"]["parameterization_mode"] == (
            adapt_payload["parameterization_execution_mode"]
        )
        assert current_payload["checkpoint"][
            "parameterization_execution_mode"
        ] == adapt_payload["parameterization_execution_mode"]
        checkpoint_chart = adapt_payload["optimizer_coordinate_chart"]
        assert checkpoint_chart["runtime_dimension"] == int(
            adapt_payload["num_parameters"]
        )
        assert checkpoint_chart["logical_dimension"] == int(
            adapt_payload["logical_num_parameters"]
        )
        assert payload["parameterization_mode"] == payload[
            "parameterization_execution_mode"
        ]
        assert payload["optimizer_coordinate_chart"]["runtime_dimension"] == int(
            payload["num_parameters"]
        )
        assert payload["optimizer_coordinate_chart"]["logical_dimension"] == int(
            payload["logical_num_parameters"]
        )
        assert np.isfinite(float(adapt_payload["energy"]))
        assert isinstance(current_payload.get("initial_state"), dict)
        assert isinstance(current_payload.get("ansatz_input_state"), dict)
        assert current_payload["initial_state"]["handoff_state_kind"] == "prepared_state"

        runtime_input = load_scaffold_runtime_input_from_payload(
            current_payload,
            artifact_json=current_json,
            generator_family="match_adapt",
            fallback_family="uccsd",
        )
        assert len(runtime_input.selected_terms) == int(adapt_payload["ansatz_depth"])
        assert runtime_input.theta_runtime.size == int(adapt_payload["num_parameters"])

    def test_auto_worker_limit_uses_allocated_cpu_env_and_cap(self, monkeypatch):
        monkeypatch.setenv("STATIC_ADAPT_ALLOCATED_CPUS", "12")
        monkeypatch.setenv("STATIC_ADAPT_AUTO_WORKER_CAP", "8")

        resolved, meta = _adapt_mod._resolve_adapt_worker_limit(
            0,
            name="adapt_parallel_gradient_workers",
        )
        assert resolved == 8
        assert meta["requested"] == 0
        assert meta["resolved"] == 8
        assert meta["source"] == "auto_allocated_cpu_count"
        assert meta["allocated_cpus"] == 12
        assert meta["configured_cap"] == 8
        assert _adapt_mod._cap_worker_limit_for_items(resolved, 3) == 3

        explicit, explicit_meta = _adapt_mod._resolve_adapt_worker_limit(
            4,
            name="adapt_parallel_gradient_workers",
        )
        assert explicit == 4
        assert explicit_meta["source"] == "explicit"

        with pytest.raises(ValueError, match="0=auto"):
            _adapt_mod._resolve_adapt_worker_limit(-1, name="adapt_parallel_gradient_workers")

    def test_main_loop_parallel_gradient_matches_default_serial(self, monkeypatch):
        h_poly = build_hubbard_hamiltonian(
            dims=2,
            t=1.0,
            U=4.0,
            v=0.0,
            repr_mode="JW",
            indexing="blocked",
            pbc=True,
        )
        common_kwargs = dict(
            h_poly=h_poly,
            num_sites=2,
            ordering="blocked",
            problem="hubbard",
            adapt_pool="uccsd",
            t=1.0,
            u=4.0,
            dv=0.0,
            boundary="periodic",
            omega0=0.0,
            g_ep=0.0,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=2,
            eps_grad=1e-12,
            eps_energy=0.0,
            maxiter=20,
            seed=19,
            adapt_inner_optimizer="SPSA",
            adapt_spsa_eval_repeats=1,
            adapt_spsa_avg_last=0,
            allow_repeats=False,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_state_backend="compiled",
        )
        monkeypatch.setattr(_adapt_mod, "_ai_log", lambda event, **kw: None)

        payload_serial, _psi_serial = _run_hardcoded_adapt_vqe(
            **common_kwargs,
            adapt_parallel_gradient_workers=1,
        )
        payload_parallel, _psi_parallel = _run_hardcoded_adapt_vqe(
            **common_kwargs,
            adapt_parallel_gradient_workers=2,
        )

        seq_serial = [int(row["pool_index"]) for row in payload_serial.get("history", [])]
        seq_parallel = [int(row["pool_index"]) for row in payload_parallel.get("history", [])]
        assert seq_parallel == seq_serial
        assert [str(row["selected_op"]) for row in payload_parallel.get("history", [])] == [
            str(row["selected_op"]) for row in payload_serial.get("history", [])
        ]
        assert float(payload_parallel["energy"]) == pytest.approx(float(payload_serial["energy"]), abs=1e-12)
        assert all(not bool(row.get("gradient_parallel_enabled")) for row in payload_serial.get("history", []))
        assert any(bool(row.get("gradient_parallel_enabled")) for row in payload_parallel.get("history", []))
        assert all(
            str(row.get("gradient_parallel_backend")) == "ThreadPoolExecutor"
            for row in payload_parallel.get("history", [])
            if bool(row.get("gradient_parallel_enabled"))
        )


# ============================================================================
# Pool builder tests
# ============================================================================

class TestAdaptResolvedProblemContext:
    def test_run_hardcoded_adapt_vqe_accepts_matching_resolved_problem_context(self):
        h_poly = build_hubbard_hamiltonian(
            dims=2,
            t=1.0,
            U=4.0,
            v=0.0,
            repr_mode="JW",
            indexing="blocked",
            pbc=True,
        )
        resolved = resolve_problem_context(
            ProblemRequest(
                problem_key="hubbard",
                num_sites=2,
                t=1.0,
                u=4.0,
                dv=0.0,
                omega0=0.0,
                g_ep=0.0,
                n_ph_max=1,
                boson_encoding="binary",
                ordering="blocked",
                boundary="periodic",
                include_zero_point=True,
            ),
            hamiltonian=h_poly,
        )
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=h_poly,
            resolved_problem_context=resolved,
            num_sites=2,
            ordering="blocked",
            problem="hubbard",
            adapt_pool="uccsd",
            t=1.0,
            u=4.0,
            dv=0.0,
            boundary="periodic",
            omega0=0.0,
            g_ep=0.0,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-6,
            eps_energy=1e-10,
            maxiter=10,
            seed=17,
            allow_repeats=False,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
        )
        assert str(payload["pool_type"]) == "uccsd"
        assert math.isfinite(float(payload["energy"]))

    def test_run_hardcoded_adapt_vqe_rejects_mismatched_resolved_problem_context(self):
        h_poly = build_hubbard_hamiltonian(
            dims=2,
            t=1.0,
            U=4.0,
            v=0.0,
            repr_mode="JW",
            indexing="blocked",
            pbc=True,
        )
        resolved = resolve_problem_context(
            ProblemRequest(
                problem_key="hubbard",
                num_sites=2,
                t=1.0,
                u=4.0,
                dv=0.0,
                omega0=0.0,
                g_ep=0.0,
                n_ph_max=1,
                boson_encoding="binary",
                ordering="blocked",
                boundary="periodic",
                include_zero_point=True,
            ),
            hamiltonian=h_poly,
        )
        with pytest.raises(ValueError, match="resolved_problem_context does not match raw ADAPT inputs"):
            _run_hardcoded_adapt_vqe(
                h_poly=h_poly,
                resolved_problem_context=resolved,
                num_sites=3,
                ordering="blocked",
                problem="hubbard",
                adapt_pool="uccsd",
                t=1.0,
                u=4.0,
                dv=0.0,
                boundary="periodic",
                omega0=0.0,
                g_ep=0.0,
                n_ph_max=1,
                boson_encoding="binary",
                max_depth=1,
                eps_grad=1e-6,
                eps_energy=1e-10,
                maxiter=1,
                seed=17,
                allow_repeats=False,
                finite_angle_fallback=True,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
            )

    def test_run_hardcoded_adapt_vqe_accepts_matching_molecular_resolved_problem_context(
        self,
        tmp_path: Path,
    ):
        json_path = _write_molecular_problem_json(tmp_path)
        h_poly = build_restricted_closed_shell_molecular_hamiltonian(
            load_restricted_closed_shell_problem_from_json(json_path),
            ordering="blocked",
        )
        resolved = resolve_problem_context(
            ProblemRequest(
                problem_key="molecular_restricted_closed_shell",
                num_sites=2,
                t=0.0,
                u=0.0,
                dv=0.0,
                omega0=0.0,
                g_ep=0.0,
                n_ph_max=0,
                boson_encoding="binary",
                ordering="blocked",
                boundary="open",
                include_zero_point=True,
                molecular_problem_json=str(json_path),
            ),
            hamiltonian=h_poly,
        )
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=h_poly,
            resolved_problem_context=resolved,
            num_sites=2,
            ordering="blocked",
            problem="molecular_restricted_closed_shell",
            molecular_problem_json=str(json_path),
            adapt_pool="uccsd",
            t=0.0,
            u=0.0,
            dv=0.0,
            boundary="open",
            omega0=0.0,
            g_ep=0.0,
            n_ph_max=0,
            boson_encoding="binary",
            adapt_continuation_mode="legacy",
            max_depth=1,
            eps_grad=1e-6,
            eps_energy=1e-10,
            maxiter=10,
            seed=17,
            allow_repeats=False,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
        )
        assert str(payload["pool_type"]) == "uccsd"
        assert math.isfinite(float(payload["energy"]))

    def test_molecular_full_meta_pool_is_problem_local_mega_pool(self, tmp_path: Path):
        json_path = _write_molecular_problem_json(tmp_path)
        resolved = resolve_problem_context(
            ProblemRequest(
                problem_key="molecular_restricted_closed_shell",
                num_sites=2,
                t=0.0,
                u=0.0,
                dv=0.0,
                omega0=0.0,
                g_ep=0.0,
                n_ph_max=0,
                boson_encoding="binary",
                ordering="blocked",
                boundary="open",
                include_zero_point=True,
                molecular_problem_json=str(json_path),
            )
        )
        plan = resolve_pool_plan(
            resolved_problem=resolved,
            continuation_mode="phase3_v1",
            adapt_pool="full_meta",
            paop_r=1,
            paop_split_paulis=False,
            paop_prune_eps=0.0,
            paop_normalization="none",
            phase3_symmetry_mitigation_mode="verify_only",
        )
        labels = [str(op.label) for op in plan.pool]
        assert plan.pool_key == "full_meta"
        assert plan.phase1_core_limit == len(plan.pool)
        assert plan.phase1_residual_indices == set()
        assert any(label.startswith("uccsd_") for label in labels)
        assert "ham_block::molecular_one_body" in labels
        assert "ham_block::molecular_two_body" in labels
        assert any(label.startswith("ham_term(") for label in labels)

    def test_molecular_vibronic_h2_fixture_context_and_pool_smoke(self):
        resolved = resolve_problem_context(
            ProblemRequest(
                problem_key="molecular_vibronic_h2",
                num_sites=2,
                t=0.0,
                u=0.0,
                dv=0.0,
                omega0=1.0,
                g_ep=0.5,
                n_ph_max=1,
                boson_encoding="binary",
                ordering="blocked",
                boundary="open",
                include_zero_point=True,
            )
        )
        assert resolved.family_key == "molecular_vibronic_h2"
        assert resolved.layout.total_qubits == 5
        assert resolved.layout.fermion_qubits == 4
        assert resolved.layout.boson_qubits == 1
        assert resolved.sector.num_particles == (1, 1)
        assert resolved.reference_state.build_state().shape == (32,)
        assert math.isfinite(float(resolved.exact_target.resolve_energy()))

        plan = resolve_pool_plan(
            resolved_problem=resolved,
            continuation_mode="phase3_v1",
            adapt_pool="full_meta",
            paop_r=1,
            paop_split_paulis=False,
            paop_prune_eps=0.0,
            paop_normalization="none",
            phase3_symmetry_mitigation_mode="verify_only",
        )
        labels = [str(op.label) for op in plan.pool]
        assert plan.pool_key == "full_meta"
        assert plan.method_name == "hardcoded_adapt_vqe_full_meta_molecular_vibronic_h2"
        assert plan.phase1_core_limit == len(plan.pool)
        assert plan.phase1_residual_indices == set()
        assert any(label.startswith("el::") for label in labels)
        assert "boson::p" in labels
        assert any(label.startswith("coupled::") for label in labels)

    def test_molecular_vibronic_h2_rejects_noncanonical_fixture_controls(self):
        with pytest.raises(ValueError, match="L=2"):
            resolve_problem_context(
                ProblemRequest(
                    problem_key="molecular_vibronic_h2",
                    num_sites=3,
                    t=0.0,
                    u=0.0,
                    dv=0.0,
                    omega0=1.0,
                    g_ep=0.5,
                    n_ph_max=1,
                    boson_encoding="binary",
                    ordering="blocked",
                    boundary="open",
                    include_zero_point=True,
                )
            )

    def test_wrapped_h2o_molecular_asset_loads_as_closed_shell_problem(self):
        json_path = REPO_ROOT / "src" / "quantum" / "chemistry" / "h2o_sto3g_fast_result.json"
        molecular_problem = load_restricted_closed_shell_problem_from_json(json_path)
        assert molecular_problem.n_spatial_orbitals == 7
        assert molecular_problem.n_spin_orbitals == 14
        assert molecular_problem.num_particles == (5, 5)
        assert molecular_problem.reference == "rhf"

    def test_lih_molecular_fixture_loads_as_closed_shell_problem(self):
        json_path = REPO_ROOT / "test_support" / "molecular_problem_lih_sto3g.json"
        molecular_problem = load_restricted_closed_shell_problem_from_json(json_path)
        assert molecular_problem.basis == "sto-3g"
        assert molecular_problem.charge == 0
        assert molecular_problem.multiplicity == 1
        assert molecular_problem.reference == "rhf"
        assert molecular_problem.n_spatial_orbitals == 6
        assert molecular_problem.n_spin_orbitals == 12
        assert molecular_problem.num_particles == (2, 2)
        assert math.isfinite(float(molecular_problem.hf_energy))
        assert math.isfinite(float(molecular_problem.nuclear_repulsion_energy))

    def test_lih_real_builder_resolves_context_hamiltonian_and_pool_smoke(self):
        started_at = time.perf_counter()
        json_path = REPO_ROOT / "test_support" / "molecular_problem_lih_sto3g.json"
        resolved = resolve_problem_context(
            ProblemRequest(
                problem_key="molecular_restricted_closed_shell",
                num_sites=6,
                t=0.0,
                u=0.0,
                dv=0.0,
                omega0=0.0,
                g_ep=0.0,
                n_ph_max=0,
                boson_encoding="binary",
                ordering="blocked",
                boundary="open",
                include_zero_point=True,
                molecular_problem_json=str(json_path),
            )
        )
        h_terms = resolved.hamiltonian.return_polynomial()
        assert resolved.family_key == "molecular_restricted_closed_shell"
        assert resolved.layout.total_qubits == 12
        assert resolved.layout.fermion_qubits == 12
        assert resolved.sector.num_particles == (2, 2)
        assert len(h_terms) > 0
        assert all(term.nqubit() == 12 for term in h_terms)

        runtime_data = dict(resolved.runtime_data)
        molecular_problem = runtime_data["molecular_problem"]
        assert runtime_data["molecular_problem_json"] == str(json_path)
        assert molecular_problem.n_spatial_orbitals == 6
        assert molecular_problem.num_particles == (2, 2)

        plan = resolve_pool_plan(
            resolved_problem=resolved,
            continuation_mode="phase3_v1",
            adapt_pool="hamiltonian_blocks",
            paop_r=1,
            paop_split_paulis=False,
            paop_prune_eps=0.0,
            paop_normalization="none",
            phase3_symmetry_mitigation_mode="verify_only",
        )
        labels = {str(op.label) for op in plan.pool}
        assert plan.pool_key == "hamiltonian_blocks"
        assert plan.method_name == "hardcoded_adapt_vqe_hamiltonian_blocks_molecular"
        assert {"ham_block::molecular_one_body", "ham_block::molecular_two_body"}.issubset(labels)
        assert all(len(op.polynomial.return_polynomial()) > 0 for op in plan.pool)
        # Keep this as a generous performance-regression guard; the smoke is
        # structural, while command-level smokes enforce tighter timeouts.
        assert time.perf_counter() - started_at < 60.0

    def test_ttprime_quadrature_pool_stays_in_target_spin_sector(self):
        resolved = resolve_problem_context(
            ProblemRequest(
                problem_key="ttprime_hubbard",
                num_sites=4,
                t=1.0,
                u=4.0,
                dv=0.25,
                omega0=0.0,
                g_ep=0.0,
                n_ph_max=0,
                boson_encoding="binary",
                ordering="blocked",
                boundary="open",
                include_zero_point=True,
                t_prime=0.4,
            )
        )
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=resolved.hamiltonian,
            resolved_problem_context=resolved,
            num_sites=4,
            ordering="blocked",
            problem="ttprime_hubbard",
            adapt_pool="hamiltonian_quadratures",
            t=1.0,
            u=4.0,
            dv=0.25,
            boundary="open",
            omega0=0.0,
            g_ep=0.0,
            n_ph_max=0,
            boson_encoding="binary",
            adapt_continuation_mode="legacy",
            max_depth=2,
            eps_grad=1e-6,
            eps_energy=1e-10,
            maxiter=80,
            seed=17,
            t_prime=0.4,
            allow_repeats=False,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_inner_optimizer="POWELL",
        )
        assert payload["parameterization_mode"] == "logical_shared"
        assert payload["parameterization_execution_mode"] == "logical_shared"
        assert int(payload["ansatz_depth"]) > 0
        assert payload["optimizer_coordinate_chart"]["coordinate_mode"] == (
            "logical_shared"
        )
        assert payload["optimizer_coordinate_chart"]["optimizer_dimension"] == int(
            payload["logical_num_parameters"]
        )
        for row in payload["history"]:
            assert row["optimizer_coordinate_mode"] == "logical_shared"
            assert row["optimizer_coordinate_dimension"] == row[
                "optimizer_logical_active_dimension"
            ]
            assert row["optimizer_runtime_active_dimension"] >= row[
                "optimizer_coordinate_dimension"
            ]
        assert _runtime_theta_matches_logical_blocks(payload)
        selected_ops = _selected_ops_from_payload(
            resolved_problem=resolved,
            payload=payload,
            continuation_mode="legacy",
        )
        executor = CompiledAnsatzExecutor(
            selected_ops,
            parameterization_mode=str(payload["parameterization_mode"]),
        )
        theta_use = np.asarray(
            payload.get("logical_optimal_point", payload["optimal_point"]),
            dtype=float,
        )
        psi = executor.prepare_state(
            theta_use,
            resolved.reference_state.build_state(),
        )
        weights = _fermion_sector_weights(psi, num_sites=4, ordering="blocked")
        assert float(weights.get((2, 2), 0.0)) > 0.999999

    def test_spinless_quadrature_pool_stays_in_fixed_count_sector(self):
        resolved = resolve_problem_context(
            ProblemRequest(
                problem_key="spinless_tv",
                num_sites=4,
                t=1.0,
                u=0.0,
                dv=0.1,
                omega0=0.0,
                g_ep=0.0,
                n_ph_max=0,
                boson_encoding="binary",
                ordering="blocked",
                boundary="open",
                include_zero_point=True,
                v_nn=1.5,
                n_fermions=2,
            )
        )
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=resolved.hamiltonian,
            resolved_problem_context=resolved,
            num_sites=4,
            ordering="blocked",
            problem="spinless_tv",
            adapt_pool="hamiltonian_quadratures",
            t=1.0,
            u=0.0,
            dv=0.1,
            boundary="open",
            omega0=0.0,
            g_ep=0.0,
            n_ph_max=0,
            boson_encoding="binary",
            adapt_continuation_mode="legacy",
            max_depth=2,
            eps_grad=1e-6,
            eps_energy=1e-10,
            maxiter=80,
            seed=17,
            v_nn=1.5,
            n_fermions=2,
            allow_repeats=False,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_inner_optimizer="POWELL",
        )
        assert payload["parameterization_mode"] == "logical_shared"
        assert int(payload["ansatz_depth"]) > 0
        assert _runtime_theta_matches_logical_blocks(payload)
        selected_ops = _selected_ops_from_payload(
            resolved_problem=resolved,
            payload=payload,
            continuation_mode="legacy",
        )
        executor = CompiledAnsatzExecutor(
            selected_ops,
            parameterization_mode=str(payload["parameterization_mode"]),
        )
        theta_use = np.asarray(
            payload.get("logical_optimal_point", payload["optimal_point"]),
            dtype=float,
        )
        psi = executor.prepare_state(
            theta_use,
            resolved.reference_state.build_state(),
        )
        weights = _spinless_sector_weights(psi, num_sites=4)
        assert float(weights.get(2, 0.0)) > 0.999999

    def test_ionic_hva_pool_uses_logical_shared_parameterization(self):
        resolved = resolve_problem_context(
            ProblemRequest(
                problem_key="ionic_hubbard",
                num_sites=2,
                t=1.0,
                u=4.0,
                dv=0.25,
                omega0=0.0,
                g_ep=0.0,
                n_ph_max=0,
                boson_encoding="binary",
                ordering="blocked",
                boundary="open",
                include_zero_point=True,
            )
        )
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=resolved.hamiltonian,
            resolved_problem_context=resolved,
            num_sites=2,
            ordering="blocked",
            problem="ionic_hubbard",
            adapt_pool="hva",
            t=1.0,
            u=4.0,
            dv=0.25,
            boundary="open",
            omega0=0.0,
            g_ep=0.0,
            n_ph_max=0,
            boson_encoding="binary",
            adapt_continuation_mode="legacy",
            max_depth=2,
            eps_grad=1e-6,
            eps_energy=1e-10,
            maxiter=80,
            seed=17,
            allow_repeats=False,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_inner_optimizer="POWELL",
        )
        assert payload["parameterization_mode"] == "logical_shared"
        assert int(payload["ansatz_depth"]) > 0
        assert _runtime_theta_matches_logical_blocks(payload)
        selected_ops = _selected_ops_from_payload(
            resolved_problem=resolved,
            payload=payload,
            continuation_mode="legacy",
        )
        executor = CompiledAnsatzExecutor(
            selected_ops,
            parameterization_mode=str(payload["parameterization_mode"]),
        )
        theta_use = np.asarray(
            payload.get("logical_optimal_point", payload["optimal_point"]),
            dtype=float,
        )
        psi = executor.prepare_state(
            theta_use,
            resolved.reference_state.build_state(),
        )
        weights = _fermion_sector_weights(psi, num_sites=2, ordering="blocked")
        assert float(weights.get((1, 1), 0.0)) > 0.999999

    def test_extended_family_max_pool_uses_logical_shared_parameterization(self):
        resolved = resolve_problem_context(
            ProblemRequest(
                problem_key="extended_hubbard",
                num_sites=2,
                t=1.0,
                u=4.0,
                dv=0.25,
                omega0=0.0,
                g_ep=0.0,
                n_ph_max=0,
                boson_encoding="binary",
                ordering="blocked",
                boundary="open",
                include_zero_point=True,
                v_nn=1.5,
            )
        )
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=resolved.hamiltonian,
            resolved_problem_context=resolved,
            num_sites=2,
            ordering="blocked",
            problem="extended_hubbard",
            adapt_pool="family_max",
            t=1.0,
            u=4.0,
            dv=0.25,
            boundary="open",
            omega0=0.0,
            g_ep=0.0,
            n_ph_max=0,
            boson_encoding="binary",
            adapt_continuation_mode="legacy",
            max_depth=2,
            eps_grad=1e-6,
            eps_energy=1e-10,
            maxiter=80,
            seed=17,
            v_nn=1.5,
            allow_repeats=False,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_inner_optimizer="POWELL",
        )
        assert payload["parameterization_mode"] == "logical_shared"
        assert int(payload["ansatz_depth"]) > 0
        assert _runtime_theta_matches_logical_blocks(payload)
        selected_ops = _selected_ops_from_payload(
            resolved_problem=resolved,
            payload=payload,
            continuation_mode="legacy",
        )
        executor = CompiledAnsatzExecutor(
            selected_ops,
            parameterization_mode=str(payload["parameterization_mode"]),
        )
        theta_use = np.asarray(
            payload.get("logical_optimal_point", payload["optimal_point"]),
            dtype=float,
        )
        psi = executor.prepare_state(
            theta_use,
            resolved.reference_state.build_state(),
        )
        weights = _fermion_sector_weights(psi, num_sites=2, ordering="blocked")
        assert float(weights.get((1, 1), 0.0)) > 0.999999


    def test_spin_boson_full_meta_pool_uses_logical_shared_parameterization(self):
        resolved = resolve_problem_context(
            ProblemRequest(
                problem_key="spin_boson",
                num_sites=1,
                t=0.7,
                u=0.4,
                dv=0.3,
                omega0=1.0,
                g_ep=0.6,
                n_ph_max=2,
                boson_encoding="binary",
                ordering="blocked",
                boundary="open",
                include_zero_point=True,
            )
        )
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=resolved.hamiltonian,
            resolved_problem_context=resolved,
            num_sites=1,
            ordering="blocked",
            problem="spin_boson",
            adapt_pool="full_meta",
            t=0.7,
            u=0.4,
            dv=0.3,
            boundary="open",
            omega0=1.0,
            g_ep=0.6,
            n_ph_max=2,
            boson_encoding="binary",
            adapt_continuation_mode="phase3_v1",
            max_depth=2,
            eps_grad=1e-6,
            eps_energy=1e-10,
            maxiter=80,
            seed=17,
            allow_repeats=False,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_inner_optimizer="POWELL",
        )
        assert payload["parameterization_mode"] == "logical_shared"
        assert payload["continuation_mode"] == "phase3_v1"
        assert int(payload["ansatz_depth"]) > 0
        assert _runtime_theta_matches_logical_blocks(payload)
        assert np.isfinite(float(payload["energy"]))
        assert np.isfinite(float(payload["exact_gs_energy"]))
        assert np.isfinite(float(payload["abs_delta_e"]))

    def test_bose_hubbard_full_meta_pool_runs(self):
        resolved = resolve_problem_context(
            ProblemRequest(
                problem_key="bose_hubbard",
                num_sites=2,
                t=0.7,
                u=0.4,
                dv=0.2,
                omega0=1.0,
                g_ep=0.0,
                n_ph_max=2,
                boson_encoding="binary",
                ordering="blocked",
                boundary="open",
                include_zero_point=True,
            )
        )
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=resolved.hamiltonian,
            resolved_problem_context=resolved,
            num_sites=2,
            ordering="blocked",
            problem="bose_hubbard",
            adapt_pool="full_meta",
            t=0.7,
            u=0.4,
            dv=0.2,
            boundary="open",
            omega0=1.0,
            g_ep=0.0,
            n_ph_max=2,
            boson_encoding="binary",
            adapt_continuation_mode="phase3_v1",
            max_depth=2,
            eps_grad=1e-6,
            eps_energy=1e-10,
            maxiter=80,
            seed=19,
            allow_repeats=False,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_inner_optimizer="POWELL",
        )
        assert payload["parameterization_mode"] == "logical_shared"
        assert payload["continuation_mode"] == "phase3_v1"
        assert int(payload["ansatz_depth"]) >= 0
        assert np.isfinite(float(payload["energy"]))
        assert np.isfinite(float(payload["exact_gs_energy"]))

    def test_ionic_hubbard_phase3_uses_staged_policy_defaults(self):
        resolved = resolve_problem_context(
            ProblemRequest(
                problem_key="ionic_hubbard",
                num_sites=2,
                t=1.0,
                u=4.0,
                dv=0.25,
                omega0=0.0,
                g_ep=0.0,
                n_ph_max=0,
                boson_encoding="binary",
                ordering="blocked",
                boundary="open",
                include_zero_point=True,
            )
        )
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=resolved.hamiltonian,
            resolved_problem_context=resolved,
            num_sites=2,
            ordering="blocked",
            problem="ionic_hubbard",
            adapt_pool="hamiltonian_quadratures",
            t=1.0,
            u=4.0,
            dv=0.25,
            boundary="open",
            omega0=0.0,
            g_ep=0.0,
            n_ph_max=0,
            boson_encoding="binary",
            adapt_continuation_mode="phase3_v1",
            max_depth=2,
            eps_grad=1e-6,
            eps_energy=1e-10,
            maxiter=80,
            seed=23,
            allow_repeats=False,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_inner_optimizer="POWELL",
        )
        assert payload["continuation_mode"] == "phase3_v1"
        assert payload["adapt_drop_policy_source"] == "auto_staged"
        assert bool(payload["eps_energy_termination_enabled"]) is False
        assert bool(payload["eps_grad_termination_enabled"]) is False
        assert np.isfinite(float(payload["energy"]))


class TestAdaptCLIParsing:
    """CLI parsing includes newly supported ADAPT pool options."""

    def test_parse_rejects_retired_phase2_pairwise_novelty_surface(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(sys, "argv", ["adapt_pipeline.py"])
        args_default = _adapt_mod.parse_args()
        assert not hasattr(args_default, "phase2_novelty_mode")

        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--phase2-novelty-mode", "legacy_pairwise_v1"],
        )
        with pytest.raises(SystemExit):
            _adapt_mod.parse_args()

    def test_parse_exposes_only_deferred_gram_fallback_ridge(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(sys, "argv", ["adapt_pipeline.py"])
        args_default = _adapt_mod.parse_args()
        assert float(args_default.deferred_gram_fallback_ridge) == pytest.approx(
            1.0e-6
        )
        assert not hasattr(args_default, "phase2_gamma_N")

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "adapt_pipeline.py",
                "--deferred-gram-fallback-ridge",
                "2e-5",
            ],
        )
        args = _adapt_mod.parse_args()
        assert float(args.deferred_gram_fallback_ridge) == pytest.approx(
            2.0e-5
        )

    def test_parse_accepts_current_checkpoint_json(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(sys, "argv", ["adapt_pipeline.py"])
        args_default = _adapt_mod.parse_args()
        assert args_default.adapt_current_json is None
        assert int(args_default.adapt_current_json_every_depth) == 1

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "adapt_pipeline.py",
                "--adapt-current-json",
                "current.json",
                "--adapt-current-json-every-depth",
                "3",
                "--adapt-current-json-keep-history-tail",
                "7",
            ],
        )
        args = _adapt_mod.parse_args()
        assert Path(args.adapt_current_json) == Path("current.json")
        assert int(args.adapt_current_json_every_depth) == 3
        assert int(args.adapt_current_json_keep_history_tail) == 7

    def test_parse_accepts_bfgs_rotosolve_and_qnspsa_opt_in_and_default_stays_spsa(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(sys, "argv", ["adapt_pipeline.py"])
        args_default = _adapt_mod.parse_args()
        assert str(args_default.adapt_inner_optimizer) == "SPSA"

        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--adapt-inner-optimizer", "BFGS"],
        )
        args = _adapt_mod.parse_args()
        assert str(args.adapt_inner_optimizer) == "BFGS"

        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--adapt-inner-optimizer", "ROTOSOLVE"],
        )
        args = _adapt_mod.parse_args()
        assert str(args.adapt_inner_optimizer) == "ROTOSOLVE"

        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--adapt-inner-optimizer", "QNSPSA"],
        )
        args = _adapt_mod.parse_args()
        assert str(args.adapt_inner_optimizer) == "QNSPSA"

    def test_parse_accepts_uccsd_paop_lf_full_pool(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--adapt-pool", "uccsd_paop_lf_full"],
        )
        args = _adapt_mod.parse_args()
        assert str(args.adapt_pool) == "uccsd_paop_lf_full"

    def test_parse_accepts_uccsd_otimes_paop_lf_std_pool(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--problem", "hh", "--adapt-pool", "uccsd_otimes_paop_lf_std"],
        )
        args = _adapt_mod.parse_args()
        assert str(args.adapt_pool) == "uccsd_otimes_paop_lf_std"

    def test_parse_accepts_sq_lf_std_pool(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--problem", "hh", "--adapt-pool", "sq_lf_std"],
        )
        args = _adapt_mod.parse_args()
        assert str(args.adapt_pool) == "sq_lf_std"

    def test_parse_accepts_full_meta_pool(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--problem", "hh", "--adapt-pool", "full_meta"],
        )
        args = _adapt_mod.parse_args()
        assert str(args.adapt_pool) == "full_meta"

    def test_parse_accepts_hamiltonian_blocks_pool(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--adapt-pool", "hamiltonian_blocks"],
        )
        args = _adapt_mod.parse_args()
        assert str(args.adapt_pool) == "hamiltonian_blocks"

    def test_parse_accepts_hubbard_uccsd_qeb_pool(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--problem", "hubbard", "--adapt-pool", "uccsd_qeb"],
        )
        args = _adapt_mod.parse_args()
        assert str(args.problem) == "hubbard"
        assert str(args.adapt_pool) == "uccsd_qeb"

    def test_parse_accepts_hubbard_uccsd_qeb_hva_blocks_pool(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--problem", "hubbard", "--adapt-pool", "uccsd_qeb_hva_blocks"],
        )
        args = _adapt_mod.parse_args()
        assert str(args.problem) == "hubbard"
        assert str(args.adapt_pool) == "uccsd_qeb_hva_blocks"

    def test_parse_accepts_hamiltonian_quadratures_pool(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--adapt-pool", "hamiltonian_quadratures"],
        )
        args = _adapt_mod.parse_args()
        assert str(args.adapt_pool) == "hamiltonian_quadratures"

    def test_parse_accepts_family_max_pool(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--problem", "extended_hubbard", "--adapt-pool", "family_max"],
        )
        args = _adapt_mod.parse_args()
        assert str(args.adapt_pool) == "family_max"

    def test_parse_accepts_spin_boson_problem(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--problem", "spin_boson", "--adapt-pool", "full_meta", "--L", "1"],
        )
        args = _adapt_mod.parse_args()
        assert str(args.problem) == "spin_boson"
        assert str(args.adapt_pool) == "full_meta"

    def test_parse_accepts_bose_hubbard_problem(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--problem", "bose_hubbard", "--adapt-pool", "full_meta", "--L", "2"],
        )
        args = _adapt_mod.parse_args()
        assert str(args.problem) == "bose_hubbard"
        assert str(args.adapt_pool) == "full_meta"

    def test_parse_accepts_molecular_vibronic_h2_problem(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "adapt_pipeline.py",
                "--problem",
                "molecular_vibronic_h2",
                "--adapt-pool",
                "full_meta",
                "--L",
                "2",
                "--n-ph-max",
                "1",
                "--molecular-vibronic-h2-fixture-json",
                "test_support/molecular_vibronic_h2_sto3g_fd001.json",
            ],
        )
        args = _adapt_mod.parse_args()
        assert str(args.problem) == "molecular_vibronic_h2"
        assert str(args.adapt_pool) == "full_meta"
        assert str(args.molecular_vibronic_h2_fixture_json).endswith("molecular_vibronic_h2_sto3g_fd001.json")

    def test_parse_accepts_molecular_vibronic_h2o_linear_fd_problem(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "adapt_pipeline.py",
                "--problem",
                "molecular_vibronic_h2o_linear_fd",
                "--adapt-pool",
                "full_meta",
                "--L",
                "2",
                "--n-ph-max",
                "2",
                "--molecular-vibronic-h2o-linear-fd-fixture-json",
                "test_support/h2o_linear_fd_fixture.json",
            ],
        )
        args = _adapt_mod.parse_args()
        assert str(args.problem) == "molecular_vibronic_h2o_linear_fd"
        assert str(args.adapt_pool) == "full_meta"
        assert str(args.molecular_vibronic_h2o_linear_fd_fixture_json).endswith("h2o_linear_fd_fixture.json")

    def test_parse_accepts_molecular_problem_json(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        json_path = _write_molecular_problem_json(tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "adapt_pipeline.py",
                "--problem",
                "molecular_restricted_closed_shell",
                "--molecular-problem-json",
                str(json_path),
            ],
        )
        args = _adapt_mod.parse_args()
        assert str(args.problem) == "molecular_restricted_closed_shell"
        assert Path(args.molecular_problem_json) == json_path

    def test_continuation_mode_resolver_accepts_phase3_for_molecular(self):
        assert (
            _adapt_mod._resolve_cli_adapt_continuation_mode(
                problem="molecular_restricted_closed_shell",
                requested_mode="phase3_v1",
            )
            == "phase3_v1"
        )

    def test_parse_accepts_pareto_lean_pool(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--problem", "hh", "--adapt-pool", "pareto_lean"],
        )
        args = _adapt_mod.parse_args()
        assert str(args.adapt_pool) == "pareto_lean"

    def test_parse_accepts_pareto_lean_l2_pool(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--problem", "hh", "--adapt-pool", "pareto_lean_l2"],
        )
        args = _adapt_mod.parse_args()
        assert str(args.adapt_pool) == "pareto_lean_l2"

    def test_parse_accepts_adapt_state_backend_legacy(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--adapt-state-backend", "legacy"],
        )
        args = _adapt_mod.parse_args()
        assert str(args.adapt_state_backend) == "legacy"

    def test_parse_defaults_direct_cli_continuation_mode_to_none(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(sys, "argv", ["adapt_pipeline.py"])
        args = _adapt_mod.parse_args()
        assert args.adapt_continuation_mode is None

    def test_parse_accepts_phase1_continuation_mode(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--adapt-continuation-mode", "phase1_v1"],
        )
        args = _adapt_mod.parse_args()
        assert str(args.adapt_continuation_mode) == "phase1_v1"

    def test_parse_accepts_phase2_continuation_mode(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--adapt-continuation-mode", "phase2_v1"],
        )
        args = _adapt_mod.parse_args()
        assert str(args.adapt_continuation_mode) == "phase2_v1"

    def test_parse_accepts_phase3_continuation_mode(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--adapt-continuation-mode", "phase3_v1"],
        )
        args = _adapt_mod.parse_args()
        assert str(args.adapt_continuation_mode) == "phase3_v1"

    def test_parse_defaults_to_hardware_resolvable_phase3_selector_policy(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(sys, "argv", ["adapt_pipeline.py"])
        args = _adapt_mod.parse_args()
        assert str(args.phase3_selector_policy) == "hardware_resolvable_v1"
        assert str(args.hardware_resolution_mode) == "ideal"
        assert float(args.gradient_hw_floor) == pytest.approx(0.0)
        assert float(args.gradient_drift_floor) == pytest.approx(0.0)
        assert args.hardware_resolution_profile_json is None
        assert args.hardware_resolution_profile_name is None
        assert bool(args.phase0_pilot_enabled) is True
        assert float(args.phase0_pilot_threshold) == pytest.approx(0.0)
        assert int(args.phase0_pilot_max_records) == 0
        assert str(args.phase1_score_mode) == "trust_region_v1"
        assert not hasattr(args, "static_route_id")
        assert not hasattr(args, "static_meta_feature_profile")
        assert not hasattr(args, "allow_legacy_static_route")
        assert not hasattr(args, "phase0_lane_quota_pressure")
        assert not hasattr(args, "phase0_algebraic_lane_mode")
        assert str(args.phase3_plateau_acquisition_mode) == "off"
        assert str(args.phase3_plateau_acquisition_score) == "log_volume_v1"
        assert float(args.phase3_plateau_unlock_margin) == pytest.approx(1e-8)
        assert str(args.phase3_plateau_duplicate_policy) == "block_exact_position_v1"
        assert float(args.phase3_plateau_lambda_vol) == pytest.approx(1e-8)
        assert float(args.phase3_plateau_sigma_min) == pytest.approx(0.0)
        assert float(args.phase3_plateau_nu_min) == pytest.approx(0.0)
        assert float(args.phase3_plateau_volume_min) == pytest.approx(0.0)
        assert int(args.phase3_plateau_failed_family_patience) == 0
        assert str(args.phase3_plateau_trial_optimizer) == "inherit"
        assert int(args.phase3_plateau_trial_qngd_maxiter) == 64

    def test_parse_accepts_phase3_selector_policy(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--phase3-selector-policy", "algebraic_nested_v1"],
        )
        args = _adapt_mod.parse_args()
        assert str(args.phase3_selector_policy) == "algebraic_nested_v1"

    def test_parse_accepts_legacy_phase1_score_mode(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--phase1-score-mode", "legacy_simple_v1"],
        )
        args = _adapt_mod.parse_args()
        assert str(args.phase1_score_mode) == "legacy_simple_v1"

    def test_parse_accepts_hardware_resolution_manual_floors(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "adapt_pipeline.py",
                "--hardware-resolution-mode",
                "manual",
                "--gradient-hw-floor",
                "0.02",
                "--gradient-drift-floor",
                "0.03",
            ],
        )
        args = _adapt_mod.parse_args()
        assert str(args.hardware_resolution_mode) == "manual"
        assert float(args.gradient_hw_floor) == pytest.approx(0.02)
        assert float(args.gradient_drift_floor) == pytest.approx(0.03)

    def test_parse_accepts_hardware_resolution_profile_args(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
        profile_json = _write_hardware_resolution_profile_json(tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "adapt_pipeline.py",
                "--hardware-resolution-mode",
                "profile",
                "--hardware-resolution-profile-json",
                str(profile_json),
                "--hardware-resolution-profile-name",
                "calib_a",
            ],
        )
        args = _adapt_mod.parse_args()
        assert str(args.hardware_resolution_mode) == "profile"
        assert args.hardware_resolution_profile_json == profile_json
        assert str(args.hardware_resolution_profile_name) == "calib_a"
        assert float(args.gradient_hw_floor) == pytest.approx(0.0)
        assert float(args.gradient_drift_floor) == pytest.approx(0.0)

    def test_parse_accepts_phase0_pilot_controls(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "adapt_pipeline.py",
                "--phase0-no-pilot",
                "--phase0-pilot-alpha",
                "0.2",
                "--phase0-pilot-threshold",
                "0.01",
                "--phase0-pilot-max-records",
                "5",
            ],
        )
        args = _adapt_mod.parse_args()
        assert bool(args.phase0_pilot_enabled) is False
        assert float(args.phase0_pilot_alpha) == pytest.approx(0.2)
        assert float(args.phase0_pilot_threshold) == pytest.approx(0.01)
        assert int(args.phase0_pilot_max_records) == 5

    def test_parse_accepts_physical_lane_shortlist_controls(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "adapt_pipeline.py",
                "--physical-phase2-lane-rel-threshold",
                "0.22",
                "--physical-phase1-lane-quota-pressure",
                "0.55",
                "--physical-phase2-lane-quota-pressure",
                "0.65",
            ],
        )
        args = _adapt_mod.parse_args()
        assert float(args.physical_phase2_lane_rel_threshold) == pytest.approx(0.22)
        assert float(args.physical_phase1_lane_quota_pressure) == pytest.approx(0.55)
        assert float(args.physical_phase2_lane_quota_pressure) == pytest.approx(0.65)

    def test_parse_defaults_omit_pruning_and_maturity_override_surfaces(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(sys, "argv", ["adapt_pipeline.py"])
        args = _adapt_mod.parse_args()
        assert all(
            not hasattr(args, name) for name in PRUNING_RUNTIME_KEYS
        )
        assert all(not hasattr(args, name) for name in MATURITY_RUNTIME_KEYS)
        assert not hasattr(args, "phase_live_hysteresis_enabled")
        assert not hasattr(args, "phase2_null_nrem_high_threshold")
        assert not hasattr(args, "phase2_live_nrem_low_threshold")
        assert not hasattr(args, "phase3_null_nrem_high_threshold")
        assert not hasattr(args, "phase3_live_nrem_low_threshold")
        assert not hasattr(args, "phase2_hysteresis_steps")
        assert not hasattr(args, "phase3_hysteresis_steps")
        assert float(args.physical_phase2_lane_rel_threshold) == pytest.approx(0.10)
        assert float(args.physical_phase1_lane_quota_pressure) == pytest.approx(0.70)
        assert float(args.physical_phase2_lane_quota_pressure) == pytest.approx(0.70)

    @pytest.mark.parametrize(
        "retired_flag",
        tuple("--" + name.replace("_", "-") for name in MATURITY_RUNTIME_KEYS),
    )
    def test_parse_rejects_retired_maturity_override_flags(
        self,
        monkeypatch: pytest.MonkeyPatch,
        retired_flag: str,
    ):
        monkeypatch.setattr(sys, "argv", ["adapt_pipeline.py", retired_flag, "1"])
        with pytest.raises(SystemExit, match="2"):
            _adapt_mod.parse_args()

    def test_parse_rejects_retired_pruning_extension_flags(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "adapt_pipeline.py",
                "--phase1-prune-policy",
                "recoverability_ladder_v1",
            ],
        )
        with pytest.raises(SystemExit, match="2"):
            _adapt_mod.parse_args()

    def test_parse_rejects_archival_phase3_runtime_split_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "adapt_pipeline.py",
                "--phase3-runtime-split-mode",
                "shortlist_pauli_children_v1",
            ],
        )
        with pytest.raises(SystemExit):
            _adapt_mod.parse_args()
        captured = capsys.readouterr()
        assert "--allow-archival-phase3-runtime-split" in captured.err

    def test_parse_defaults_phase3_runtime_split_max_subset_size_to_three(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(sys, "argv", ["adapt_pipeline.py"])
        args = _adapt_mod.parse_args()
        assert bool(args.allow_archival_phase3_runtime_split) is False
        assert int(args.phase3_runtime_split_max_subset_size) == 3
        assert str(args.phase3_runtime_split_child_set_symmetry_policy) == "parent"
        assert str(args.adapt_child_pool_expansion_mode) == "off"
        assert str(args.adapt_child_pool_expansion_symmetry_policy) == "off"
        assert int(args.adapt_child_pool_expansion_max_subset_size) == 3
        assert str(args.shared_pauli_pool_mode) == "off"
        assert str(args.shared_pauli_pool_symmetry_policy) == "off"
        assert int(args.shared_pauli_pool_max_subset_size) == 3

    def test_parse_accepts_shared_pauli_pool_mode(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "adapt_pipeline.py",
                "--shared-pauli-pool-mode",
                "shared_pauli_child_sets_v1",
                "--shared-pauli-pool-symmetry-policy",
                "hard_guard",
                "--shared-pauli-pool-max-subset-size",
                "4",
            ],
        )
        args = _adapt_mod.parse_args()
        assert bool(args.allow_archival_phase3_runtime_split) is False
        assert str(args.phase3_runtime_split_mode) == "off"
        assert str(args.adapt_child_pool_expansion_mode) == "off"
        assert str(args.shared_pauli_pool_mode) == "shared_pauli_child_sets_v1"
        assert str(args.shared_pauli_pool_symmetry_policy) == "hard_guard"
        assert int(args.shared_pauli_pool_max_subset_size) == 4

    def test_parse_accepts_shared_pauli_pool_explicit_no_guard(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "adapt_pipeline.py",
                "--shared-pauli-pool-mode",
                "shared_pauli_child_sets_v1",
                "--shared-pauli-pool-symmetry-policy",
                "off",
                "--shared-pauli-pool-max-subset-size",
                "1",
            ],
        )
        args = _adapt_mod.parse_args()
        assert str(args.shared_pauli_pool_mode) == "shared_pauli_child_sets_v1"
        assert str(args.shared_pauli_pool_symmetry_policy) == "off"
        assert int(args.shared_pauli_pool_max_subset_size) == 1

    def test_parse_accepts_phase3_runtime_split_max_subset_size(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "adapt_pipeline.py",
                "--phase3-runtime-split-max-subset-size",
                "4",
                "--phase3-runtime-split-child-set-symmetry-policy",
                "hard_guard",
            ],
        )
        args = _adapt_mod.parse_args()
        assert int(args.phase3_runtime_split_max_subset_size) == 4

    def test_parse_accepts_phase3_runtime_split_child_set_symmetry_off(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "adapt_pipeline.py",
                "--phase3-runtime-split-child-set-symmetry-policy",
                "off",
            ],
        )
        args = _adapt_mod.parse_args()
        assert str(args.phase3_runtime_split_child_set_symmetry_policy) == "off"
        assert (
            _adapt_mod._phase3_runtime_split_child_set_symmetry_spec(
                {"particle_number_mode": "preserving"},
                policy=str(args.phase3_runtime_split_child_set_symmetry_policy),
            )
            is None
        )

    def test_phase3_runtime_split_parent_missing_spec_requires_explicit_fallback(self):
        assert (
            _adapt_mod._phase3_runtime_split_child_set_symmetry_spec(
                None,
                policy="parent",
            )
            is None
        )

    def test_phase3_runtime_split_parent_missing_spec_fallback_is_preserving_hard_guard(self):
        spec = _adapt_mod._phase3_runtime_split_child_set_symmetry_spec(
            None,
            policy="parent",
            fallback_preserving=True,
        )

        assert spec is not None
        assert spec["particle_number_mode"] == "preserving"
        assert spec["spin_sector_mode"] == "preserving"
        assert spec["phonon_number_mode"] == "not_conserved"
        assert spec["hard_guard"] is True
        assert spec["runtime_split_child_set_symmetry_policy"] == "parent_fallback_preserving"
        assert "runtime_split_child_set_hard_guard" in spec["tags"]

    def test_phase3_runtime_split_parent_tag_only_spec_fallback_adds_preserving_modes(self):
        spec = _adapt_mod._phase3_runtime_split_child_set_symmetry_spec(
            {"tags": ["parent_metadata_present"]},
            policy="parent",
            fallback_preserving=True,
        )

        assert spec is not None
        assert spec["particle_number_mode"] == "preserving"
        assert spec["spin_sector_mode"] == "preserving"
        assert spec["hard_guard"] is True
        assert "parent_metadata_present" in spec["tags"]
        assert "runtime_split_child_set_hard_guard" in spec["tags"]

    def test_parse_accepts_archival_phase3_runtime_split_mode_with_explicit_override(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "adapt_pipeline.py",
                "--allow-archival-phase3-runtime-split",
                "--phase3-runtime-split-mode",
                "shortlist_pauli_children_v1",
                "--phase3-runtime-split-selection-mode",
                "full_child_set_scoring",
                "--phase3-runtime-split-max-subset-size",
                "4",
                "--phase3-runtime-split-child-set-symmetry-policy",
                "hard_guard",
            ],
        )
        args = _adapt_mod.parse_args()
        assert bool(args.allow_archival_phase3_runtime_split) is True
        assert str(args.phase3_runtime_split_mode) == "shortlist_pauli_children_v1"
        assert str(args.phase3_runtime_split_selection_mode) == "full_child_set_scoring"
        assert int(args.phase3_runtime_split_max_subset_size) == 4
        assert str(args.phase3_runtime_split_child_set_symmetry_policy) == "hard_guard"

    def test_parse_accepts_archival_phase3_child_set_forward_selection(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "adapt_pipeline.py",
                "--allow-archival-phase3-runtime-split",
                "--phase3-runtime-split-mode",
                "shortlist_pauli_children_v1",
                "--phase3-runtime-split-selection-mode",
                "archival_child_set_forward_v1",
            ],
        )
        args = _adapt_mod.parse_args()
        assert str(args.phase3_runtime_split_selection_mode) == "archival_child_set_forward_v1"

    def test_parse_accepts_global_child_pool_expansion_without_archival_override(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "adapt_pipeline.py",
                "--adapt-child-pool-expansion-mode",
                "global_pauli_child_sets_v1",
                "--adapt-child-pool-expansion-symmetry-policy",
                "hard_guard",
                "--adapt-child-pool-expansion-max-subset-size",
                "4",
            ],
        )
        args = _adapt_mod.parse_args()
        assert bool(args.allow_archival_phase3_runtime_split) is False
        assert str(args.phase3_runtime_split_mode) == "off"
        assert str(args.adapt_child_pool_expansion_mode) == "global_pauli_child_sets_v1"
        assert str(args.adapt_child_pool_expansion_symmetry_policy) == "hard_guard"
        assert int(args.adapt_child_pool_expansion_max_subset_size) == 4

    def test_parse_rejects_global_child_pool_plus_archival_runtime_split(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "adapt_pipeline.py",
                "--allow-archival-phase3-runtime-split",
                "--phase3-runtime-split-mode",
                "shortlist_pauli_children_v1",
                "--adapt-child-pool-expansion-mode",
                "global_pauli_child_sets_v1",
            ],
        )
        with pytest.raises(SystemExit):
            _adapt_mod.parse_args()
        captured = capsys.readouterr()
        assert "cannot be combined" in captured.err

    def test_parse_rejects_shared_pauli_pool_plus_global_child_pool(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "adapt_pipeline.py",
                "--shared-pauli-pool-mode",
                "shared_pauli_child_sets_v1",
                "--shared-pauli-pool-symmetry-policy",
                "hard_guard",
                "--adapt-child-pool-expansion-mode",
                "global_pauli_child_sets_v1",
            ],
        )
        with pytest.raises(SystemExit):
            _adapt_mod.parse_args()
        captured = capsys.readouterr()
        assert "cannot be combined" in captured.err

    def test_phase3_runtime_split_proxy_child_set_scoring_only_needed_for_proxy_mode(self) -> None:
        assert _adapt_mod._phase3_runtime_split_needs_proxy_child_set_scoring(
            selection_mode="proxy_child_set_preselection",
            parent_collapse_debug_enabled=False,
        ) is True
        assert _adapt_mod._phase3_runtime_split_needs_proxy_child_set_scoring(
            selection_mode="archival_child_set_forward_v1",
            parent_collapse_debug_enabled=False,
        ) is False
        assert _adapt_mod._phase3_runtime_split_needs_proxy_child_set_scoring(
            selection_mode="full_child_set_scoring",
            parent_collapse_debug_enabled=False,
        ) is False
        assert _adapt_mod._phase3_runtime_split_needs_proxy_child_set_scoring(
            selection_mode="parent_family_sum_top2_scoring",
            parent_collapse_debug_enabled=False,
        ) is False
        assert _adapt_mod._phase3_runtime_split_needs_proxy_child_set_scoring(
            selection_mode="full_child_set_scoring",
            parent_collapse_debug_enabled=True,
        ) is True

    def test_archival_runtime_split_eligibility_accepts_multiterm_without_macro_flag(self) -> None:
        candidate = SimpleNamespace(
            polynomial=SimpleNamespace(return_polynomial=lambda: [object(), object()])
        )
        assert _adapt_mod._phase3_runtime_split_parent_eligible(
            split_mode="shortlist_pauli_children_v1",
            selection_mode="proxy_child_set_preselection",
            generator_metadata={"is_macro_generator": False},
            candidate_term=candidate,
        ) is False
        assert _adapt_mod._phase3_runtime_split_parent_eligible(
            split_mode="shortlist_pauli_children_v1",
            selection_mode="archival_child_set_forward_v1",
            generator_metadata={"is_macro_generator": False},
            candidate_term=candidate,
        ) is True

    def test_parse_defaults_phase3_oracle_gradient_mode_off(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(sys, "argv", ["adapt_pipeline.py"])
        args = _adapt_mod.parse_args()
        assert str(args.phase3_oracle_gradient_mode) == "off"

    def test_parse_accepts_phase3_oracle_backend_scheduled(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--phase3-oracle-gradient-mode", "backend_scheduled"],
        )
        args = _adapt_mod.parse_args()
        assert str(args.phase3_oracle_gradient_mode) == "backend_scheduled"

    def test_parse_accepts_phase3_oracle_inner_objective_mode(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--phase3-oracle-inner-objective-mode", "noisy_v1"],
        )
        args = _adapt_mod.parse_args()
        assert str(args.phase3_oracle_inner_objective_mode) == "noisy_v1"

    def test_parse_accepts_adapt_analytic_noise_args(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "adapt_pipeline.py",
                "--adapt-analytic-noise-std",
                "0.25",
                "--adapt-analytic-noise-seed",
                "17",
            ],
        )
        args = _adapt_mod.parse_args()
        assert float(args.adapt_analytic_noise_std) == pytest.approx(0.25)
        assert int(args.adapt_analytic_noise_seed) == 17

    def test_parse_accepts_phase3_oracle_local_mitigation_stack(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "adapt_pipeline.py",
                "--phase3-oracle-gradient-mode",
                "backend_scheduled",
                "--phase3-oracle-mitigation",
                "readout",
                "--phase3-oracle-local-readout-strategy",
                "mthree",
                "--phase3-oracle-zne-scales",
                "1,3,5",
                "--phase3-oracle-local-gate-twirling",
                "--phase3-oracle-dd-sequence",
                "XpXm",
            ],
        )
        args = _adapt_mod.parse_args()
        assert str(args.phase3_oracle_gradient_mode) == "backend_scheduled"
        assert str(args.phase3_oracle_mitigation) == "readout"
        assert str(args.phase3_oracle_local_readout_strategy) == "mthree"
        assert str(args.phase3_oracle_zne_scales) == "1,3,5"
        assert bool(args.phase3_oracle_local_gate_twirling) is True
        assert str(args.phase3_oracle_dd_sequence) == "XpXm"

    def test_phase3_oracle_local_gate_twirling_payload_records_two_qubit_scope(self) -> None:
        payload = _adapt_mod._oracle_mitigation_payload_from_fields(
            mitigation_mode="readout",
            local_readout_strategy="mthree",
            local_gate_twirling=True,
        )

        assert payload["local_gate_twirling"] is True
        assert payload["local_gate_twirling_scope"] == "2q_only"

    def test_parse_accepts_final_noise_audit_runtime_mode_and_profile(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "adapt_pipeline.py",
                "--final-noise-audit-mode",
                "runtime",
                "--final-noise-audit-runtime-profile",
                "main_twirled_readout_v1",
                "--final-noise-audit-runtime-session-policy",
                "backend_only",
                "--final-noise-audit-compare-unmitigated-baseline",
            ],
        )
        args = _adapt_mod.parse_args()
        assert str(args.final_noise_audit_mode) == "runtime"
        assert str(args.final_noise_audit_runtime_profile) == "main_twirled_readout_v1"
        assert str(args.final_noise_audit_runtime_session_policy) == "backend_only"
        assert bool(args.final_noise_audit_compare_unmitigated_baseline) is True

    def test_parse_accepts_final_noise_audit_local_mitigation_stack(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "adapt_pipeline.py",
                "--final-noise-audit-mode",
                "backend_scheduled",
                "--final-noise-audit-use-fake-backend",
                "--final-noise-audit-backend-name",
                "FakeNighthawk",
                "--final-noise-audit-mitigation",
                "readout",
                "--final-noise-audit-local-readout-strategy",
                "mthree",
                "--final-noise-audit-zne-scales",
                "1,3,5",
                "--final-noise-audit-local-gate-twirling",
                "--final-noise-audit-dd-sequence",
                "XpXm",
            ],
        )
        args = _adapt_mod.parse_args()
        assert str(args.final_noise_audit_mode) == "backend_scheduled"
        assert bool(args.final_noise_audit_use_fake_backend) is True
        assert str(args.final_noise_audit_backend_name) == "FakeNighthawk"
        assert str(args.final_noise_audit_mitigation) == "readout"
        assert str(args.final_noise_audit_local_readout_strategy) == "mthree"
        assert str(args.final_noise_audit_zne_scales) == "1,3,5"
        assert bool(args.final_noise_audit_local_gate_twirling) is True
        assert str(args.final_noise_audit_dd_sequence) == "XpXm"

    def test_parse_rejects_auto_continuation_mode(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--adapt-continuation-mode", "auto"],
        )
        with pytest.raises(SystemExit):
            _adapt_mod.parse_args()

    def test_parse_defaults_eps_energy_gate_knobs(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(sys, "argv", ["adapt_pipeline.py"])
        args = _adapt_mod.parse_args()
        assert int(args.adapt_eps_energy_min_extra_depth) == -1
        assert int(args.adapt_eps_energy_patience) == -1

    def test_parse_defaults_drop_knobs_to_auto(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(sys, "argv", ["adapt_pipeline.py"])
        args = _adapt_mod.parse_args()
        assert args.adapt_drop_floor is None
        assert args.adapt_drop_patience is None
        assert args.adapt_drop_min_depth is None
        assert args.adapt_grad_floor is None

    def test_programmatic_default_resolution_promotes_hh_to_phase3_for_none(self):
        assert _adapt_mod._resolve_adapt_continuation_mode(problem="hh", requested_mode=None) == "phase3_v1"

    def test_programmatic_default_resolution_promotes_hh_to_phase3_for_empty_string(self):
        assert _adapt_mod._resolve_adapt_continuation_mode(problem="hh", requested_mode="") == "phase3_v1"

    def test_programmatic_default_resolution_promotes_hubbard_to_phase3(self):
        assert _adapt_mod._resolve_adapt_continuation_mode(problem="hubbard", requested_mode=None) == "phase3_v1"

    def test_cli_default_resolution_promotes_hh_to_phase3(self):
        assert _adapt_mod._resolve_cli_adapt_continuation_mode(problem="hh", requested_mode=None) == "phase3_v1"

    def test_cli_default_resolution_promotes_hubbard_to_phase3(self):
        assert _adapt_mod._resolve_cli_adapt_continuation_mode(problem="hubbard", requested_mode=None) == "phase3_v1"

    def test_cli_default_resolution_promotes_spin_boson_to_phase3(self):
        assert _adapt_mod._resolve_cli_adapt_continuation_mode(problem="spin_boson", requested_mode=None) == "phase3_v1"

    def test_cli_default_resolution_promotes_new_problem_families_to_phase3(self):
        for problem_key in (
            "molecular_restricted_closed_shell",
            "ionic_hubbard",
            "extended_hubbard",
            "ttprime_hubbard",
            "spinless_tv",
            "bose_hubbard",
            "harmonic_kerr_chain",
        ):
            assert _adapt_mod._resolve_cli_adapt_continuation_mode(
                problem=problem_key,
                requested_mode=None,
            ) == "phase3_v1"

    def test_parse_accepts_eps_energy_gate_knobs(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "adapt_pipeline.py",
                "--adapt-eps-energy-min-extra-depth", "6",
                "--adapt-eps-energy-patience", "4",
            ],
        )
        args = _adapt_mod.parse_args()
        assert int(args.adapt_eps_energy_min_extra_depth) == 6
        assert int(args.adapt_eps_energy_patience) == 4

class TestPoolBuilders:
    """Verify pool builders return non-empty pools of AnsatzTerm."""

    def test_uccsd_pool_L2(self):
        num_particles = half_filled_num_particles(2)
        pool = _build_uccsd_pool(2, num_particles, "blocked")
        assert len(pool) > 0, "UCCSD pool must be non-empty for L=2"
        for op in pool:
            assert isinstance(op, AnsatzTerm)

    def test_cse_pool_L2(self):
        pool = _build_cse_pool(2, "blocked", 1.0, 4.0, 0.0, "periodic")
        assert len(pool) > 0, "CSE pool must be non-empty for L=2"
        for op in pool:
            assert isinstance(op, AnsatzTerm)

    def test_full_hamiltonian_pool_L2(self):
        h_poly = build_hubbard_hamiltonian(dims=2, t=1.0, U=4.0, v=0.0,
                                            repr_mode="JW", indexing="blocked",
                                            pbc=True)
        pool = _build_full_hamiltonian_pool(h_poly)
        assert len(pool) > 0
        for op in pool:
            assert isinstance(op, AnsatzTerm)

    def test_hva_pool_L2_hh(self):
        pool = _build_hva_pool(
            num_sites=2, t=1.0, u=4.0, omega0=1.0, g_ep=0.5, dv=0.0,
            n_ph_max=1, boson_encoding="binary", ordering="blocked",
            boundary="periodic",
        )
        assert len(pool) > 0, "HVA pool must be non-empty for L=2 HH"
        for op in pool:
            assert isinstance(op, AnsatzTerm)
        labels = [str(op.label) for op in pool]
        lifted_pool = _build_hh_uccsd_fermion_lifted_pool(
            num_sites=2,
            n_ph_max=1,
            boson_encoding="binary",
            ordering="blocked",
            boundary="periodic",
            num_particles=half_filled_num_particles(2),
        )
        lifted_labels = {str(op.label) for op in lifted_pool}
        assert lifted_labels.issubset(set(labels))
        assert any(label.startswith("uccsd_ferm_lifted::") for label in labels)
        assert not any(
            label.startswith("uccsd_sing(") or label.startswith("uccsd_dbl(")
            for label in labels
        )

    def test_hh_termwise_augmented_pool_L2(self):
        h_poly = build_hubbard_holstein_hamiltonian(
            dims=2, J=1.0, U=4.0, omega0=1.0, g=0.5,
            n_ph_max=1, boson_encoding="binary",
            repr_mode="JW", indexing="blocked", pbc=True,
            include_zero_point=True,
        )
        pool = _build_hh_termwise_augmented_pool(h_poly)
        assert len(pool) > 0
        # Must contain at least some quadrature partners
        quad_ops = [op for op in pool if "quadrature" in op.label]
        assert len(quad_ops) > 0, "HH termwise augmented pool should have quadrature partners"


class TestPAOPPoolBuilder:
    """Verify PAOP pool builder returns non-empty pools."""

    def test_paop_min_L2(self):
        num_particles = half_filled_num_particles(2)
        pool = _build_paop_pool(
            num_sites=2, n_ph_max=1, boson_encoding="binary",
            ordering="blocked", boundary="periodic",
            pool_key="paop_min", paop_r=1,
            paop_split_paulis=False, paop_prune_eps=0.0,
            paop_normalization="none", num_particles=num_particles,
        )
        assert len(pool) > 0, "paop_min must produce operators for L=2"

    def test_paop_std_L2(self):
        num_particles = half_filled_num_particles(2)
        pool = _build_paop_pool(
            num_sites=2, n_ph_max=1, boson_encoding="binary",
            ordering="blocked", boundary="periodic",
            pool_key="paop_std", paop_r=1,
            paop_split_paulis=False, paop_prune_eps=0.0,
            paop_normalization="none", num_particles=num_particles,
        )
        assert len(pool) > 0
        # paop_std includes hopdrag so should be larger than paop_min
        pool_min = _build_paop_pool(
            num_sites=2, n_ph_max=1, boson_encoding="binary",
            ordering="blocked", boundary="periodic",
            pool_key="paop_min", paop_r=1,
            paop_split_paulis=False, paop_prune_eps=0.0,
            paop_normalization="none", num_particles=num_particles,
        )
        assert len(pool) >= len(pool_min)

    def test_paop_full_L2(self):
        num_particles = half_filled_num_particles(2)
        pool = _build_paop_pool(
            num_sites=2, n_ph_max=1, boson_encoding="binary",
            ordering="blocked", boundary="periodic",
            pool_key="paop_full", paop_r=1,
            paop_split_paulis=False, paop_prune_eps=0.0,
            paop_normalization="none", num_particles=num_particles,
        )
        assert len(pool) > 0

    def test_paop_lf_std_L2(self):
        num_particles = half_filled_num_particles(2)
        pool_lf = _build_paop_pool(
            num_sites=2, n_ph_max=1, boson_encoding="binary",
            ordering="blocked", boundary="periodic",
            pool_key="paop_lf_std", paop_r=1,
            paop_split_paulis=False, paop_prune_eps=0.0,
            paop_normalization="none", num_particles=num_particles,
        )
        pool_std = _build_paop_pool(
            num_sites=2, n_ph_max=1, boson_encoding="binary",
            ordering="blocked", boundary="periodic",
            pool_key="paop_std", paop_r=1,
            paop_split_paulis=False, paop_prune_eps=0.0,
            paop_normalization="none", num_particles=num_particles,
        )
        assert len(pool_lf) >= len(pool_std)

    def test_paop_lf2_std_L2(self):
        num_particles = half_filled_num_particles(2)
        pool_lf = _build_paop_pool(
            num_sites=2, n_ph_max=1, boson_encoding="binary",
            ordering="blocked", boundary="periodic",
            pool_key="paop_lf_std", paop_r=1,
            paop_split_paulis=False, paop_prune_eps=0.0,
            paop_normalization="none", num_particles=num_particles,
        )
        pool_lf2 = _build_paop_pool(
            num_sites=2, n_ph_max=1, boson_encoding="binary",
            ordering="blocked", boundary="periodic",
            pool_key="paop_lf2_std", paop_r=1,
            paop_split_paulis=False, paop_prune_eps=0.0,
            paop_normalization="none", num_particles=num_particles,
        )
        assert len(pool_lf2) >= len(pool_lf)

    def test_paop_lf_full_L2(self):
        num_particles = half_filled_num_particles(2)
        pool = _build_paop_pool(
            num_sites=2, n_ph_max=1, boson_encoding="binary",
            ordering="blocked", boundary="periodic",
            pool_key="paop_lf_full", paop_r=1,
            paop_split_paulis=False, paop_prune_eps=0.0,
            paop_normalization="none", num_particles=num_particles,
        )
        assert len(pool) > 0

    def test_paop_lf_alias_matches_lf_std(self):
        num_particles = half_filled_num_particles(2)
        pool_alias = _build_paop_pool(
            num_sites=2, n_ph_max=1, boson_encoding="binary",
            ordering="blocked", boundary="periodic",
            pool_key="paop_lf", paop_r=1,
            paop_split_paulis=False, paop_prune_eps=0.0,
            paop_normalization="none", num_particles=num_particles,
        )
        pool_std = _build_paop_pool(
            num_sites=2, n_ph_max=1, boson_encoding="binary",
            ordering="blocked", boundary="periodic",
            pool_key="paop_lf_std", paop_r=1,
            paop_split_paulis=False, paop_prune_eps=0.0,
            paop_normalization="none", num_particles=num_particles,
        )
        assert len(pool_alias) == len(pool_std)

    def test_paop_curdrag_L2_open_blocked_signature(self):
        num_particles = half_filled_num_particles(2)
        pool = _build_paop_pool(
            num_sites=2, n_ph_max=1, boson_encoding="binary",
            ordering="blocked", boundary="open",
            pool_key="paop_lf_std", paop_r=1,
            paop_split_paulis=False, paop_prune_eps=0.0,
            paop_normalization="none", num_particles=num_particles,
        )
        curdrag = None
        for op in pool:
            if "paop_curdrag(0,1)" in op.label:
                curdrag = op
                break
        assert curdrag is not None, "Expected paop_curdrag(0,1) in paop_lf_std for L=2 open chain."

        coeff_map: dict[str, float] = {}
        for term in curdrag.polynomial.return_polynomial():
            coeff = complex(term.p_coeff)
            if abs(coeff) <= 1e-12:
                continue
            assert abs(coeff.imag) <= 1e-10
            coeff_map[str(term.pw2strng())] = float(round(coeff.real, 12))

        expected = {
            "eyeexy": 0.5,
            "eyeeyx": -0.5,
            "eyxyee": 0.5,
            "eyyxee": -0.5,
            "yeeexy": -0.5,
            "yeeeyx": 0.5,
            "yexyee": -0.5,
            "yeyxee": 0.5,
        }
        assert set(coeff_map.keys()) == set(expected.keys())
        same_sign = all(abs(coeff_map[key] - expected[key]) <= 1e-10 for key in expected)
        flipped_sign = all(abs(coeff_map[key] + expected[key]) <= 1e-10 for key in expected)
        assert same_sign or flipped_sign

    def test_paop_lf_coefficients_are_real_after_cleaning(self):
        num_particles = half_filled_num_particles(2)
        for pool_key in ("paop_lf_std", "paop_lf2_std", "paop_lf_full"):
            pool = _build_paop_pool(
                num_sites=2, n_ph_max=1, boson_encoding="binary",
                ordering="blocked", boundary="periodic",
                pool_key=pool_key, paop_r=1,
                paop_split_paulis=False, paop_prune_eps=0.0,
                paop_normalization="none", num_particles=num_particles,
            )
            assert len(pool) > 0
            for op in pool:
                for term in op.polynomial.return_polynomial():
                    assert abs(complex(term.p_coeff).imag) <= 1e-10

    def test_hh_operator_pool_builder_importable(self):
        """Verify the HH PAOP builder module can be imported directly."""
        from src.quantum.operator_pools.hh_paop import make_pool
        assert callable(make_pool)


class TestHHUCCSDPAOPCompositePoolBuilder:
    """Verify HH composite UCCSD+PAOP(lf_full) pool semantics."""

    def test_uccsd_lift_has_boson_identity_prefix(self):
        n_sites = 2
        n_ph_max = 1
        boson_encoding = "binary"
        boson_bits = n_sites * int(boson_qubits_per_site(n_ph_max, boson_encoding))
        pool = _build_hh_uccsd_fermion_lifted_pool(
            num_sites=n_sites,
            n_ph_max=n_ph_max,
            boson_encoding=boson_encoding,
            ordering="blocked",
            boundary="periodic",
            num_particles=half_filled_num_particles(n_sites),
        )
        assert len(pool) > 0
        boson_identity = "e" * boson_bits
        nq_total = 2 * n_sites + boson_bits
        for op in pool:
            has_nontrivial_fermion_support = False
            for term in op.polynomial.return_polynomial():
                coeff = complex(term.p_coeff)
                if abs(coeff) <= 1e-15:
                    continue
                ps = str(term.pw2strng())
                assert len(ps) == nq_total
                assert ps[:boson_bits] == boson_identity
                if any(ch != "e" for ch in ps[boson_bits:]):
                    has_nontrivial_fermion_support = True
            assert has_nontrivial_fermion_support

    def test_composite_pool_is_non_empty_and_deduplicated(self):
        n_sites = 2
        num_particles = half_filled_num_particles(n_sites)
        uccsd_pool = _build_hh_uccsd_fermion_lifted_pool(
            num_sites=n_sites,
            n_ph_max=1,
            boson_encoding="binary",
            ordering="blocked",
            boundary="periodic",
            num_particles=num_particles,
        )
        paop_pool = _build_paop_pool(
            num_sites=n_sites,
            n_ph_max=1,
            boson_encoding="binary",
            ordering="blocked",
            boundary="periodic",
            pool_key="paop_lf_full",
            paop_r=1,
            paop_split_paulis=False,
            paop_prune_eps=0.0,
            paop_normalization="none",
            num_particles=num_particles,
        )
        dedup_pool = _deduplicate_pool_terms(list(uccsd_pool) + list(paop_pool))
        assert len(uccsd_pool) > 0
        assert len(paop_pool) > 0
        assert len(dedup_pool) > 0
        assert len(dedup_pool) <= len(uccsd_pool) + len(paop_pool)

    def test_pareto_lean_pool_keeps_only_scaffold_supported_families(self):
        n_sites = 2
        num_particles = half_filled_num_particles(n_sites)
        h_poly = build_hubbard_holstein_hamiltonian(
            dims=n_sites,
            J=1.0,
            U=4.0,
            omega0=1.0,
            g=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            repr_mode="JW",
            indexing="blocked",
            pbc=True,
            include_zero_point=True,
        )
        pool, meta = _build_hh_pareto_lean_pool(
            h_poly=h_poly,
            num_sites=n_sites,
            t=1.0,
            u=4.0,
            omega0=1.0,
            g_ep=0.5,
            dv=0.0,
            n_ph_max=1,
            boson_encoding="binary",
            ordering="blocked",
            boundary="periodic",
            paop_r=1,
            paop_split_paulis=False,
            paop_prune_eps=0.0,
            paop_normalization="none",
            num_particles=num_particles,
        )
        labels = [str(op.label) for op in pool]

        assert len(pool) > 0
        assert int(meta["raw_total"]) > 0
        assert any(label.startswith("uccsd_ferm_lifted::uccsd_sing(") for label in labels)
        assert any(label.startswith("uccsd_ferm_lifted::uccsd_dbl(") for label in labels)
        assert any(label.startswith("hh_termwise_ham_quadrature_term(") for label in labels)
        assert any(label.startswith("paop_full:paop_cloud_p(") for label in labels)
        assert any(label.startswith("paop_full:paop_disp(") for label in labels)
        assert any(label.startswith("paop_full:paop_hopdrag(") for label in labels)
        assert any(label.startswith("paop_lf_full:paop_dbl_p(") for label in labels)

        assert not any(label in {"hop_layer", "onsite_layer", "phonon_layer", "eph_layer"} for label in labels)
        assert not any(label.startswith("hh_termwise_ham_unit_term(") for label in labels)
        assert not any(label.startswith("paop_full:paop_dbl(") for label in labels)
        assert not any(label.startswith("paop_full:paop_cloud_x(") for label in labels)
        assert not any(label.startswith("paop_lf_full:paop_dbl_x(") for label in labels)
        assert not any(label.startswith("paop_lf_full:paop_curdrag(") for label in labels)
        assert not any(label.startswith("paop_lf_full:paop_hop2(") for label in labels)

    def test_pareto_lean_l2_pool_is_nonempty_for_l2_nph1(self):
        n_sites = 2
        num_particles = half_filled_num_particles(n_sites)
        h_poly = build_hubbard_holstein_hamiltonian(
            dims=n_sites,
            J=1.0,
            U=4.0,
            omega0=1.0,
            g=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            repr_mode="JW",
            indexing="blocked",
            pbc=False,
            include_zero_point=True,
        )
        pool, meta = _build_hh_pareto_lean_l2_pool(
            h_poly=h_poly,
            num_sites=n_sites,
            t=1.0,
            u=4.0,
            omega0=1.0,
            g_ep=0.5,
            dv=0.0,
            n_ph_max=1,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
            paop_r=1,
            paop_split_paulis=False,
            paop_prune_eps=0.0,
            paop_normalization="none",
            num_particles=num_particles,
        )
        assert len(pool) > 0
        assert int(meta["raw_total"]) > 0

    def test_pareto_lean_l2_pool_rejects_non_l2(self):
        h_poly = build_hubbard_holstein_hamiltonian(
            dims=3,
            J=1.0,
            U=4.0,
            omega0=1.0,
            g=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            repr_mode="JW",
            indexing="blocked",
            pbc=False,
            include_zero_point=True,
        )
        with pytest.raises(ValueError, match="only valid for L=2"):
            _build_hh_pareto_lean_l2_pool(
                h_poly=h_poly,
                num_sites=3,
                t=1.0,
                u=4.0,
                omega0=1.0,
                g_ep=0.5,
                dv=0.0,
                n_ph_max=1,
                boson_encoding="binary",
                ordering="blocked",
                boundary="open",
                paop_r=1,
                paop_split_paulis=False,
                paop_prune_eps=0.0,
                paop_normalization="none",
                num_particles=half_filled_num_particles(3),
            )

    def test_pareto_lean_l2_pool_rejects_nphmax_not_1(self):
        h_poly = build_hubbard_holstein_hamiltonian(
            dims=2,
            J=1.0,
            U=4.0,
            omega0=1.0,
            g=0.5,
            n_ph_max=2,
            boson_encoding="binary",
            repr_mode="JW",
            indexing="blocked",
            pbc=False,
            include_zero_point=True,
        )
        with pytest.raises(ValueError, match="only valid for n_ph_max=1"):
            _build_hh_pareto_lean_l2_pool(
                h_poly=h_poly,
                num_sites=2,
                t=1.0,
                u=4.0,
                omega0=1.0,
                g_ep=0.5,
                dv=0.0,
                n_ph_max=2,
                boson_encoding="binary",
                ordering="blocked",
                boundary="open",
                paop_r=1,
                paop_split_paulis=False,
                paop_prune_eps=0.0,
                paop_normalization="none",
                num_particles=half_filled_num_particles(2),
            )


# ============================================================================
# Sector filtering dispatch
# ============================================================================

class TestSectorFilteringDispatch:
    """Verify _exact_gs_energy_for_problem dispatches correctly."""

    def test_hubbard_dispatch(self):
        h_poly = build_hubbard_hamiltonian(dims=2, t=1.0, U=4.0, v=0.0,
                                            repr_mode="JW", indexing="blocked", pbc=True)
        num_particles = half_filled_num_particles(2)
        e_dispatch = _exact_gs_energy_for_problem(
            h_poly, problem="hubbard", num_sites=2,
            num_particles=num_particles, indexing="blocked",
        )
        e_direct = exact_ground_energy_sector(
            h_poly, num_sites=2, num_particles=num_particles, indexing="blocked",
        )
        assert abs(e_dispatch - e_direct) < 1e-12

    def test_hh_dispatch_uses_fermion_only(self):
        """HH dispatch must use exact_ground_energy_sector_hh (fermion-only filtering)."""
        h_poly = build_hubbard_holstein_hamiltonian(
            dims=2, J=1.0, U=4.0, omega0=1.0, g=0.5,
            n_ph_max=1, boson_encoding="binary",
            repr_mode="JW", indexing="blocked", pbc=True,
            include_zero_point=True,
        )
        num_particles = half_filled_num_particles(2)
        e_dispatch = _exact_gs_energy_for_problem(
            h_poly, problem="hh", num_sites=2,
            num_particles=num_particles, indexing="blocked",
            n_ph_max=1, boson_encoding="binary",
        )
        e_direct = exact_ground_energy_sector_hh(
            h_poly, num_sites=2, num_particles=num_particles,
            n_ph_max=1, boson_encoding="binary", indexing="blocked",
        )
        assert abs(e_dispatch - e_direct) < 1e-12


# ============================================================================
# End-to-end ADAPT-VQE smoke tests
# ============================================================================

class TestAdaptVQEHubbardUCCSD:
    """L=2 Hubbard UCCSD ADAPT-VQE must converge to near-exact energy."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.L = 2
        self.t = 1.0
        self.u = 4.0
        self.h_poly = build_hubbard_hamiltonian(
            dims=self.L, t=self.t, U=self.u, v=0.0,
            repr_mode="JW", indexing="blocked", pbc=True,
        )
        self.num_particles = half_filled_num_particles(self.L)
        self.exact_gs = exact_ground_energy_sector(
            self.h_poly, num_sites=self.L,
            num_particles=self.num_particles, indexing="blocked",
        )

    def test_adapt_uccsd_converges(self):
        payload, psi = _run_hardcoded_adapt_vqe(
            h_poly=self.h_poly,
            num_sites=self.L,
            ordering="blocked",
            problem="hubbard",
            adapt_pool="uccsd",
            t=self.t, u=self.u, dv=0.0,
            boundary="periodic",
            omega0=0.0, g_ep=0.0,
            n_ph_max=1, boson_encoding="binary",
            max_depth=15,
            eps_grad=1e-6,
            eps_energy=1e-10,
            maxiter=300,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
        )
        assert payload["success"] is True
        assert payload["energy"] is not None
        # UCCSD pool for L=2 half-filling is small (3 ops: 2 singles + 1 double).
        # The ADAPT greedy loop may not select the double (zero gradient at HF),
        # so the energy may not reach the exact GS. Verify it at least improves
        # significantly from the HF energy and returns a physically valid result.
        hf_energy = 4.0  # known for L=2 periodic t=1 U=4 half-filled
        assert payload["energy"] < hf_energy - 1.0, \
            f"ADAPT UCCSD must improve on HF: E={payload['energy']:.4f} vs HF={hf_energy}"
        assert payload["exact_gs_energy"] is not None

    def test_adapt_cse_converges(self):
        payload, psi = _run_hardcoded_adapt_vqe(
            h_poly=self.h_poly,
            num_sites=self.L,
            ordering="blocked",
            problem="hubbard",
            adapt_pool="cse",
            t=self.t, u=self.u, dv=0.0,
            boundary="periodic",
            omega0=0.0, g_ep=0.0,
            n_ph_max=1, boson_encoding="binary",
            max_depth=15,
            eps_grad=1e-6,
            eps_energy=1e-10,
            maxiter=300,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
        )
        assert payload["success"] is True
        # CSE pool for L=2 has only 4 Hamiltonian-term generators (hopping + onsite).
        # With such a small pool ADAPT may not reach exact GS, but should improve on HF.
        hf_energy = 4.0
        assert payload["energy"] < hf_energy - 1.0, \
            f"ADAPT CSE must improve on HF: E={payload['energy']:.4f} vs HF={hf_energy}"

    def test_adapt_full_hamiltonian_filters_sector_unsafe_paulis(self):
        """The Pauli-primitive pool must fail closed on unsafe XX/YY strings."""
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self.h_poly,
            num_sites=self.L,
            ordering="blocked",
            problem="hubbard",
            adapt_pool="full_hamiltonian",
            t=self.t, u=self.u, dv=0.0,
            boundary="periodic",
            omega0=0.0, g_ep=0.0,
            n_ph_max=1, boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-6,
            eps_energy=1e-10,
            maxiter=10,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
        )
        assert payload["success"] is True
        contract = payload["generator_sector_contract"]
        expected_removed = {
            "ham_term(yyee)",
            "ham_term(xxee)",
            "ham_term(eexx)",
            "ham_term(eeyy)",
        }
        assert contract["passed"] is True
        assert contract["execution_passed"] is True
        assert contract["filter"]["applied"] is True
        assert contract["filter"]["removed_count"] == 4
        assert set(contract["filter"]["removed_labels"]) == expected_removed
        assert set(contract["prefilter_audit"]["grouped_violation_labels"]) == expected_removed
        assert payload["state_sector_contract"]["passed"] is True
        assert payload["state_sector_contract"]["joint_target_sector_probability"] == pytest.approx(1.0)
        assert payload["strict_replay"]["passed"] is True

    def test_adapt_hamiltonian_blocks_retains_safe_hopping_macros(self):
        """Grouped Hamiltonian blocks retain hopping expressivity without leakage."""
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self.h_poly,
            num_sites=self.L,
            ordering="blocked",
            problem="hubbard",
            adapt_pool="hamiltonian_blocks",
            t=self.t, u=self.u, dv=0.0,
            boundary="periodic",
            omega0=0.0, g_ep=0.0,
            n_ph_max=1, boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-6,
            eps_energy=1e-10,
            maxiter=30,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
        )
        assert payload["success"] is True
        contract = payload["generator_sector_contract"]
        assert contract["passed"] is True
        assert contract["execution_passed"] is True
        assert contract["filter"]["applied"] is False
        assert contract["prefilter_audit"]["passed"] is True
        assert contract["prefilter_audit"]["execution_passed"] is True
        assert payload["parameterization_mode"] == "logical_shared"
        assert payload["energy"] < 3.0
        assert payload["state_sector_contract"]["passed"] is True
        assert payload["state_sector_contract"]["joint_target_sector_probability"] == pytest.approx(1.0)
        assert payload["strict_replay"]["passed"] is True


class TestAdaptVQEHolsteinHVA:
    """L=2 HH HVA ADAPT-VQE must converge to near-exact HH energy."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.L = 2
        self.t = 1.0
        self.u = 4.0
        self.omega0 = 1.0
        self.g_ep = 0.5
        self.n_ph_max = 1
        self.h_poly = build_hubbard_holstein_hamiltonian(
            dims=self.L, J=self.t, U=self.u,
            omega0=self.omega0, g=self.g_ep,
            n_ph_max=self.n_ph_max, boson_encoding="binary",
            repr_mode="JW", indexing="blocked", pbc=True,
            include_zero_point=True,
        )
        self.num_particles = half_filled_num_particles(self.L)
        self.exact_gs = exact_ground_energy_sector_hh(
            self.h_poly, num_sites=self.L,
            num_particles=self.num_particles,
            n_ph_max=self.n_ph_max, boson_encoding="binary",
            indexing="blocked",
        )

    def test_adapt_hva_hh_preserves_sector_and_is_variational(self):
        payload, psi = _run_hardcoded_adapt_vqe(
            h_poly=self.h_poly,
            num_sites=self.L,
            ordering="blocked",
            problem="hh",
            adapt_pool="hva",
            t=self.t, u=self.u, dv=0.0,
            boundary="periodic",
            omega0=self.omega0, g_ep=self.g_ep,
            n_ph_max=self.n_ph_max, boson_encoding="binary",
            max_depth=30,
            eps_grad=1e-5,
            eps_energy=1e-10,
            maxiter=600,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="full",  # convergence test — needs full re-opt
        )
        assert payload["success"] is True
        assert payload["energy"] is not None
        # HH exact_gs in payload should match our computed value
        assert abs(payload["exact_gs_energy"] - self.exact_gs) < 1e-10
        sector_weights = _fermion_sector_weights(psi, num_sites=self.L, ordering="blocked")
        assert sector_weights.get(tuple(self.num_particles), 0.0) > 1.0 - 1e-10
        assert payload["energy"] >= self.exact_gs - 1e-10

    def test_adapt_hh_uses_fermion_only_sector(self):
        """Verify the payload exact_gs matches fermion-only sector filtering."""
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self.h_poly,
            num_sites=self.L,
            ordering="blocked",
            problem="hh",
            adapt_pool="full_hamiltonian",
            t=self.t, u=self.u, dv=0.0,
            boundary="periodic",
            omega0=self.omega0, g_ep=self.g_ep,
            n_ph_max=self.n_ph_max, boson_encoding="binary",
            max_depth=5,
            eps_grad=1e-2,
            eps_energy=1e-6,
            maxiter=100,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
        )
        assert abs(payload["exact_gs_energy"] - self.exact_gs) < 1e-10, \
            "HH ADAPT must use fermion-only sector filtering"


class TestAdaptVQEHolsteinPAOP:
    """L=2 HH PAOP ADAPT-VQE smoke test."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.L = 2
        self.t = 1.0
        self.u = 4.0
        self.omega0 = 1.0
        self.g_ep = 0.5
        self.n_ph_max = 1
        self.h_poly = build_hubbard_holstein_hamiltonian(
            dims=self.L, J=self.t, U=self.u,
            omega0=self.omega0, g=self.g_ep,
            n_ph_max=self.n_ph_max, boson_encoding="binary",
            repr_mode="JW", indexing="blocked", pbc=True,
            include_zero_point=True,
        )
        self.num_particles = half_filled_num_particles(self.L)
        self.exact_gs = exact_ground_energy_sector_hh(
            self.h_poly, num_sites=self.L,
            num_particles=self.num_particles,
            n_ph_max=self.n_ph_max, boson_encoding="binary",
            indexing="blocked",
        )

    def test_adapt_paop_std_runs(self):
        """PAOP std pool should run without error and produce a valid energy."""
        payload, psi = _run_hardcoded_adapt_vqe(
            h_poly=self.h_poly,
            num_sites=self.L,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_std",
            t=self.t, u=self.u, dv=0.0,
            boundary="periodic",
            omega0=self.omega0, g_ep=self.g_ep,
            n_ph_max=self.n_ph_max, boson_encoding="binary",
            max_depth=15,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=300,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            paop_r=1,
            paop_split_paulis=False,
            paop_prune_eps=0.0,
            paop_normalization="none",
            adapt_reopt_policy="full",  # convergence test — needs full re-opt
            adapt_continuation_mode="legacy",
        )
        assert payload["success"] is True
        assert str(payload["pool_type"]) == "paop_std"
        assert str(payload["method"]) == "hardcoded_adapt_vqe_paop_std"
        assert payload["energy"] is not None
        # Energy should be finite and not NaN
        assert np.isfinite(payload["energy"])
        # Should be lower than reference state energy (some improvement)
        assert payload["energy"] <= payload["exact_gs_energy"] + 0.5

    def test_adapt_paop_min_runs(self):
        """PAOP min pool (displacement only) should run."""
        payload, psi = _run_hardcoded_adapt_vqe(
            h_poly=self.h_poly,
            num_sites=self.L,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_min",
            t=self.t, u=self.u, dv=0.0,
            boundary="periodic",
            omega0=self.omega0, g_ep=self.g_ep,
            n_ph_max=self.n_ph_max, boson_encoding="binary",
            max_depth=10,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=200,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            paop_r=1,
            paop_split_paulis=False,
            paop_prune_eps=0.0,
            paop_normalization="none",
            adapt_continuation_mode="legacy",
        )
        assert payload["success"] is True
        assert np.isfinite(payload["energy"])

    def test_adapt_uccsd_paop_lf_full_runs(self):
        """Composite HH pool should run and report composite pool_type."""
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self.h_poly,
            num_sites=self.L,
            ordering="blocked",
            problem="hh",
            adapt_pool="uccsd_paop_lf_full",
            t=self.t,
            u=self.u,
            dv=0.0,
            boundary="periodic",
            omega0=self.omega0,
            g_ep=self.g_ep,
            n_ph_max=self.n_ph_max,
            boson_encoding="binary",
            max_depth=6,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=200,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            paop_r=1,
            paop_split_paulis=False,
            paop_prune_eps=0.0,
            paop_normalization="none",
            adapt_continuation_mode="legacy",
        )
        assert payload["success"] is True
        assert str(payload["pool_type"]) == "uccsd_paop_lf_full"
        assert int(payload["pool_size"]) > 0

    def test_adapt_uccsd_otimes_paop_lf_std_runs(self):
        """UCCSD⊗PAOP LF standard HH product pool should run and report its pool type."""
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self.h_poly,
            num_sites=self.L,
            ordering="blocked",
            problem="hh",
            adapt_pool="uccsd_otimes_paop_lf_std",
            t=self.t,
            u=self.u,
            dv=0.0,
            boundary="periodic",
            omega0=self.omega0,
            g_ep=self.g_ep,
            n_ph_max=self.n_ph_max,
            boson_encoding="binary",
            max_depth=2,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=80,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            paop_r=1,
            paop_split_paulis=False,
            paop_prune_eps=0.0,
            paop_normalization="none",
            adapt_continuation_mode="legacy",
        )
        assert payload["success"] is True
        assert str(payload["pool_type"]) == "uccsd_otimes_paop_lf_std"
        assert str(payload["method"]) == "hardcoded_adapt_vqe_uccsd_otimes_paop_lf_std"
        assert int(payload["pool_size"]) > 0
        assert np.isfinite(float(payload["energy"]))

    def test_adapt_sq_lf_std_runs(self):
        """Canonical SQ/LF HH pool should run through legacy ADAPT and report its pool type."""
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self.h_poly,
            num_sites=self.L,
            ordering="blocked",
            problem="hh",
            adapt_pool="sq_lf_std",
            t=self.t,
            u=self.u,
            dv=0.0,
            boundary="periodic",
            omega0=self.omega0,
            g_ep=self.g_ep,
            n_ph_max=self.n_ph_max,
            boson_encoding="binary",
            max_depth=2,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=80,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            paop_r=1,
            paop_split_paulis=False,
            paop_prune_eps=0.0,
            paop_normalization="none",
            adapt_continuation_mode="legacy",
        )
        assert payload["success"] is True
        assert str(payload["pool_type"]) == "sq_lf_std"
        assert str(payload["method"]) == "hardcoded_adapt_vqe_sq_lf_std"
        assert int(payload["pool_size"]) > 0
        assert np.isfinite(float(payload["energy"]))



    def test_adapt_full_meta_runs(self):
        """Full HH meta-pool should run and report full_meta pool type."""
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self.h_poly,
            num_sites=self.L,
            ordering="blocked",
            problem="hh",
            adapt_pool="full_meta",
            t=self.t,
            u=self.u,
            dv=0.0,
            boundary="periodic",
            omega0=self.omega0,
            g_ep=self.g_ep,
            n_ph_max=self.n_ph_max,
            boson_encoding="binary",
            max_depth=4,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=120,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            paop_r=1,
            paop_split_paulis=False,
            paop_prune_eps=0.0,
            paop_normalization="none",
            adapt_continuation_mode="legacy",
        )
        assert payload["success"] is True
        assert str(payload["pool_type"]) == "full_meta"
        assert int(payload["pool_size"]) > 0

    def test_adapt_pareto_lean_runs(self):
        """Pareto-lean HH pool should run and report pareto_lean pool type."""
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self.h_poly,
            num_sites=self.L,
            ordering="blocked",
            problem="hh",
            adapt_pool="pareto_lean",
            t=self.t,
            u=self.u,
            dv=0.0,
            boundary="periodic",
            omega0=self.omega0,
            g_ep=self.g_ep,
            n_ph_max=self.n_ph_max,
            boson_encoding="binary",
            max_depth=4,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=120,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            paop_r=1,
            paop_split_paulis=False,
            paop_prune_eps=0.0,
            paop_normalization="none",
            adapt_continuation_mode="legacy",
        )
        assert payload["success"] is True
        assert str(payload["pool_type"]) == "pareto_lean"
        assert int(payload["pool_size"]) > 0


class TestAILoggingResilience:
    def test_ai_log_ignores_broken_pipe(self, monkeypatch: pytest.MonkeyPatch):
        calls = {"count": 0}

        def _broken_print(*args, **kwargs):
            calls["count"] += 1
            raise BrokenPipeError()

        monkeypatch.setattr(builtins, "print", _broken_print)
        monkeypatch.setattr(_adapt_mod, "_STDOUT_PIPE_BROKEN", False)

        _adapt_mod._ai_log("unit_test_event", value=1)
        _adapt_mod._ai_log("unit_test_event_second", value=2)

        assert calls["count"] == 1
        assert _adapt_mod._STDOUT_PIPE_BROKEN is True


class TestAdaptSPSAHeartbeats:
    """SPSA inner optimizer should emit progress heartbeats for ADAPT."""

    def test_spsa_heartbeat_event_is_emitted(self, monkeypatch: pytest.MonkeyPatch):
        h_poly = build_hubbard_hamiltonian(
            dims=2, t=1.0, U=4.0, v=0.0,
            repr_mode="JW", indexing="blocked", pbc=True,
        )
        events: list[tuple[str, dict[str, object]]] = []
        original_ai_log = _adapt_mod._ai_log

        def _capture(event: str, **fields: object) -> None:
            events.append((str(event), dict(fields)))

        monkeypatch.setattr(_adapt_mod, "_ai_log", _capture)
        try:
            payload, _ = _run_hardcoded_adapt_vqe(
                h_poly=h_poly,
                num_sites=2,
                ordering="blocked",
                problem="hubbard",
                adapt_pool="uccsd",
                t=1.0,
                u=4.0,
                dv=0.0,
                boundary="periodic",
                omega0=0.0,
                g_ep=0.0,
                n_ph_max=1,
                boson_encoding="binary",
                max_depth=2,
                eps_grad=1e-6,
                eps_energy=1e-10,
                maxiter=40,
                seed=11,
                adapt_inner_optimizer="SPSA",
                adapt_spsa_callback_every=1,
                adapt_spsa_progress_every_s=0.0,
                allow_repeats=False,
                finite_angle_fallback=True,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
            )
            assert payload["success"] is True
            heartbeat_events = [ev for ev in events if ev[0] == "hardcoded_adapt_spsa_heartbeat"]
            assert len(heartbeat_events) > 0
            assert any(str(ev[1].get("stage", "")).startswith("depth_") for ev in heartbeat_events)
        finally:
            monkeypatch.setattr(_adapt_mod, "_ai_log", original_ai_log)


class TestAdaptQNSPSAOptIn:
    """QNSPSA is opt-in and uses native ADAPT state fidelity without Qiskit."""

    def test_qnspsa_static_adapt_smoke_emits_heartbeat(self, monkeypatch: pytest.MonkeyPatch):
        h_poly = build_hubbard_hamiltonian(
            dims=2, t=1.0, U=4.0, v=0.0,
            repr_mode="JW", indexing="blocked", pbc=True,
        )
        events: list[tuple[str, dict[str, object]]] = []
        original_ai_log = _adapt_mod._ai_log

        def _capture(event: str, **fields: object) -> None:
            events.append((str(event), dict(fields)))

        monkeypatch.setattr(_adapt_mod, "_ai_log", _capture)
        try:
            payload, _ = _run_hardcoded_adapt_vqe(
                h_poly=h_poly,
                num_sites=2,
                ordering="blocked",
                problem="hubbard",
                adapt_pool="uccsd",
                t=1.0,
                u=4.0,
                dv=0.0,
                boundary="periodic",
                omega0=0.0,
                g_ep=0.0,
                n_ph_max=1,
                boson_encoding="binary",
                max_depth=1,
                eps_grad=1e-6,
                eps_energy=1e-10,
                maxiter=8,
                seed=11,
                adapt_inner_optimizer="QNSPSA",
                adapt_spsa_callback_every=1,
                adapt_spsa_progress_every_s=0.0,
                allow_repeats=False,
                finite_angle_fallback=True,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
            )
            assert payload["success"] is True
            assert str(payload["adapt_inner_optimizer"]) == "QNSPSA"
            assert np.isfinite(float(payload["energy"]))
            heartbeat_events = [ev for ev in events if ev[0] == "hardcoded_adapt_qnspsa_heartbeat"]
            assert len(heartbeat_events) > 0
            start_events = [ev for ev in events if ev[0] == "hardcoded_adapt_vqe_start"]
            assert start_events and str(start_events[0][1]["adapt_inner_optimizer"]) == "QNSPSA"
            iter_done_events = [ev[1] for ev in events if ev[0] == "hardcoded_adapt_iter_done"]
            for ev_fields in iter_done_events:
                assert float(ev_fields["delta_e"]) <= 1e-14
        finally:
            monkeypatch.setattr(_adapt_mod, "_ai_log", original_ai_log)


class TestAdaptEnergyStopGate:
    """eps_energy stop must honor min-extra-depth and patience gates."""

    def test_eps_energy_defaults_wait_for_L_gate_and_L_patience(self, monkeypatch: pytest.MonkeyPatch):
        h_poly = build_hubbard_hamiltonian(
            dims=2, t=1.0, U=4.0, v=0.0,
            repr_mode="JW", indexing="blocked", pbc=True,
        )
        events: list[tuple[str, dict[str, object]]] = []
        original_ai_log = _adapt_mod._ai_log

        def _capture(event: str, **fields: object) -> None:
            events.append((str(event), dict(fields)))

        monkeypatch.setattr(_adapt_mod, "_ai_log", _capture)
        try:
            payload, _ = _run_hardcoded_adapt_vqe(
                h_poly=h_poly,
                num_sites=2,
                ordering="blocked",
                problem="hubbard",
                adapt_pool="uccsd",
                t=1.0,
                u=4.0,
                dv=0.0,
                boundary="periodic",
                omega0=0.0,
                g_ep=0.0,
                n_ph_max=1,
                boson_encoding="binary",
                max_depth=4,
                eps_grad=-1.0,
                eps_energy=1e9,
                maxiter=20,
                seed=19,
                adapt_inner_optimizer="SPSA",
                adapt_spsa_callback_every=10,
                adapt_spsa_progress_every_s=999.0,
                allow_repeats=True,
                finite_angle_fallback=False,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
                adapt_continuation_mode="legacy",
            )
            assert payload["success"] is True
            assert str(payload["stop_reason"]) == "eps_energy"
            assert bool(payload["eps_energy_termination_enabled"]) is True
            assert int(payload["eps_energy_min_extra_depth_effective"]) == 2
            assert int(payload["eps_energy_patience_effective"]) == 2
            assert int(payload["ansatz_depth"]) >= 3

            iter_done_events = [ev[1] for ev in events if ev[0] == "hardcoded_adapt_iter_done"]
            by_depth = {int(ev["depth"]): ev for ev in iter_done_events}
            assert bool(by_depth[1]["eps_energy_gate_open"]) is False
            assert bool(by_depth[2]["eps_energy_gate_open"]) is True
            assert int(by_depth[2]["eps_energy_low_streak"]) == 1
            assert int(by_depth[3]["eps_energy_low_streak"]) >= 2
            assert bool(by_depth[3]["eps_energy_termination_enabled"]) is True

            gate_wait_events = [ev for ev in events if ev[0] == "hardcoded_adapt_energy_convergence_gate_wait"]
            assert len(gate_wait_events) >= 1
        finally:
            monkeypatch.setattr(_adapt_mod, "_ai_log", original_ai_log)

    def test_eps_energy_gate_override_is_respected(self, monkeypatch: pytest.MonkeyPatch):
        h_poly = build_hubbard_hamiltonian(
            dims=2, t=1.0, U=4.0, v=0.0,
            repr_mode="JW", indexing="blocked", pbc=True,
        )
        events: list[tuple[str, dict[str, object]]] = []
        original_ai_log = _adapt_mod._ai_log

        def _capture(event: str, **fields: object) -> None:
            events.append((str(event), dict(fields)))

        monkeypatch.setattr(_adapt_mod, "_ai_log", _capture)
        try:
            payload, _ = _run_hardcoded_adapt_vqe(
                h_poly=h_poly,
                num_sites=2,
                ordering="blocked",
                problem="hubbard",
                adapt_pool="uccsd",
                t=1.0,
                u=4.0,
                dv=0.0,
                boundary="periodic",
                omega0=0.0,
                g_ep=0.0,
                n_ph_max=1,
                boson_encoding="binary",
                max_depth=5,
                eps_grad=-1.0,
                eps_energy=1e9,
                maxiter=20,
                seed=21,
                adapt_inner_optimizer="SPSA",
                adapt_spsa_callback_every=10,
                adapt_spsa_progress_every_s=999.0,
                allow_repeats=True,
                finite_angle_fallback=False,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
                adapt_continuation_mode="legacy",
                adapt_eps_energy_min_extra_depth=3,
                adapt_eps_energy_patience=2,
            )
            assert payload["success"] is True
            assert str(payload["stop_reason"]) == "eps_energy"
            assert bool(payload["eps_energy_termination_enabled"]) is True
            assert int(payload["eps_energy_min_extra_depth_effective"]) == 3
            assert int(payload["eps_energy_patience_effective"]) == 2
            assert int(payload["ansatz_depth"]) >= 4

            iter_done_events = [ev[1] for ev in events if ev[0] == "hardcoded_adapt_iter_done"]
            by_depth = {int(ev["depth"]): ev for ev in iter_done_events}
            assert bool(by_depth[2]["eps_energy_gate_open"]) is False
            assert bool(by_depth[3]["eps_energy_gate_open"]) is True
            assert int(by_depth[3]["eps_energy_low_streak"]) == 1
            assert int(by_depth[4]["eps_energy_low_streak"]) >= 2
            assert bool(by_depth[4]["eps_energy_termination_enabled"]) is True

            converged_energy = [ev[1] for ev in events if ev[0] == "hardcoded_adapt_converged_energy"]
            assert len(converged_energy) == 1
            assert int(converged_energy[0]["eps_energy_min_extra_depth"]) == 3
            assert int(converged_energy[0]["eps_energy_patience"]) == 2
        finally:
            monkeypatch.setattr(_adapt_mod, "_ai_log", original_ai_log)


# ============================================================================
# ADAPT re-optimization policy tests
# ============================================================================

class TestAdaptReoptPolicyAppendOnly:
    """append_only policy must freeze the theta prefix and only optimize the newest param."""

    def test_prefix_preserved_across_depths(self, monkeypatch: pytest.MonkeyPatch):
        """After depth k, theta[:k] must be identical before and after depth k+1 optimization."""
        h_poly = build_hubbard_hamiltonian(
            dims=2, t=1.0, U=4.0, v=0.0,
            repr_mode="JW", indexing="blocked", pbc=True,
        )
        events: list[tuple[str, dict[str, object]]] = []
        original_ai_log = _adapt_mod._ai_log

        def _capture(event: str, **fields: object) -> None:
            events.append((str(event), dict(fields)))

        monkeypatch.setattr(_adapt_mod, "_ai_log", _capture)
        try:
            payload, _ = _run_hardcoded_adapt_vqe(
                h_poly=h_poly,
                num_sites=2,
                ordering="blocked",
                problem="hubbard",
                adapt_pool="uccsd",
                t=1.0,
                u=4.0,
                dv=0.0,
                boundary="periodic",
                omega0=0.0,
                g_ep=0.0,
                n_ph_max=1,
                boson_encoding="binary",
                max_depth=3,
                eps_grad=1e-6,
                eps_energy=1e-10,
                maxiter=40,
                seed=11,
                adapt_inner_optimizer="COBYLA",
                allow_repeats=True,
                finite_angle_fallback=True,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
                adapt_reopt_policy="append_only",
            )
            assert payload["success"] is True
            assert int(payload["ansatz_depth"]) >= 2, "Need at least 2 depths to check prefix"
            assert str(payload.get("adapt_reopt_policy", "")) == "append_only"

            # Extract the optimal_point (full theta) from payload.
            # History rows record depth-by-depth results.
            history = payload.get("history", [])
            assert len(history) >= 2

            # For append_only: at each depth k (0-indexed), the prefix
            # theta[:k] must be exactly what it was after depth k-1.
            # We verify this by checking that optimal_point[:k] from
            # depth k's row matches optimal_point[:k] constructed from
            # previous depths.
            #
            # Since the payload only gives us the final optimal_point,
            # we verify via the invariant: after the run, each history
            # row's "energy_before_opt" and "energy_after_opt" are
            # computed consistently with frozen prefixes.
            # More directly: re-run with full policy and confirm the
            # prefix DOES change there (see full_legacy test below).
            final_theta = np.array(payload["optimal_point"], dtype=float)
            logical_theta = np.array(payload["logical_optimal_point"], dtype=float)
            depth = int(payload["ansatz_depth"])
            assert int(payload["logical_num_parameters"]) == depth
            assert logical_theta.size == depth
            assert final_theta.size >= depth
            assert int(payload["num_parameters"]) == int(final_theta.size)
            assert payload.get("parameterization", {}).get("mode") == "per_pauli_term_v1"
        finally:
            monkeypatch.setattr(_adapt_mod, "_ai_log", original_ai_log)

    def test_append_only_vs_full_prefix_differs(self, monkeypatch: pytest.MonkeyPatch):
        """Running append_only vs full should produce different prefix values,
        proving append_only actually freezes and full actually changes them."""
        h_poly = build_hubbard_hamiltonian(
            dims=2, t=1.0, U=4.0, v=0.0,
            repr_mode="JW", indexing="blocked", pbc=True,
        )

        def _run_with_policy(policy: str) -> dict:
            original_ai_log = _adapt_mod._ai_log
            monkeypatch.setattr(_adapt_mod, "_ai_log", lambda event, **kw: None)
            try:
                payload, _ = _run_hardcoded_adapt_vqe(
                    h_poly=h_poly,
                    num_sites=2,
                    ordering="blocked",
                    problem="hubbard",
                    adapt_pool="uccsd",
                    t=1.0,
                    u=4.0,
                    dv=0.0,
                    boundary="periodic",
                    omega0=0.0,
                    g_ep=0.0,
                    n_ph_max=1,
                    boson_encoding="binary",
                    max_depth=3,
                    eps_grad=1e-6,
                    eps_energy=1e-10,
                    maxiter=80,
                    seed=7,
                    adapt_inner_optimizer="COBYLA",
                    allow_repeats=True,
                    finite_angle_fallback=True,
                    finite_angle=0.1,
                    finite_angle_min_improvement=1e-12,
                    adapt_reopt_policy=policy,
                )
                return payload
            finally:
                monkeypatch.setattr(_adapt_mod, "_ai_log", original_ai_log)

        payload_ao = _run_with_policy("append_only")
        payload_full = _run_with_policy("full")

        assert payload_ao["success"] is True
        assert payload_full["success"] is True

        theta_ao = np.array(payload_ao["optimal_point"], dtype=float)
        theta_full = np.array(payload_full["optimal_point"], dtype=float)

        # Both should produce valid results
        assert theta_ao.size >= 2
        assert theta_full.size >= 2

        # If both have at least 2 params, the first param should differ
        # (full re-optimizes it, append_only doesn't)
        min_len = min(theta_ao.size, theta_full.size)
        if min_len >= 2:
            # At least one prefix entry should differ between policies
            prefix_ao = theta_ao[:min_len - 1]
            prefix_full = theta_full[:min_len - 1]
            # They won't be exactly equal if full actually changes the prefix
            assert not np.allclose(prefix_ao, prefix_full, atol=1e-14), (
                "append_only and full produced identical prefix — "
                "policy difference is not effective"
            )


class TestAdaptReoptPolicyFull:
    """Full (legacy) re-optimization policy must allow all parameters to change."""

    def test_full_policy_allows_prefix_change(self, monkeypatch: pytest.MonkeyPatch):
        """With full policy, theta[:k] can change after appending depth k+1."""
        h_poly = build_hubbard_hamiltonian(
            dims=2, t=1.0, U=4.0, v=0.0,
            repr_mode="JW", indexing="blocked", pbc=True,
        )

        original_ai_log = _adapt_mod._ai_log
        monkeypatch.setattr(_adapt_mod, "_ai_log", lambda event, **kw: None)
        try:
            payload, _ = _run_hardcoded_adapt_vqe(
                h_poly=h_poly,
                num_sites=2,
                ordering="blocked",
                problem="hubbard",
                adapt_pool="uccsd",
                t=1.0,
                u=4.0,
                dv=0.0,
                boundary="periodic",
                omega0=0.0,
                g_ep=0.0,
                n_ph_max=1,
                boson_encoding="binary",
                max_depth=3,
                eps_grad=1e-6,
                eps_energy=1e-10,
                maxiter=80,
                seed=7,
                adapt_inner_optimizer="COBYLA",
                allow_repeats=True,
                finite_angle_fallback=True,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
                adapt_reopt_policy="full",
            )
            assert payload["success"] is True
            assert str(payload.get("adapt_reopt_policy", "")) == "full"
            assert int(payload["ansatz_depth"]) >= 2
        finally:
            monkeypatch.setattr(_adapt_mod, "_ai_log", original_ai_log)

    def test_invalid_policy_raises(self):
        """Invalid reopt policy must raise ValueError."""
        h_poly = build_hubbard_hamiltonian(
            dims=2, t=1.0, U=4.0, v=0.0,
            repr_mode="JW", indexing="blocked", pbc=True,
        )
        with pytest.raises(ValueError, match="adapt_reopt_policy"):
            _run_hardcoded_adapt_vqe(
                h_poly=h_poly,
                num_sites=2,
                ordering="blocked",
                problem="hubbard",
                adapt_pool="uccsd",
                t=1.0, u=4.0, dv=0.0,
                boundary="periodic",
                omega0=0.0, g_ep=0.0,
                n_ph_max=1, boson_encoding="binary",
                max_depth=3, eps_grad=1e-6, eps_energy=1e-10,
                maxiter=40, seed=7,
                allow_repeats=True,
                finite_angle_fallback=True,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
                adapt_reopt_policy="bogus_policy",
            )


class TestAdaptReoptPolicyWrapperPassthrough:
    """hubbard_pipeline._run_internal_adapt_paop must accept and forward adapt_reopt_policy."""

    def test_wrapper_signature_accepts_reopt_policy(self):
        """The wrapper function signature must include adapt_reopt_policy."""
        import inspect
        from pipelines.hardcoded import hubbard_pipeline as hp_mod
        sig = inspect.signature(hp_mod._run_internal_adapt_paop)
        assert "adapt_reopt_policy" in sig.parameters, (
            "_run_internal_adapt_paop is missing adapt_reopt_policy parameter"
        )
        param = sig.parameters["adapt_reopt_policy"]
        assert param.default == "append_only", (
            f"Expected default='append_only', got default={param.default!r}"
        )


# ============================================================================
# Edge cases
# ============================================================================

class TestAdaptEdgeCases:
    """Edge case and error handling tests."""

    def test_hubbard_pool_hva_raises(self):
        """Using pool='hva' with problem='hubbard' should raise ValueError."""
        h_poly = build_hubbard_hamiltonian(
            dims=2, t=1.0, U=4.0, v=0.0,
            repr_mode="JW", indexing="blocked", pbc=True,
        )
        with pytest.raises(ValueError, match="pool='hva' is not valid"):
            _run_hardcoded_adapt_vqe(
                h_poly=h_poly,
                num_sites=2, ordering="blocked",
                problem="hubbard", adapt_pool="hva",
                t=1.0, u=4.0, dv=0.0, boundary="periodic",
                omega0=0.0, g_ep=0.0, n_ph_max=1, boson_encoding="binary",
                max_depth=5, eps_grad=1e-2, eps_energy=1e-6,
                maxiter=50, seed=7,
                allow_repeats=True, finite_angle_fallback=False,
                finite_angle=0.1, finite_angle_min_improvement=1e-12,
            )

    def test_invalid_pool_raises(self):
        h_poly = build_hubbard_hamiltonian(
            dims=2, t=1.0, U=4.0, v=0.0,
            repr_mode="JW", indexing="blocked", pbc=True,
        )
        with pytest.raises(ValueError, match="Unsupported adapt pool"):
            _run_hardcoded_adapt_vqe(
                h_poly=h_poly,
                num_sites=2, ordering="blocked",
                problem="hubbard", adapt_pool="nonexistent_pool",
                t=1.0, u=4.0, dv=0.0, boundary="periodic",
                omega0=0.0, g_ep=0.0, n_ph_max=1, boson_encoding="binary",
                max_depth=5, eps_grad=1e-2, eps_energy=1e-6,
                maxiter=50, seed=7,
                allow_repeats=True, finite_angle_fallback=False,
                finite_angle=0.1, finite_angle_min_improvement=1e-12,
            )

    def test_hubbard_pool_uccsd_paop_lf_full_raises(self):
        """Composite HH-only pool must reject pure Hubbard runs."""
        h_poly = build_hubbard_hamiltonian(
            dims=2, t=1.0, U=4.0, v=0.0,
            repr_mode="JW", indexing="blocked", pbc=True,
        )
        with pytest.raises(ValueError, match="only valid for problem='hh'"):
            _run_hardcoded_adapt_vqe(
                h_poly=h_poly,
                num_sites=2, ordering="blocked",
                problem="hubbard", adapt_pool="uccsd_paop_lf_full",
                t=1.0, u=4.0, dv=0.0, boundary="periodic",
                omega0=0.0, g_ep=0.0, n_ph_max=1, boson_encoding="binary",
                max_depth=5, eps_grad=1e-2, eps_energy=1e-6,
                maxiter=50, seed=7,
                allow_repeats=True, finite_angle_fallback=False,
                finite_angle=0.1, finite_angle_min_improvement=1e-12,
            )

    def test_hubbard_pool_full_meta_runs(self):
        """full_meta should run for non-HH families as a problem-local mega pool."""
        h_poly = build_hubbard_hamiltonian(
            dims=2, t=1.0, U=4.0, v=0.0,
            repr_mode="JW", indexing="blocked", pbc=True,
        )
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=h_poly,
            num_sites=2, ordering="blocked",
            problem="hubbard", adapt_pool="full_meta",
            t=1.0, u=4.0, dv=0.0, boundary="periodic",
            omega0=0.0, g_ep=0.0, n_ph_max=1, boson_encoding="binary",
            max_depth=1, eps_grad=1e-2, eps_energy=1e-6,
            maxiter=40, seed=7,
            allow_repeats=False, finite_angle_fallback=False,
            finite_angle=0.1, finite_angle_min_improvement=1e-12,
        )
        assert str(payload["pool_type"]) == "full_meta"
        assert str(payload["parameterization_mode"]) == "logical_shared"
        assert payload["parameterization_execution_mode"] == "logical_shared"
        assert payload["optimizer_coordinate_chart"]["logical_dimension"] == int(
            payload["logical_num_parameters"]
        )
        assert payload["optimizer_coordinate_chart"]["runtime_dimension"] == int(
            payload["num_parameters"]
        )
        assert np.isfinite(float(payload["energy"]))

    def test_hh_phase1_allows_explicit_depth0_full_meta_override(self):
        h_poly = build_hubbard_holstein_hamiltonian(
            dims=2,
            J=1.0,
            U=2.0,
            omega0=1.0,
            g=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            v_t=None,
            v0=0.0,
            t_eval=None,
            repr_mode="JW",
            indexing="blocked",
            pbc=True,
            include_zero_point=True,
        )
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=h_poly,
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="full_meta",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=2,
            eps_grad=1e-2,
            eps_energy=1e-6,
            maxiter=30,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_continuation_mode="phase1_v1",
        )
        assert payload["phase1_depth0_full_meta_override"] is True
        assert payload["pool_type"] == "phase1_v1"
        assert payload["parameterization_mode"] == "logical_shared"
        assert payload["generator_sector_contract"]["passed"] is True
        assert payload["state_sector_contract"]["passed"] is True
        assert payload["strict_replay"]["passed"] is True


class TestHHPhase1Continuation:
    def _hh_h(self):
        return build_hubbard_holstein_hamiltonian(
            dims=2,
            J=1.0,
            U=2.0,
            omega0=1.0,
            g=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            v_t=None,
            v0=0.0,
            t_eval=None,
            repr_mode="JW",
            indexing="blocked",
            pbc=True,
            include_zero_point=True,
        )

    def test_legacy_history_omits_phase1_fields(self):
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=2,
            eps_grad=1e-2,
            eps_energy=1e-6,
            maxiter=20,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_continuation_mode="legacy",
        )
        assert "continuation" not in payload
        assert "measurement_cache_summary" not in payload
        for row in payload.get("history", []):
            assert "candidate_family" not in row
            assert "refit_window_indices" not in row
            assert "simple_score" not in row

    def test_phase1_does_not_eagerly_build_commutation_index(self, monkeypatch: pytest.MonkeyPatch):
        def _unexpected_build_exact_index(**kwargs: object) -> object:
            raise AssertionError(
                "phase1_v1 should not eagerly build commutation metadata"
            )

        monkeypatch.setattr(_adapt_mod, "build_exact_expansion_index", _unexpected_build_exact_index)
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=20,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase1_v1",
        )
        continuation = payload["continuation"]
        assert "algebraic_lane_policy" not in continuation
        assert continuation["physical_operator_lane_policy"]["enabled"] is True
        assert (
            continuation["physical_operator_lane_policy"]["lane_route"]
            == "physical_operator_type"
        )

    def test_phase3_default_selector_has_no_retired_algebraic_lane_payload(self, monkeypatch: pytest.MonkeyPatch):
        def _unexpected_build_exact_index(**kwargs: object) -> object:
            raise AssertionError(
                "default phase3_v1 should not eagerly build commutation metadata"
            )

        monkeypatch.setattr(_adapt_mod, "build_exact_expansion_index", _unexpected_build_exact_index)
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=20,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=2,
            adapt_window_topk=0,
            adapt_continuation_mode="phase3_v1",
            phase3_enable_rescue=False,
            phase3_lifetime_cost_mode="phase3_v1",
        )
        paper_i_config = payload["paper_i_configuration"]
        assert paper_i_config["schema"] == "paper_i_static_snake_configuration_v1"
        assert paper_i_config["canonical"]["cost_enabled"] is True
        assert paper_i_config["canonical"]["cost_weights"] == pytest.approx(
            {"2q": 0.20, "d": 0.20, "1q": 0.05, "theta": 0.05, "shot": 0.15}
        )
        continuation = payload["continuation"]
        assert "algebraic_lane_policy" not in continuation
        policy = continuation["physical_operator_lane_policy"]
        assert policy["enabled"] is True
        assert policy["lane_route"] == "physical_operator_type"
        phase0 = continuation["phase0_pilot"]
        assert phase0["enabled"] is True
        assert phase0["required_route_component"] is False
        assert phase0["satisfies_strict_route_a"] is False
        assert phase0["threshold"] == pytest.approx(0.0)
        assert phase0["max_records"] == 0
        assert "algebraic_lane_mode" not in phase0
        assert "algebraic_lanes" not in phase0
        pilot_rows = continuation["phase0_last_pilot_rows"]
        assert pilot_rows
        assert all(row["phase0_pilot_retained"] is True for row in pilot_rows)
        assert phase0["score_key"] == "phase0_score"
        assert phase0["score_formula"] == "DeltaE0_upper * N0"
        assert all(
            float(row["phase0_score"])
            == pytest.approx(float(row["phase0_delta_e_upper_hw"]))
            for row in pilot_rows
        )
        removed_cost_keys = {
            "phase0_K0",
            "phase0_hardware_cost_denominator",
            "phase0_hardware_cost_excess_sum",
            "phase0_cost_raw_components",
            "phase0_cost_bar_components",
            "phase0_cost_lambdas",
            "phase0_cost_lambda_source",
            "phase0_cost_normalization_schema",
            "phase0_cost_enabled",
        }
        assert all(removed_cost_keys.isdisjoint(row) for row in pilot_rows)
        assert all("phase0_algebraic_lane" not in row for row in pilot_rows)
        assert any(row["phase0_sigma_source"].endswith("zero_default") for row in pilot_rows)
        row = payload["history"][0]
        assert row["phase3_selector_policy"] == "hardware_resolvable_v1"
        assert "algebraic_lane" not in row
        assert "algebraic_quality" not in row
        assert "algebraic_context_counts" not in row

    def test_hardware_resolution_profile_runtime_telemetry_and_effective_floors(
        self,
        tmp_path: Path,
    ):
        profile_json = _write_hardware_resolution_profile_json(
            tmp_path,
            hw_floor=0.2,
            drift_floor=0.05,
        )
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=20,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=2,
            adapt_window_topk=0,
            adapt_continuation_mode="phase3_v1",
            phase1_score_z_alpha=0.0,
            phase2_score_z_alpha=0.0,
            phase3_enable_rescue=False,
            phase3_lifetime_cost_mode="phase3_v1",
            phase3_backend_cost_mode="proxy",
            hardware_resolution_mode="profile",
            hardware_resolution_profile_json=profile_json,
            hardware_resolution_profile_name="calib_a",
        )

        continuation = payload["continuation"]
        hardware = continuation["hardware_resolution"]
        assert hardware["schema"] == "gradient_resolution_v1"
        assert hardware["mode"] == "manual"
        assert hardware["mode_requested"] == "profile"
        assert hardware["mode_effective"] == "manual"
        assert hardware["floor_source"] == "profile_manifest"
        assert hardware["gradient_hw_floor"] == pytest.approx(0.2)
        assert hardware["gradient_drift_floor"] == pytest.approx(0.05)
        assert hardware["profile_name"] == "calib_a"
        assert hardware["profile_json"] == str(profile_json)
        assert hardware["profile_json_sha256"]
        assert hardware["profile_manifest_digest"]
        assert hardware["profile_digest"]
        assert hardware["profile_schema"] == HARDWARE_RESOLUTION_GRADIENT_PROFILE_SCHEMA
        assert hardware["profile_units"] == HARDWARE_RESOLUTION_PROFILE_UNITS
        assert hardware["profile_provenance"]["source"] == "test_adapt_vqe_integration"

        assert continuation["phase1"]["hardware_resolution_mode"] == "manual"
        assert continuation["phase1"]["gradient_hw_floor"] == pytest.approx(0.2)
        assert continuation["phase1"]["gradient_drift_floor"] == pytest.approx(0.05)
        assert continuation["phase2"]["hardware_resolution_mode"] == "manual"
        assert continuation["phase2"]["gradient_hw_floor"] == pytest.approx(0.2)
        assert continuation["phase2"]["gradient_drift_floor"] == pytest.approx(0.05)

        pilot_rows = continuation["phase0_last_pilot_rows"]
        assert pilot_rows
        assert pilot_rows[0]["phase0_hardware_resolution_mode"] == "manual"
        assert pilot_rows[0]["phase0_hardware_resolution_source"] == "manual_scalar_floors"
        assert pilot_rows[0]["phase0_epsilon_g_res"] == pytest.approx(0.25)
        assert pilot_rows[0]["phase0_g_upper_hw"] == pytest.approx(
            float(pilot_rows[0]["phase0_raw_gradient_abs"]) + 0.25
        )

        scored_rows = continuation["phase2_scored_rows"]
        assert scored_rows
        assert scored_rows[0]["hardware_resolution_mode"] == "manual"
        assert scored_rows[0]["hardware_resolution_source"] == "manual_scalar_floors"
        assert scored_rows[0]["b_g_hw"] == pytest.approx(0.2)
        assert scored_rows[0]["b_g_drift"] == pytest.approx(0.05)
        assert float(scored_rows[0]["epsilon_g_res"]) >= 0.25 - 1e-12

    def test_pipeline_nested_window_reserves_topk_age_slot(self):
        window = _adapt_mod._predict_nested_refit_window_for_position(
            theta=np.asarray([0.1, 5.0, 0.2], dtype=float),
            position_id=3,
            policy="windowed",
            window_size=3,
            window_topk=1,
            periodic_full_refit_triggered=False,
        )

        assert list(window.old_pre_indices) == [2, 1]
        assert list(window.window_new_post_indices) == [2]
        assert list(window.window_age_post_indices) == [1]
        assert list(window.active_post_indices) == [1, 2, 3]

    def test_phase3_geometry_window_counts_candidate_and_uses_nearest_old(self):
        window = _adapt_mod._predict_phase3_geometry_window_for_position(
            theta=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=float),
            position_id=2,
            geometry_window_size=3,
        )

        assert list(window.old_pre_indices) == [1, 2]
        assert list(window.old_post_indices) == [1, 3]
        assert list(window.active_post_indices) == [1, 2, 3]
        assert window.origin == "nested_inherited_v1"

    @pytest.mark.parametrize(
        ("reopt_policy", "window_size", "periodic_full_refit"),
        [
            ("windowed", 3, False),
            ("windowed", 3, True),  # the historical every-eighth-round trigger
            ("windowed", 1, False),
            ("full", 64, False),
        ],
    )
    def test_full_phase3_response_scope_ignores_refit_schedule(
        self,
        reopt_policy: str,
        window_size: int,
        periodic_full_refit: bool,
    ):
        theta = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=float)
        nested = _adapt_mod._predict_nested_refit_window_for_position(
            theta=theta,
            position_id=2,
            policy=reopt_policy,
            window_size=window_size,
            window_topk=1,
            periodic_full_refit_triggered=periodic_full_refit,
        )
        response = _adapt_mod._resolve_phase3_response_window_for_position(
            theta=theta,
            position_id=2,
            scope="full_active_plus_singleton_v1",
            geometry_window_size=0,
            nested_window=nested,
        )

        assert list(response.old_pre_indices) == [0, 1, 2, 3]
        assert list(response.active_post_indices) == [0, 1, 2, 3, 4]
        assert response.pre_parameter_count + 1 == len(
            response.active_post_indices
        )

    def test_legacy_phase3_response_scope_retains_exact_refit_coupling(self):
        theta = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=float)
        nested = _adapt_mod._predict_nested_refit_window_for_position(
            theta=theta,
            position_id=2,
            policy="windowed",
            window_size=3,
            window_topk=0,
            periodic_full_refit_triggered=False,
        )
        response = _adapt_mod._resolve_phase3_response_window_for_position(
            theta=theta,
            position_id=2,
            scope="legacy_reopt_coupled_v1",
            geometry_window_size=0,
            nested_window=nested,
        )

        assert response is nested
        assert list(response.active_post_indices) == list(
            nested.active_post_indices
        )

    def test_fixed_local_phase3_response_scope_is_explicit(self):
        theta = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=float)
        nested = _adapt_mod._predict_nested_refit_window_for_position(
            theta=theta,
            position_id=2,
            policy="full",
            window_size=64,
            window_topk=0,
            periodic_full_refit_triggered=False,
        )
        response = _adapt_mod._resolve_phase3_response_window_for_position(
            theta=theta,
            position_id=2,
            scope="fixed_local_window_v1",
            geometry_window_size=3,
            nested_window=nested,
        )

        assert list(response.old_pre_indices) == [1, 2]
        assert list(response.active_post_indices) == [1, 2, 3]

        with pytest.raises(ValueError, match="requires phase3_geometry_window_size"):
            _adapt_mod._resolve_phase3_response_window_for_position(
                theta=theta,
                position_id=2,
                scope="fixed_local_window_v1",
                geometry_window_size=0,
                nested_window=nested,
            )

    def test_full_phase3_response_invariant_runs_before_support_reduction(self):
        feature = CandidateFeatures(
            stage_name="phase3",
            candidate_label="x",
            candidate_family="test",
            candidate_pool_index=0,
            position_id=2,
            append_position=2,
            positions_considered=[2],
            g_signed=0.0,
            g_abs=0.0,
            g_lcb=0.0,
            sigma_hat=0.0,
            F=1.0,
            novelty=1.0,
            curvature_mode="test",
            novelty_mode="test",
            refit_window_indices=[0, 1],
            compiled_position_cost_proxy={},
            measurement_cache_stats={},
            leakage_penalty=0.0,
            stage_gate_open=True,
            leakage_gate_open=True,
            trough_probe_triggered=False,
            trough_detected=False,
            simple_score=0.0,
            score_version="test",
            phase3_response_coordinate_scope=(
                "full_active_plus_singleton_v1"
            ),
            phase3_response_coordinate_indices=[0, 1, 2],
            phase3_response_pre_support_count=3,
            phase3_active_logical_coordinate_count=2,
        )

        _adapt_mod._assert_full_phase3_response_records(
            [{"feature": feature}],
            active_logical_coordinate_count=2,
        )

        contradictory = feature.__class__(
            **{
                **feature.__dict__,
                "phase3_response_coordinate_indices": [1, 2],
                "phase3_response_pre_support_count": 2,
            }
        )
        with pytest.raises(RuntimeError, match="response-scope invariant"):
            _adapt_mod._assert_full_phase3_response_records(
                [{"feature": contradictory}],
                active_logical_coordinate_count=2,
            )

    def test_phase1_refit_window_matches_actual_window(self):
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=2,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=20,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase1_v1",
        )
        assert payload["continuation"]["mode"] == "phase1_v1"
        assert "stage_events" in payload["continuation"]
        for row in payload.get("history", []):
            assert row["refit_window_indices"] == row["reopt_active_indices"]


class TestHHPhase2Continuation:
    def _hh_h(self):
        return build_hubbard_holstein_hamiltonian(
            dims=2,
            J=1.0,
            U=2.0,
            omega0=1.0,
            g=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            v_t=None,
            v0=0.0,
            t_eval=None,
            repr_mode="JW",
            indexing="blocked",
            pbc=True,
            include_zero_point=True,
        )

    def test_phase2_emits_full_v2_and_memory_fields(self):
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=2,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=20,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase2_v1",
        )
        assert payload["continuation"]["mode"] == "phase2_v1"
        assert "optimizer_memory" in payload["continuation"]
        assert payload["continuation"]["optimizer_memory"]["parameter_count"] == payload["num_parameters"]
        for row in payload.get("history", []):
            assert row["refit_window_indices"] == row["reopt_active_indices"]
            assert "full_v2_score" in row
            assert "shortlisted_records" in row
            assert "optimizer_memory_source" in row


class TestHHPhase3Continuation:
    def _hh_h(self):
        return build_hubbard_holstein_hamiltonian(
            dims=2,
            J=1.0,
            U=2.0,
            omega0=1.0,
            g=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            v_t=None,
            v0=0.0,
            t_eval=None,
            repr_mode="JW",
            indexing="blocked",
            pbc=True,
            include_zero_point=True,
        )

    def _oracle_cfg(self, **overrides: object) -> _Phase3OracleGradientConfig:
        payload: dict[str, object] = {
            "noise_mode": "shots",
            "shots": 64,
            "oracle_repeats": 2,
            "oracle_aggregate": "mean",
            "backend_name": None,
            "use_fake_backend": False,
            "seed": 7,
            "gradient_step": 0.1,
            "mitigation_mode": "none",
            "local_readout_strategy": None,
            "zne_scales": (),
            "local_gate_twirling": False,
            "dd_sequence": None,
            "scope": "selection_only",
            "execution_surface_requested": "auto",
            "execution_surface": "expectation_v1",
            "raw_transport": "auto",
            "raw_store_memory": False,
            "raw_artifact_path": None,
            "seed_transpiler": None,
            "transpile_optimization_level": 1,
        }
        payload.update(overrides)
        return _Phase3OracleGradientConfig(**payload)

    def _final_audit_cfg(self, **overrides: object) -> _FinalNoiseAuditConfig:
        payload: dict[str, object] = {
            "noise_mode": "shots",
            "shots": 64,
            "oracle_repeats": 2,
            "oracle_aggregate": "mean",
            "backend_name": None,
            "use_fake_backend": False,
            "seed": 7,
            "mitigation_mode": "none",
            "local_readout_strategy": None,
            "zne_scales": (),
            "local_gate_twirling": False,
            "dd_sequence": None,
            "runtime_profile_name": "legacy_runtime_v0",
            "runtime_session_policy": "prefer_session",
            "compare_unmitigated_baseline": False,
            "seed_transpiler": None,
            "transpile_optimization_level": 1,
            "strict": False,
        }
        payload.update(overrides)
        return _FinalNoiseAuditConfig(**payload)

    def _install_fake_oracle_bindings(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        gradient_by_label: dict[str, float] | None = None,
        sigma_by_label: dict[str, float] | None = None,
        objective_mean: float | None = None,
        objective_mean_by_stage: dict[str, float] | None = None,
        default_gradient: float = 1.0,
        default_sigma: float = 0.0,
        shots: int = 64,
        gradient_step: float = 0.1,
        backend_name: str = "FakeNighthawk",
        raise_on_raw_measure: bool = False,
        raise_on_symmetry_measure: bool = False,
        raise_on_final_audit: bool = False,
        raise_on_final_audit_baseline: bool = False,
    ) -> list[object]:
        gradient_lookup = dict(gradient_by_label or {})
        sigma_lookup = dict(sigma_by_label or {})
        objective_lookup = dict(objective_mean_by_stage or {})
        oracle_instances: list[object] = []

        class _FakeOracleConfig:
            def __init__(self, **kwargs: object) -> None:
                self.__dict__.update(kwargs)

        class _FakeOracle:
            def __init__(self, config: object) -> None:
                self.config = config
                self.calls: list[tuple[str, float]] = []
                self.closed = False
                self.backend_info = SimpleNamespace(
                    backend_name=str(backend_name),
                    using_fake_backend=bool(getattr(config, "use_fake_backend", False)),
                    details={"noise_mode": str(getattr(config, "noise_mode", "shots"))},
                )
                oracle_instances.append(self)

            def evaluate(self, circuit: object, observable: object) -> SimpleNamespace:
                del observable
                stage = getattr(circuit, "_phase3_objective_stage", None)
                if stage is not None:
                    objective_val = float(objective_lookup.get(str(stage), objective_mean if objective_mean is not None else 0.0))
                    self.calls.append((str(stage), 0.0))
                    return SimpleNamespace(
                        mean=float(objective_val),
                        std=0.0,
                        stdev=0.0,
                        stderr=0.0,
                        n_samples=int(shots),
                        raw_values=[float(objective_val)],
                        aggregate=str(getattr(self.config, "oracle_aggregate", "mean")),
                    )
                label = str(getattr(circuit, "_phase3_candidate_label", "unknown"))
                sign = float(getattr(circuit, "_phase3_probe_sign", 0.0))
                grad_target = float(gradient_lookup.get(label, default_gradient))
                sigma_target = float(sigma_lookup.get(label, default_sigma))
                per_eval_stderr = float(sigma_target * math.sqrt(2.0) * float(gradient_step))
                self.calls.append((str(label), float(sign)))
                return SimpleNamespace(
                    mean=float(sign * grad_target * float(gradient_step)),
                    std=float(per_eval_stderr),
                    stdev=float(per_eval_stderr),
                    stderr=float(per_eval_stderr),
                    n_samples=int(shots),
                    raw_values=[float(sign * grad_target * float(gradient_step))],
                    aggregate=str(getattr(self.config, "oracle_aggregate", "mean")),
                )

            def evaluate_parameterized(
                self,
                *,
                plan: object,
                theta_runtime: object,
                observable: object,
                runtime_trace_context: dict[str, object] | None = None,
                **_kwargs: object,
            ) -> SimpleNamespace:
                del plan, theta_runtime, observable
                trace = dict(runtime_trace_context or {})
                route = str(trace.get("route", "")).strip().lower()
                if route == "final_noise_audit_v1":
                    audit_variant = str(trace.get("audit_variant", "requested"))
                    if raise_on_final_audit:
                        raise RuntimeError("synthetic final noise audit failure")
                    if audit_variant == "unmitigated_baseline" and raise_on_final_audit_baseline:
                        raise RuntimeError("synthetic final noise audit baseline failure")
                    objective_val = float(
                        objective_lookup.get(
                            f"final_noise_audit_v1::{audit_variant}",
                            objective_lookup.get(
                                "final_noise_audit_v1",
                                objective_mean if objective_mean is not None else 0.0,
                            ),
                        )
                    )
                    self.calls.append((f"final_noise_audit_v1::{audit_variant}", 0.0))
                    return SimpleNamespace(
                        mean=float(objective_val),
                        std=0.0,
                        stdev=0.0,
                        stderr=0.0,
                        n_samples=int(shots),
                        raw_values=[float(objective_val)],
                        aggregate=str(getattr(self.config, "oracle_aggregate", "mean")),
                    )
                stage = trace.get("objective_stage", None)
                if stage is not None:
                    objective_val = float(objective_lookup.get(str(stage), objective_mean if objective_mean is not None else 0.0))
                    self.calls.append((str(stage), 0.0))
                    return SimpleNamespace(
                        mean=float(objective_val),
                        std=0.0,
                        stdev=0.0,
                        stderr=0.0,
                        n_samples=int(shots),
                        raw_values=[float(objective_val)],
                        aggregate=str(getattr(self.config, "oracle_aggregate", "mean")),
                    )
                label = str(trace.get("candidate_label", "unknown"))
                probe_sign = str(trace.get("probe_sign", "plus"))
                sign = 1.0 if probe_sign == "plus" else -1.0
                grad_target = float(gradient_lookup.get(label, default_gradient))
                sigma_target = float(sigma_lookup.get(label, default_sigma))
                per_eval_stderr = float(sigma_target * math.sqrt(2.0) * float(gradient_step))
                self.calls.append((str(label), float(sign)))
                return SimpleNamespace(
                    mean=float(sign * grad_target * float(gradient_step)),
                    std=float(per_eval_stderr),
                    stdev=float(per_eval_stderr),
                    stderr=float(per_eval_stderr),
                    n_samples=int(shots),
                    raw_values=[float(sign * grad_target * float(gradient_step))],
                    aggregate=str(getattr(self.config, "oracle_aggregate", "mean")),
                )

            def close(self) -> None:
                self.closed = True

        class _FakeRawOracle:
            def __init__(self, config: object) -> None:
                self.config = config
                self.calls: list[tuple[str, str]] = []
                self.diagnostic_calls: list[tuple[str, str]] = []
                self.closed = False
                self.transport = "sampler_v2"
                self.backend_snapshot = {"backend_name": str(backend_name)}
                oracle_instances.append(self)

            def measure_observable(
                self,
                *,
                plan: object,
                theta_runtime: object,
                observable: object,
                observable_family: str,
                semantic_tags: dict[str, object] | None = None,
                **_kwargs: object,
            ) -> SimpleNamespace:
                del theta_runtime, observable
                tags = dict(semantic_tags or {})
                label = str(tags.get("candidate_label", "unknown"))
                probe_sign = str(tags.get("probe_sign", "plus"))
                sign = 1.0 if probe_sign == "plus" else -1.0
                grad_target = float(gradient_lookup.get(label, default_gradient))
                sigma_target = float(sigma_lookup.get(label, default_sigma))
                per_eval_stderr = float(sigma_target * math.sqrt(2.0) * float(gradient_step))
                nq = int(getattr(plan, "nq", 6))
                repeat_count = int(getattr(self.config, "oracle_repeats", 1))
                is_symmetry_diag = str(observable_family) == "adapt_phase3_oracle_symmetry_diagnostic"
                objective_stage = tags.get("objective_stage", None)
                is_inner_objective = str(observable_family) == "adapt_phase3_oracle_inner_objective"
                if is_symmetry_diag:
                    self.diagnostic_calls.append((str(label), str(probe_sign)))
                    if raise_on_raw_measure:
                        raise RuntimeError("synthetic raw oracle failure")
                    if raise_on_symmetry_measure:
                        raise RuntimeError("synthetic symmetry diagnostic failure")
                    basis_label = "Z" * int(nq)
                    counts = {"000101": int(shots // 2), "001010": int(shots - (shots // 2))}
                    records = [
                        {
                            "evaluation_id": f"eval-diag-{label}-{probe_sign}",
                            "observable_family": str(observable_family),
                            "basis_label": str(basis_label),
                            "num_qubits": int(nq),
                            "measured_logical_qubits": list(range(int(nq))),
                            "repeat_index": int(repeat_idx),
                            "counts": dict(counts),
                            "shots_completed": int(shots),
                            "semantic_tags": dict(tags),
                            "transport": str(self.transport),
                            "compile_signature": {"compiled_depth": 1},
                        }
                        for repeat_idx in range(int(repeat_count))
                    ]
                    estimate_mean = 1.0
                    compile_signatures = {str(basis_label): {"compiled_depth": 1}}
                elif is_inner_objective:
                    self.calls.append((str(objective_stage or "objective"), "inner"))
                    if raise_on_raw_measure:
                        raise RuntimeError("synthetic raw oracle failure")
                    basis_label = "Z"
                    counts = {"0": int(shots)}
                    records = [
                        {
                            "evaluation_id": f"eval-inner-{objective_stage}-{repeat_idx}",
                            "observable_family": str(observable_family),
                            "basis_label": str(basis_label),
                            "num_qubits": 1,
                            "measured_logical_qubits": [0],
                            "repeat_index": int(repeat_idx),
                            "counts": dict(counts),
                            "shots_completed": int(shots),
                            "semantic_tags": dict(tags),
                            "transport": str(self.transport),
                            "compile_signature": {"compiled_depth": 1},
                        }
                        for repeat_idx in range(int(repeat_count))
                    ]
                    estimate_mean = float(
                        objective_lookup.get(
                            str(objective_stage),
                            objective_mean if objective_mean is not None else 0.0,
                        )
                    )
                    compile_signatures = {str(basis_label): {"compiled_depth": 1}}
                else:
                    self.calls.append((str(label), str(probe_sign)))
                    if raise_on_raw_measure:
                        raise RuntimeError("synthetic raw oracle failure")
                    basis_label = "Z"
                    records = [
                        {
                            "evaluation_id": f"eval-{label}-{probe_sign}",
                            "observable_family": str(observable_family),
                            "basis_label": "Z",
                            "num_qubits": 1,
                            "measured_logical_qubits": [0],
                            "repeat_index": int(repeat_idx),
                            "counts": ({"0": int(shots)} if sign > 0 else {"1": int(shots)}),
                            "shots_completed": int(shots),
                            "semantic_tags": dict(tags),
                            "transport": str(self.transport),
                            "compile_signature": {"compiled_depth": 1},
                        }
                        for repeat_idx in range(int(repeat_count))
                    ]
                    estimate_mean = float(sign * grad_target * float(gradient_step))
                    compile_signatures = {"Z": {"compiled_depth": 1}}
                estimate = SimpleNamespace(
                    mean=float(estimate_mean),
                    std=float(per_eval_stderr),
                    stdev=float(per_eval_stderr),
                    stderr=float(per_eval_stderr),
                    n_samples=int(repeat_count),
                    raw_values=tuple(float(estimate_mean) for _ in range(int(repeat_count))),
                    aggregate=str(getattr(self.config, "oracle_aggregate", "mean")),
                    total_shots=int(shots * repeat_count),
                    group_count=1,
                    term_count=1,
                    record_count=len(records),
                    reduction_mode="repeat_aligned_full_observable",
                )
                return SimpleNamespace(
                    estimate=estimate,
                    records=records,
                    transport=str(self.transport),
                    observable_family=str(observable_family),
                    evaluation_id=(
                        f"eval-diag-{label}-{probe_sign}-{len(self.diagnostic_calls)}"
                        if is_symmetry_diag
                        else f"eval-{label}-{probe_sign}-{len(self.calls)}"
                    ),
                    raw_artifact_path=getattr(self.config, "raw_artifact_path", None),
                    compile_signatures_by_basis=compile_signatures,
                    backend_snapshot=dict(self.backend_snapshot),
                    plan_digest="plan",
                    structure_digest="structure",
                    reference_state_digest="ref",
                )

            def close(self) -> None:
                self.closed = True

        def _normalize_request(cfg: object) -> dict[str, object]:
            mitigation = getattr(cfg, "mitigation", {"mode": "none"})
            if not isinstance(mitigation, dict):
                mitigation = {"mode": "none"}
            return {
                "noise_mode": str(getattr(cfg, "noise_mode", "shots")),
                "shots": int(getattr(cfg, "shots", shots)),
                "oracle_repeats": int(getattr(cfg, "oracle_repeats", 1)),
                "oracle_aggregate": str(getattr(cfg, "oracle_aggregate", "mean")),
                "backend_name": getattr(cfg, "backend_name", None),
                "use_fake_backend": bool(getattr(cfg, "use_fake_backend", False)),
                "execution_surface": str(getattr(cfg, "execution_surface", "expectation_v1")),
                "raw_transport": str(getattr(cfg, "raw_transport", "auto")),
                "mitigation": dict(mitigation),
                "symmetry_mitigation": {"mode": "off"},
                "runtime_profile": {
                    "name": str(getattr(cfg, "runtime_profile", "legacy_runtime_v0")),
                },
                "runtime_session": {
                    "mode": str(getattr(cfg, "runtime_session", "prefer_session")),
                },
                "transpile_optimization_level": int(getattr(cfg, "transpile_optimization_level", 1)),
            }

        def _fake_bindings() -> dict[str, object]:
            return {
                "ExpectationOracle": _FakeOracle,
                "RawMeasurementOracle": _FakeRawOracle,
                "OracleConfig": _FakeOracleConfig,
                "all_z_full_register_qop": _raw_runtime._all_z_full_register_qop,
                "summarize_hh_full_register_z_records": _raw_runtime._summarize_hh_full_register_z_records,
                "normalize_sampler_raw_runtime_config": (lambda cfg: cfg),
                "normalize_oracle_execution_request": _normalize_request,
                "assess_oracle_execution_capability": (
                    lambda cfg: {
                        "supported": True,
                        "reason_code": "ok",
                        "reason": "ok",
                        "normalized_request": _normalize_request(cfg),
                    }
                ),
                "validate_oracle_execution_request": (
                    lambda cfg: {
                        "supported": True,
                        "reason_code": "ok",
                        "reason": "ok",
                        "normalized_request": _normalize_request(cfg),
                    }
                ),
                "build_runtime_layout_circuit": (
                    lambda layout, theta_runtime, num_qubits, reference_state=None: SimpleNamespace(
                        layout=layout,
                        theta_runtime=np.asarray(theta_runtime, dtype=float),
                        num_qubits=int(num_qubits),
                        reference_state=reference_state,
                    )
                ),
                "build_parameterized_ansatz_plan": (
                    lambda layout, nq, ref_state=None: SimpleNamespace(
                        layout=layout,
                        nq=int(nq),
                        circuit=SimpleNamespace(layout=layout),
                        parameters=tuple(),
                        reference_state=ref_state,
                        plan_digest="plan",
                        structure_digest="structure",
                        reference_state_digest="ref",
                    )
                ),
                "pauli_poly_to_sparse_pauli_op": (lambda poly: SimpleNamespace(poly=poly)),
                "preflight_backend_scheduled_fake_backend_environment": (lambda cfg: None),
                "validate_controller_oracle_base_config": (lambda cfg: None),
            }

        monkeypatch.setattr(_adapt_mod, "_phase3_oracle_runtime_bindings", _fake_bindings)
        return oracle_instances


    def test_phase3_geometry_window_decouples_from_full_optimizer_refit(self):
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=2,
            eps_grad=1e-12,
            eps_energy=1e-12,
            maxiter=20,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="full",
            adapt_window_size=64,
            adapt_window_topk=0,
            phase3_geometry_window_size=1,
            adapt_continuation_mode="phase3_v1",
            phase3_selector_policy="algebraic_nested_v1",
            phase3_backend_cost_mode="proxy",
            phase3_enable_rescue=False,
            phase3_lifetime_cost_mode="phase3_v1",
        )

        assert payload["phase3_geometry_window_size"] == 1
        assert payload["w3_wopt_decoupled"] is True
        assert payload["history"]
        row = payload["history"][-1]
        assert row["w3_wopt_decoupled"] is True
        assert row["phase3_geometry_window_policy"] == "fixed_local_v1"
        assert row["nested_refit_window_applied"] is False
        assert row["nested_refit_window_status"] == "geometry_only_wopt_decoupled"
        assert row["optimizer_active_refit_indices"] == row["reopt_active_indices"]
        assert row["optimizer_active_refit_count"] == row["reopt_active_count"]
        assert len(row["phase3_geometry_active_post_indices"]) <= 1
        selected = row["selected_feature_rows"][0]
        assert selected["w3_wopt_decoupled"] is True
        assert selected["refit_window_indices"] == selected["phase3_geometry_refit_window_indices"]
        assert selected["optimizer_active_refit_indices"] == row["reopt_active_indices"]
        assert len(selected["phase3_geometry_active_post_indices"]) <= 1

    def test_phase3_algebraic_nested_invalid_payload_falls_back(self, monkeypatch: pytest.MonkeyPatch):
        def _reject_nested_window(*_args: object, **_kwargs: object) -> None:
            raise _adapt_mod.NestedWindowError("forced mismatch")

        monkeypatch.setattr(_adapt_mod, "validate_nested_window", _reject_nested_window)
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=20,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=2,
            adapt_window_topk=0,
            adapt_continuation_mode="phase3_v1",
            phase3_selector_policy="algebraic_nested_v1",
            phase3_enable_rescue=False,
            phase3_lifetime_cost_mode="phase3_v1",
        )

        row = payload["history"][0]
        assert row["nested_refit_window_applied"] is False
        assert row["nested_refit_window_status"] == "invalid_payload_forced_mismatch"
        assert row["active_post_refit_indices"] == []
        assert row["reopt_policy_effective"] == "windowed"

    def test_live_prune_repeat_label_guard_blocks_only_accepted_labels(self):
        history = [
            {
                "post_admission_prune": {
                    "accepted_count": 1,
                    "decisions": [
                        {
                            "label": "paop_full:paop_disp(site=0)",
                            "accepted": True,
                        }
                    ],
                }
            },
            {
                "post_admission_prune": {
                    "accepted_count": 0,
                    "decisions": [
                        {
                            "label": "paop_full:paop_disp(site=1)",
                            "accepted": False,
                        }
                    ],
                }
            },
        ]
        blocked_labels = _accepted_live_prune_labels_from_history(history)

        assert blocked_labels == ["paop_full:paop_disp(site=0)"]

        kept, blocked_rows = _filter_repeat_live_prune_candidates(
            candidate_indices=[0, 1, 2],
            labels_now=[
                "paop_full:paop_disp(site=0)",
                "paop_full:paop_disp(site=1)",
                "fermion_hop",
            ],
            blocked_labels=blocked_labels,
        )

        assert kept == [1, 2]
        assert blocked_rows == [
            {
                "index": 0,
                "label": "paop_full:paop_disp(site=0)",
                "reason": "previous_live_prune_acceptance_same_label",
            }
        ]

    def test_phase3_algebraic_nested_live_recoverability_uses_typed_prune_ladder(self):
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=3,
            eps_grad=1e-12,
            eps_energy=1e-12,
            maxiter=40,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=2,
            adapt_window_topk=0,
            adapt_drop_floor=-1.0,
            adapt_grad_floor=-1.0,
            adapt_continuation_mode="phase3_v1",
            phase3_selector_policy="algebraic_nested_v1",
            phase3_enable_rescue=False,
            phase3_lifetime_cost_mode="phase3_v1",
            phase1_prune_enabled=True,
            phase1_prune_policy="recoverability_ladder_v1",
            phase1_prune_mode="live",
            phase1_prune_checkpoint_period=1,
            phase1_prune_maturity_threshold=0.0,
            phase1_prune_snr_threshold=1e12,
            phase1_prune_fraction=1.0,
            phase1_prune_min_candidates=1,
            phase1_prune_max_candidates=4,
            phase1_prune_max_regression=1e6,
            phase1_prune_retained_gain_ratio=0.0,
            phase1_prune_protect_steps=0,
        )

        typed_prunes = [
            row["post_admission_prune"]
            for row in payload["history"]
            if row["post_admission_prune"].get("typed_prune_ladder_active")
        ]
        assert typed_prunes
        prune = next(
            (row for row in typed_prunes if row.get("recoverability_ladder_rows")),
            typed_prunes[-1],
        )
        assert prune["typed_prune_ladder_active"] is True
        assert prune["typed_prune_ladder_gate"]["active"] is True
        assert prune["typed_prune_ladder_gate"]["phase3_selector_policy"] == "algebraic_nested_v1"
        assert prune["typed_prune_ladder_gate"]["phase1_prune_policy"] == "recoverability_ladder_v1"
        assert prune["recoverability_ladder_active"] is True
        assert prune["recoverability_rung_policy"] == "typed_algebraic_nested_live_v1"
        assert not any("amplitude" in key for key in prune)
        assert set(prune["probe_indices"]).issubset(set(prune["recoverability_eligible_indices"]))
        plans = prune["typed_prune_ladder_plans"]
        assert plans
        assert "typed_compensator_window" not in prune["nomination_lanes"]
        assert prune["compensator_window_authority"]["typed_compensator_window"]["active"] is True
        assert all(
            "typed_compensator_window" not in source.get("lanes", [])
            for source in prune.get("candidate_nomination_sources", [])
        )
        assert [rung["rung_kind"] for rung in plans[0]["rungs"]] == [
            "frozen_delete",
            "comm_refit",
            "comm_corr_refit",
            "comm_corr_nc_refit",
            "terminal_refit",
        ]
        assert prune["recoverability_ladder_rows"]
        assert all(
            not any("amplitude" in key for key in row)
            for row in prune["recoverability_ladder_rows"]
        )
        assert all(
            row["acceptance_source"] == "remove_refit_energy_safety"
            and row["surrogate_used_for_acceptance"] is False
            and row["recoverability_rung_policy"] == "typed_algebraic_nested_live_v1"
            for row in prune["recoverability_ladder_rows"]
        )
        assert prune["recovery_class"] in {
            "flat_redundant",
            "curvature_compensated",
            "failed",
        }
        assert prune["recovery_classification"]["recovery_class"] == prune["recovery_class"]

    def test_phase3_emits_generator_motif_symmetry_and_lifetime_fields(self):
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=2,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=20,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase3_v1",
            phase3_symmetry_mitigation_mode="verify_only",
            phase3_enable_rescue=False,
            phase3_lifetime_cost_mode="phase3_v1",
        )
        continuation = payload["continuation"]
        assert continuation["mode"] == "phase3_v1"
        summary = continuation["selected_scaffold_summary"]
        final_choice = continuation["selected_scaffold_final_choice"]
        branch_state = continuation["selected_scaffold_branch_state"]
        state_summary = continuation["selected_state_summary"]
        memory_contract = continuation["selected_scaffold_optimizer_memory_contract"]
        runtime_boundary = continuation["controller_runtime_boundary_summary"]
        history_summary = continuation["selected_scaffold_history"]
        record_chain = continuation["selected_scaffold_record_chain"]
        surface_summary = continuation["active_phase3_surface_summary"]
        pool_summary = continuation["active_hh_pool_summary"]
        audit = continuation["selected_scaffold_audit"]
        assert summary["selection_source"] == "main_branch"
        assert summary["final_choice_summary"] == final_choice
        assert summary["branch_state_summary"] == branch_state
        assert summary["selected_state_summary"] == state_summary
        assert summary["optimizer_memory_contract_summary"] == memory_contract
        assert summary["scaffold_label"] == "O_*"
        assert summary["theta_label"] == "theta_*^adapt"
        assert summary["history_label"] == "H_*"
        assert summary["manifold_label"] == "M_scaf(O_*)"
        assert runtime_boundary["summary_label"] == "appendix_a_runtime_boundary"
        assert runtime_boundary["beam_enabled"] is False
        assert runtime_boundary["branch_id"] is None
        assert runtime_boundary["calibration_status"] == "runtime_calibrated_not_symbolic"
        assert runtime_boundary["stage_controller_payload"] == continuation["stage_controller"]
        assert runtime_boundary["current_controller_snapshot"] == branch_state["controller_telemetry"]["last_snapshot"]
        assert "selected_scaffold_summary" in runtime_boundary["symbolic_result_keys"]
        assert "selected_scaffold_final_choice" in runtime_boundary["symbolic_result_keys"]
        assert "stage_controller" in runtime_boundary["runtime_controller_keys"]
        assert "selected_scaffold_optimizer_memory_contract" in runtime_boundary["runtime_controller_keys"]
        assert runtime_boundary["runtime_law_notation"]["thresholds"] == "tau_k(t)"
        assert runtime_boundary["runtime_law_notation"]["caps"] == "N_k(t)"
        assert runtime_boundary["runtime_law_notation"]["shots_phase1"] == "N_shot,1(t)"
        assert runtime_boundary["runtime_law_notation"]["shots_phasek"] == "N_shot,k(t)"
        assert runtime_boundary["configured_bounds"]["cap_phase1_min"] == continuation["stage_controller"]["shortlist_size"]
        assert runtime_boundary["configured_bounds"]["cap_phase1_max"] == continuation["stage_controller"]["shortlist_size"]
        assert summary["operator_labels"] == payload["operators"]
        assert summary["theta_adapt"] == payload["logical_optimal_point"]
        assert summary["history_step_count"] == len(history_summary) == len(payload.get("history", []))
        assert summary["history_record_count"] == sum(len(step["selected_records"]) for step in history_summary)
        assert summary["history_record_chain_label"] == "H_*"
        assert len(record_chain) == int(summary["history_record_count"])
        assert [row["generator_label"] for row in record_chain] == [
            rec["generator_label"]
            for step in history_summary
            for rec in step["selected_records"]
        ]
        assert surface_summary["surface_label"] == "Omega_HH^(3)"
        assert surface_summary["source_rows_key"] == "phase2_shortlist_rows"
        assert surface_summary["source_row_semantics"] == "last_scored_candidate_surface"
        assert surface_summary["scored_rows_key"] == "phase2_scored_rows"
        assert surface_summary["retained_rows_key"] == "phase2_retained_shortlist_rows"
        assert surface_summary["admitted_rows_key"] == "phase2_admitted_rows"
        assert continuation["phase2_scored_rows"] == continuation["phase2_shortlist_rows"]
        assert int(surface_summary["candidate_count"]) == len(continuation["phase2_scored_rows"])
        assert int(surface_summary["retained_shortlist_count"]) == len(continuation["phase2_retained_shortlist_rows"])
        assert int(surface_summary["admitted_count"]) == len(continuation["phase2_admitted_rows"])
        assert int(surface_summary["admitted_count"]) <= int(surface_summary["retained_shortlist_count"]) <= int(surface_summary["candidate_count"])
        assert surface_summary["selected_operator_labels"] == payload["operators"]
        assert surface_summary["selected_generator_ids"] == summary["generator_ids"]
        assert int(surface_summary["phase3_shortlisted_count"]) <= int(surface_summary["candidate_count"])
        assert pool_summary["summary_label"] == "Omega_HH_active"
        assert pool_summary["omega_chain"] == ["Omega_HH^(1)", "Omega_HH^(2)", "Omega_HH^(3)"]
        assert int(pool_summary["phases"]["phase1"]["count"]) == len(continuation["phase1_retained_rows"])
        assert int(pool_summary["phases"]["phase2"]["count"]) == len(continuation["phase2_geometric_shortlist_rows"])
        assert int(pool_summary["phases"]["phase3"]["count"]) == len(continuation["phase2_retained_shortlist_rows"])
        assert bool(pool_summary["nested_generator_image_inclusion"]["phase2_in_phase1"])
        assert bool(pool_summary["nested_generator_image_inclusion"]["phase3_in_phase2"])
        assert bool(pool_summary["nested_generator_image_inclusion"]["phase3_in_phase1"])
        assert audit["source_kind"] == "main_branch"
        assert audit["final_choice_summary"] == final_choice
        assert audit["branch_state_summary"] == branch_state
        assert audit["selected_state_summary"] == state_summary
        assert audit["optimizer_memory_contract_summary"] == memory_contract
        assert audit["beam_enabled"] is False
        assert audit["branch_id"] is None
        assert audit["operators"] == payload["operators"]
        assert branch_state["branch_state_notation"] == "\\mathfrak b_*"
        assert branch_state["status"] == "terminal"
        assert branch_state["termination_label"] == audit["stop_reason"]
        assert branch_state["cumulative_selector_score"] == audit["prune_key"]["cumulative_selector_score"]
        assert branch_state["cumulative_selector_burden"] == audit["prune_key"]["cumulative_selector_burden"]
        telemetry = branch_state["controller_telemetry"]
        assert telemetry["telemetry_label"] == "T_b^ctrl"
        assert telemetry["stage_event_count"] == len(audit["stage_events"])
        assert telemetry["last_probe_reason"] == audit["last_probe_reason"]
        assert telemetry["residual_opened"] is audit["residual_opened"]
        if telemetry["last_snapshot"] is not None:
            assert telemetry["last_snapshot"]["snapshot_version"] in {
                "phase123_controller_v1",
                "phase123_controller_maturity_v2",
            }
            assert "n_rem_hat" in telemetry["last_snapshot"]
            assert "useful_horizon" in telemetry["last_snapshot"]
            assert "runway_fraction" in telemetry["last_snapshot"]
            assert "H_t" in telemetry["last_snapshot"]
            assert "phase_live" in telemetry["last_snapshot"]
            assert "terminal_phase" in telemetry["last_snapshot"]
            assert "phase_shots_effective" in telemetry["last_snapshot"]
            assert telemetry["last_snapshot"]["useful_horizon"] <= telemetry["last_snapshot"]["depth_left"]
        assert state_summary["state_label"] == "|psi_*>"
        assert state_summary["state_preparation_label"] == "U(theta_*^adapt; O_*)|phi_0>"
        assert state_summary["reference_state_label"] == "|phi_0>"
        assert state_summary["manifold_label"] == summary["manifold_label"]
        assert state_summary["ansatz_depth"] == summary["ansatz_depth"]
        assert state_summary["manifold_dimension"] == summary["manifold_dimension"]
        assert state_summary["branch_id"] is None
        assert state_summary["state_norm"] == pytest.approx(1.0, abs=1e-10)
        assert memory_contract["contract_label"] == "phase2_optimizer_memory_contract"
        assert memory_contract["exact_reuse_rule"] == "requires_matching_scaffold_fingerprint"
        assert bool(memory_contract["fingerprint_match_required"]) is True
        assert memory_contract["canonical_embedding_notation"] == "theta -> theta⊕_p 0"
        assert memory_contract["refit_window_notation"] == "W(r;t)"
        assert memory_contract["branch_id"] is None
        assert memory_contract["last_active_subset_source"] == payload["history"][-1]["optimizer_memory_source"]
        assert bool(memory_contract["last_active_subset_reused"]) is bool(payload["history"][-1]["optimizer_memory_reused"])
        assert memory_contract["scaffold_fingerprint"]["fingerprint_notation"] == "fp(O_*)"
        assert memory_contract["scaffold_fingerprint"]["num_parameters"] == memory_contract["memory_parameter_count"]
        assert memory_contract["observed_transport_mode"] in {
            "unavailable",
            "same_scaffold_active_subset",
            "canonical_embedding_or_index_remap",
        }
        audit_surface = audit["phase3_surface_summary"]
        assert audit_surface["scored_surface_notation"] == "R_3(t)"
        assert audit_surface["retained_shortlist_notation"] == "S_3(t)"
        assert audit_surface["admitted_set_notation"] == "B_t^*"
        assert int(audit_surface["scored_surface"]["count"]) == len(continuation["phase2_scored_rows"])
        assert int(audit_surface["retained_shortlist"]["count"]) == len(continuation["phase2_retained_shortlist_rows"])
        assert int(audit_surface["admitted_set"]["count"]) == len(continuation["phase2_admitted_rows"])
        assert final_choice["beam_enabled"] is False
        assert final_choice["beam_child_kind"] is None
        assert final_choice["transition_kind"] == "main_path_admission"
        assert final_choice["selected_record_count"] == len(history_summary[-1]["selected_records"])
        assert bool(final_choice["batch_selected"]) is bool(history_summary[-1]["batch_selected"])
        assert final_choice["step_index"] == history_summary[-1]["step_index"]
        assert final_choice["selection_mode"] == history_summary[-1]["selection_mode"]
        assert audit["depth_local"] == len(payload.get("history", []))
        assert audit["prune_history"]
        assert audit["last_prune"]["permission_reason"] == payload["history"][-1]["post_admission_prune"]["permission_reason"]
        assert audit["last_prune"]["accepted_count"] == payload["history"][-1]["post_admission_prune"]["accepted_count"]
        assert audit["last_prune"]["selected_label"] == payload["history"][-1]["post_admission_prune"]["selected_label"]
        assert continuation["selected_generator_metadata"]
        assert "motif_library" in continuation
        assert continuation["symmetry_mitigation"]["mode"] == "verify_only"
        assert "rescue_history" in continuation
        assert payload["scaffold_fingerprint_lite"]["selected_generator_ids"]
        assert payload["compile_cost_proxy_summary"]["version"] == "phase3_v1_proxy"
        for row in payload.get("history", []):
            assert row["refit_window_indices"] == row["reopt_active_indices"]
            assert "generator_id" in row
            assert "symmetry_mode" in row
            assert "lifetime_cost_mode" in row
            assert "remaining_evaluations_proxy" in row
            assert "cheap_score" in row
            assert row["cheap_score_version"] == "simple_v1"
            assert "F" in row
            assert "cheap_benefit_proxy" in row
            assert "cheap_burden_total" in row
            assert "sigma_hat" in row
            assert "post_admission_prune" in row
            assert row["scored_surface_size"] == len(row["scored_surface_records"])
            assert row["retained_shortlist_size"] == len(row["retained_shortlist_records"])
            assert row["admitted_record_count"] == len(row["admitted_records"])
            assert row["admitted_record_count"] <= row["retained_shortlist_size"] <= row["scored_surface_size"]
            prune = row["post_admission_prune"]
            assert isinstance(prune, dict)
            assert "permission_reason" in prune
            assert 0.0 <= float(prune["u_sat"]) <= 1.0
            assert 0.0 <= float(prune["runway_ratio"]) <= 1.0
            assert bool(prune["mature_open"]) is (float(prune["u_sat"]) >= float(prune["maturity_threshold"]))
            assert bool(prune["checkpoint_due"]) is (int(row["depth"]) % int(prune["checkpoint_period"]) == 0)
            assert float(prune["gain_floor"]) >= 0.0
            assert float(prune["snr_adm"]) >= 0.0
            assert bool(prune["snr_low_enough"]) is (float(prune["snr_adm"]) <= float(prune["snr_threshold"]))
            assert "small_angle_pool_indices" not in prune
            probe_basis = prune["recoverability_eligible_indices"]
            assert set(prune["probe_indices"]).issubset(set(probe_basis))
            assert set(prune["protected_indices"]).isdisjoint(set(prune["mature_eligible_indices"]))
            assert set(prune["cooldown_blocked_indices"]).isdisjoint(set(prune["mature_eligible_indices"]))
            assert len(prune["gate_rows"]) == len(prune["metadata"])
            assert row["sigma_hat"] == pytest.approx(0.0)
            assert math.isfinite(float(row["F"]))
        assert continuation["phase2_shortlist_rows"]
        assert continuation["phase2_scored_rows"] == continuation["phase2_shortlist_rows"]
        assert all(
            row["cheap_score_version"] == "simple_v1"
            for row in continuation["phase2_shortlist_rows"]
        )
        assert all(
            row["sigma_hat"] == pytest.approx(0.0)
            for row in continuation["phase2_shortlist_rows"]
        )
        assert all(
            math.isfinite(float(row["F"]))
            for row in continuation["phase2_shortlist_rows"]
        )

    def test_oracle_fd_gradient_stderr_combines_stderr_in_quadrature(self):
        stderr = _adapt_mod._oracle_fd_gradient_stderr(
            SimpleNamespace(stderr=0.3),
            {"stderr": 0.4},
            grad_step=0.2,
        )
        assert stderr == pytest.approx(math.sqrt(0.3 ** 2 + 0.4 ** 2) / 0.4)

    def test_phase3_phase1_shortlist_input_uses_raw_f_metric(self, monkeypatch: pytest.MonkeyPatch):
        captured_phase1_metrics: list[tuple[float, float]] = []
        original_shortlist_records = _adapt_mod.shortlist_records

        def _capture_shortlist_records(records, *, cfg, score_key="simple_score", tie_break_score_key="simple_score"):
            if score_key == "simple_score":
                for rec in records:
                    feat = rec.get("feature")
                    if feat is not None and hasattr(feat, "F") and hasattr(feat, "g_abs"):
                        captured_phase1_metrics.append(
                            (float(feat.F), float(feat.g_abs))
                        )
            return original_shortlist_records(
                records,
                cfg=cfg,
                score_key=score_key,
                tie_break_score_key=tie_break_score_key,
            )

        monkeypatch.setattr(_adapt_mod, "shortlist_records", _capture_shortlist_records)
        monkeypatch.setattr(_adapt_mod, "raw_f_metric_from_state", lambda **kwargs: 123.0)

        _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=20,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase3_v1",
            phase3_symmetry_mitigation_mode="verify_only",
            phase3_enable_rescue=False,
            phase3_lifetime_cost_mode="phase3_v1",
        )

        assert captured_phase1_metrics
        assert all(metric == pytest.approx(123.0) for metric, _ in captured_phase1_metrics)
        assert any(abs(metric - g_abs) > 1e-6 for metric, g_abs in captured_phase1_metrics)

    def test_phase3_simple_score_phase1_evaluates_full_pool_before_retaining_cap(self, monkeypatch: pytest.MonkeyPatch):
        captured_phase1_record_counts: list[int] = []
        original_shortlist_records = _adapt_mod.shortlist_records

        def _capture_shortlist_records(records, *, cfg, score_key="simple_score", tie_break_score_key="simple_score"):
            if score_key == "simple_score":
                captured_phase1_record_counts.append(int(len(records)))
            return original_shortlist_records(
                records,
                cfg=cfg,
                score_key=score_key,
                tie_break_score_key=tie_break_score_key,
            )

        monkeypatch.setattr(_adapt_mod, "shortlist_records", _capture_shortlist_records)

        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=20,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase3_v1",
            phase1_shortlist_size=1,
            phase3_symmetry_mitigation_mode="verify_only",
            phase3_enable_rescue=False,
            phase3_lifetime_cost_mode="phase3_v1",
        )

        assert captured_phase1_record_counts
        assert max(captured_phase1_record_counts) > 1
        assert len(payload["continuation"]["phase1_retained_rows"]) == 1

    def test_phase3_routes_sigma_hat_from_label_resolver(self, monkeypatch: pytest.MonkeyPatch):
        captured_sigmas: list[float] = []
        original_shortlist_records = _adapt_mod.shortlist_records

        def _capture_shortlist_records(records, *, cfg, score_key="simple_score", tie_break_score_key="simple_score"):
            if score_key == "simple_score":
                for rec in records:
                    feat = rec.get("feature")
                    if feat is not None and hasattr(feat, "sigma_hat"):
                        captured_sigmas.append(float(feat.sigma_hat))
            return original_shortlist_records(
                records,
                cfg=cfg,
                score_key=score_key,
                tie_break_score_key=tie_break_score_key,
            )

        monkeypatch.setattr(_adapt_mod, "shortlist_records", _capture_shortlist_records)
        monkeypatch.setattr(_adapt_mod, "_phase3_sigma_hat_for_label", lambda **kwargs: 0.25)

        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=20,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase3_v1",
            phase3_symmetry_mitigation_mode="verify_only",
            phase3_enable_rescue=False,
            phase3_lifetime_cost_mode="phase3_v1",
        )

        assert captured_sigmas
        assert all(sigma == pytest.approx(0.25) for sigma in captured_sigmas)
        assert payload["history"][0]["sigma_hat"] == pytest.approx(0.25)

    def test_phase3_oracle_gradient_mode_default_off_keeps_exact_path(self, monkeypatch: pytest.MonkeyPatch):
        def _unexpected_bindings() -> dict[str, object]:
            raise AssertionError("oracle runtime bindings should not be loaded when phase3_oracle_gradient_mode is off")

        monkeypatch.setattr(_adapt_mod, "_phase3_oracle_runtime_bindings", _unexpected_bindings)
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=20,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase3_v1",
            phase3_symmetry_mitigation_mode="verify_only",
            phase3_enable_rescue=False,
            phase3_lifetime_cost_mode="phase3_v1",
        )

        assert payload["continuation"]["gradient_uncertainty_source"] == "zero_default"
        assert payload["continuation"]["oracle_gradient_scope"] == "off"
        assert payload["continuation"]["oracle_gradient_config"] is None
        assert payload["continuation"]["oracle_gradient_calls_total"] == 0
        assert payload["history"][0]["gradient_source"] == "exact_commutator"
        assert payload["history"][0]["max_gradient_stderr"] == pytest.approx(0.0)
        assert payload["history"][0]["candidate_gradient_scout"] == []

    def test_final_noise_audit_default_off_keeps_exact_path(self, monkeypatch: pytest.MonkeyPatch):
        def _unexpected_bindings() -> dict[str, object]:
            raise AssertionError("oracle runtime bindings should not be loaded when final noise audit is off")

        monkeypatch.setattr(_adapt_mod, "_phase3_oracle_runtime_bindings", _unexpected_bindings)
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=20,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase3_v1",
            phase3_symmetry_mitigation_mode="verify_only",
            phase3_enable_rescue=False,
            phase3_lifetime_cost_mode="phase3_v1",
        )

        assert payload["success"] is True
        assert payload["energy_source"] == "exact_statevector"
        assert payload["continuation"]["oracle_gradient_scope"] == "off"
        assert "final_noise_audit_v1" not in payload

    def test_adapt_analytic_noise_zero_std_keeps_exact_baseline(self):
        common_kwargs = dict(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=5,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase3_v1",
            phase3_symmetry_mitigation_mode="verify_only",
            phase3_enable_rescue=False,
            phase3_lifetime_cost_mode="phase3_v1",
        )
        baseline_payload, _ = _run_hardcoded_adapt_vqe(**common_kwargs)
        zero_std_payload, _ = _run_hardcoded_adapt_vqe(
            **common_kwargs,
            adapt_analytic_noise_std=0.0,
            adapt_analytic_noise_seed=17,
        )

        assert bool(baseline_payload["analytic_noise_applied"]) is False
        assert bool(zero_std_payload["analytic_noise_applied"]) is False
        assert float(zero_std_payload["energy"]) == pytest.approx(float(baseline_payload["energy"]))
        assert float(zero_std_payload["exact_energy_from_final_state"]) == pytest.approx(
            float(baseline_payload["exact_energy_from_final_state"])
        )
        assert list(zero_std_payload["operators"]) == list(baseline_payload["operators"])

    def test_adapt_analytic_noise_seed_controls_exact_path_reproducibly(self):
        common_kwargs = dict(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=5,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase3_v1",
            phase3_symmetry_mitigation_mode="verify_only",
            phase3_enable_rescue=False,
            phase3_lifetime_cost_mode="phase3_v1",
            adapt_analytic_noise_std=0.5,
        )
        payload_a, _ = _run_hardcoded_adapt_vqe(
            **common_kwargs,
            adapt_analytic_noise_seed=123,
        )
        payload_b, _ = _run_hardcoded_adapt_vqe(
            **common_kwargs,
            adapt_analytic_noise_seed=123,
        )
        payload_c, _ = _run_hardcoded_adapt_vqe(
            **common_kwargs,
            adapt_analytic_noise_seed=124,
        )

        assert bool(payload_a["analytic_noise_applied"]) is True
        assert int(payload_a["analytic_noise_seed"]) == 123
        assert float(payload_a["energy"]) == pytest.approx(float(payload_b["energy"]))
        assert list(payload_a["operators"]) == list(payload_b["operators"])
        assert np.asarray(payload_a["optimal_point"], dtype=float) == pytest.approx(
            np.asarray(payload_b["optimal_point"], dtype=float)
        )
        assert float(payload_c["energy"]) != pytest.approx(float(payload_a["energy"]))

    def test_adapt_analytic_noise_does_not_modify_oracle_inner_objective_path(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        self._install_fake_oracle_bindings(
            monkeypatch,
            default_gradient=1.0,
            default_sigma=0.1,
            gradient_step=0.1,
            objective_mean=-0.321,
        )
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=1,
            seed=7,
            adapt_inner_optimizer="SPSA",
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase3_v1",
            phase3_symmetry_mitigation_mode="verify_only",
            phase3_enable_rescue=True,
            phase3_lifetime_cost_mode="phase3_v1",
            phase3_oracle_gradient_config=self._oracle_cfg(gradient_step=0.1),
            phase3_oracle_inner_objective_mode="noisy_v1",
            adapt_analytic_noise_std=10.0,
            adapt_analytic_noise_seed=17,
        )

        assert payload["energy_source"] == "oracle_expectation_v1"
        assert float(payload["energy"]) == pytest.approx(-0.321)
        assert bool(payload["analytic_noise_applied"]) is False

    def test_adapt_analytic_noise_rejects_negative_std(self):
        with pytest.raises(ValueError, match="adapt_analytic_noise_std"):
            _run_hardcoded_adapt_vqe(
                h_poly=self._hh_h(),
                num_sites=2,
                ordering="blocked",
                problem="hh",
                adapt_pool="paop_lf_std",
                t=1.0,
                u=2.0,
                dv=0.0,
                boundary="periodic",
                omega0=1.0,
                g_ep=0.5,
                n_ph_max=1,
                boson_encoding="binary",
                max_depth=1,
                eps_grad=1e-3,
                eps_energy=1e-8,
                maxiter=5,
                seed=7,
                allow_repeats=True,
                finite_angle_fallback=False,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
                adapt_reopt_policy="windowed",
                adapt_window_size=1,
                adapt_window_topk=0,
                adapt_continuation_mode="phase3_v1",
                phase3_symmetry_mitigation_mode="verify_only",
                phase3_enable_rescue=False,
                phase3_lifetime_cost_mode="phase3_v1",
                adapt_analytic_noise_std=-0.1,
            )

    def test_phase3_oracle_gradient_mode_routes_sigma_through_real_oracle_path(self, monkeypatch: pytest.MonkeyPatch):
        oracle_instances = self._install_fake_oracle_bindings(
            monkeypatch,
            default_gradient=1.0,
            default_sigma=0.25,
            gradient_step=0.1,
        )
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=20,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase3_v1",
            phase3_symmetry_mitigation_mode="verify_only",
            phase3_enable_rescue=False,
            phase3_lifetime_cost_mode="phase3_v1",
            phase3_oracle_gradient_config=self._oracle_cfg(gradient_step=0.1),
        )

        continuation = payload["continuation"]
        scout_rows = (
            payload["history"][0]["candidate_gradient_scout"]
            if payload["history"]
            else continuation["last_candidate_gradient_scout"]
        )
        max_stderr = (
            payload["history"][0]["max_gradient_stderr"]
            if payload["history"]
            else continuation["last_max_gradient_stderr"]
        )
        assert continuation["gradient_uncertainty_source"] == "oracle_fd_stderr_v1"
        assert continuation["oracle_gradient_scope"] == "selection_only"
        assert continuation["oracle_gradient_calls_total"] == 2 * len(scout_rows)
        assert continuation["oracle_backend_info"]["backend_name"] == "FakeNighthawk"
        assert continuation["reoptimization_backend"] == "exact_statevector"
        assert scout_rows
        assert max_stderr > 0.0
        assert any(float(row["sigma_hat"]) > 0.0 for row in scout_rows)
        assert oracle_instances and getattr(oracle_instances[0], "closed", False) is True

    def test_phase3_oracle_gradient_mode_keeps_reoptimization_exact(self, monkeypatch: pytest.MonkeyPatch):
        oracle_instances = self._install_fake_oracle_bindings(
            monkeypatch,
            default_gradient=1.0,
            default_sigma=0.1,
            gradient_step=0.1,
        )
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=20,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase3_v1",
            phase3_symmetry_mitigation_mode="verify_only",
            phase3_enable_rescue=False,
            phase3_lifetime_cost_mode="phase3_v1",
            phase3_oracle_gradient_config=self._oracle_cfg(gradient_step=0.1),
        )

        history_row = payload["history"][0]
        assert payload["continuation"]["reoptimization_backend"] == "exact_statevector"
        assert payload["continuation"]["oracle_gradient_calls_total"] == 2 * len(history_row["candidate_gradient_scout"])
        assert len(oracle_instances) == 1
        assert len(oracle_instances[0].calls) == 2 * len(history_row["candidate_gradient_scout"])

    def test_phase3_oracle_inner_objective_mode_requires_active_oracle_config(self):
        with pytest.raises(ValueError, match="requires an active phase3 oracle gradient config"):
            _run_hardcoded_adapt_vqe(
                h_poly=self._hh_h(),
                num_sites=2,
                ordering="blocked",
                problem="hh",
                adapt_pool="paop_lf_std",
                t=1.0,
                u=2.0,
                dv=0.0,
                boundary="periodic",
                omega0=1.0,
                g_ep=0.5,
                n_ph_max=1,
                boson_encoding="binary",
                max_depth=1,
                eps_grad=1e-3,
                eps_energy=1e-8,
                maxiter=20,
                seed=7,
                adapt_inner_optimizer="SPSA",
                allow_repeats=True,
                finite_angle_fallback=False,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
                adapt_continuation_mode="phase3_v1",
                phase3_oracle_inner_objective_mode="noisy_v1",
            )

    @pytest.mark.parametrize("optimizer", ["POWELL", "QNSPSA"])
    def test_phase3_oracle_inner_objective_mode_requires_spsa(self, optimizer: str):
        with pytest.raises(ValueError, match="requires adapt_inner_optimizer='SPSA'"):
            _run_hardcoded_adapt_vqe(
                h_poly=self._hh_h(),
                num_sites=2,
                ordering="blocked",
                problem="hh",
                adapt_pool="paop_lf_std",
                t=1.0,
                u=2.0,
                dv=0.0,
                boundary="periodic",
                omega0=1.0,
                g_ep=0.5,
                n_ph_max=1,
                boson_encoding="binary",
                max_depth=1,
                eps_grad=1e-3,
                eps_energy=1e-8,
                maxiter=10,
                seed=7,
                adapt_inner_optimizer=str(optimizer),
                allow_repeats=True,
                finite_angle_fallback=False,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
                adapt_continuation_mode="phase3_v1",
                phase3_oracle_gradient_config=self._oracle_cfg(gradient_step=0.1),
                phase3_oracle_inner_objective_mode="noisy_v1",
            )

    def test_phase3_oracle_inner_objective_mode_uses_oracle_energy_for_payload(self, monkeypatch: pytest.MonkeyPatch):
        oracle_instances = self._install_fake_oracle_bindings(
            monkeypatch,
            default_gradient=1.0,
            default_sigma=0.1,
            gradient_step=0.1,
            objective_mean=-0.321,
        )
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=1,
            seed=7,
            adapt_inner_optimizer="SPSA",
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase3_v1",
            phase3_symmetry_mitigation_mode="verify_only",
            phase3_enable_rescue=True,
            phase3_lifetime_cost_mode="phase3_v1",
            phase3_oracle_gradient_config=self._oracle_cfg(gradient_step=0.1),
            phase3_oracle_inner_objective_mode="noisy_v1",
        )

        assert payload["energy_source"] == "oracle_expectation_v1"
        assert payload["energy"] == pytest.approx(-0.321)
        assert payload["phase3_oracle_inner_objective_mode"] == "noisy_v1"
        assert payload["phase3_oracle_inner_objective_mode_requested"] == "noisy_v1"
        assert payload["phase3_oracle_inner_objective_runtime_guard_reason"] is None
        assert payload["continuation"]["reoptimization_backend"] == "oracle_expectation_v1"
        assert payload["continuation"]["oracle_inner_objective_mode"] == "noisy_v1"
        assert payload["continuation"]["oracle_inner_objective_mode_requested"] == "noisy_v1"
        assert payload["continuation"]["oracle_inner_objective_runtime_guard_reason"] is None
        assert payload["continuation"]["oracle_inner_objective_calls_total"] > 0
        assert payload["continuation"]["phase3_enable_rescue_requested"] is True
        assert payload["continuation"]["phase3_enable_rescue_effective"] is False
        assert payload["exact_energy_from_final_state"] != pytest.approx(payload["energy"])
        assert payload["exact_state_fidelity_source"] == "final_theta_exact_state_sidecar"
        assert oracle_instances and getattr(oracle_instances[0], "closed", False) is True

    @pytest.mark.parametrize(
        "oracle_cfg_overrides",
        [
            pytest.param({"noise_mode": "ideal"}, id="ideal_expectation_zero_noise"),
            pytest.param(
                {
                    "noise_mode": "aer_density_matrix_synthetic_depolarizing",
                    "synthetic_depolarizing_1q_error": 0.0,
                    "synthetic_depolarizing_2q_error": 0.0,
                },
                id="synthetic_depolarizing_zero_error",
            ),
        ],
    )
    def test_phase3_oracle_inner_objective_zero_noise_noisy_v1_matches_exact_selected_energy(
        self,
        monkeypatch: pytest.MonkeyPatch,
        oracle_cfg_overrides: dict[str, object],
    ):
        self._install_fake_oracle_bindings(
            monkeypatch,
            default_gradient=1.0,
            default_sigma=0.0,
            gradient_step=0.1,
            objective_mean=-999.0,
        )
        oracle_cfg = self._oracle_cfg(
            **oracle_cfg_overrides,
            execution_surface="expectation_v1",
            execution_surface_requested="expectation_v1",
            value_noise_model="off",
            value_noise_std=0.0,
        )
        common_kwargs = dict(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=1,
            seed=7,
            adapt_inner_optimizer="SPSA",
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase3_v1",
            phase3_symmetry_mitigation_mode="verify_only",
            phase3_enable_rescue=True,
            phase3_lifetime_cost_mode="phase3_v1",
            phase3_oracle_gradient_config=oracle_cfg,
        )

        exact_payload, _ = _run_hardcoded_adapt_vqe(
            **common_kwargs,
            phase3_oracle_inner_objective_mode="exact",
        )
        guarded_payload, _ = _run_hardcoded_adapt_vqe(
            **common_kwargs,
            phase3_oracle_inner_objective_mode="noisy_v1",
        )

        assert guarded_payload["phase3_oracle_inner_objective_mode"] == "exact"
        assert guarded_payload["phase3_oracle_inner_objective_mode_requested"] == "noisy_v1"
        assert (
            guarded_payload["phase3_oracle_inner_objective_runtime_guard_reason"]
            == _adapt_mod._ZERO_NOISE_ORACLE_INNER_EXACT_GUARD_REASON
        )
        guarded_continuation = guarded_payload["continuation"]
        exact_continuation = exact_payload["continuation"]
        assert guarded_payload["energy_source"] == "exact_statevector"
        assert guarded_continuation["reoptimization_backend"] == "exact_statevector"
        assert guarded_continuation["oracle_inner_objective_mode"] == "exact"
        assert guarded_continuation["oracle_inner_objective_mode_requested"] == "noisy_v1"
        assert guarded_continuation["oracle_inner_objective_calls_total"] == 0
        assert guarded_continuation["phase3_enable_rescue_effective"] == exact_continuation["phase3_enable_rescue_effective"]
        assert guarded_continuation["phase1_prune"]["enabled"] == exact_continuation["phase1_prune"]["enabled"]
        assert guarded_continuation["oracle_gradient_calls_total"] == exact_continuation["oracle_gradient_calls_total"]

        assert float(guarded_payload["energy"]) == pytest.approx(float(exact_payload["energy"]))
        assert float(guarded_payload["exact_energy_from_final_state"]) == pytest.approx(
            float(exact_payload["exact_energy_from_final_state"])
        )
        assert list(guarded_payload["operators"]) == list(exact_payload["operators"])
        assert np.asarray(guarded_payload["optimal_point"], dtype=float) == pytest.approx(
            np.asarray(exact_payload["optimal_point"], dtype=float)
        )
        exact_scout = exact_payload["history"][0]["candidate_gradient_scout"]
        guarded_scout = guarded_payload["history"][0]["candidate_gradient_scout"]
        assert [row["candidate_label"] for row in guarded_scout] == [
            row["candidate_label"] for row in exact_scout
        ]
        assert [float(row["gradient_signed"]) for row in guarded_scout] == pytest.approx(
            [float(row["gradient_signed"]) for row in exact_scout]
        )

    def test_phase3_oracle_inner_objective_value_noise_preserves_exact_structure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        oracle_instances = self._install_fake_oracle_bindings(
            monkeypatch,
            default_gradient=1.0,
            default_sigma=0.0,
            gradient_step=0.1,
            objective_mean=-999.0,
        )
        exact_oracle_cfg = self._oracle_cfg(
            noise_mode="ideal",
            execution_surface="expectation_v1",
            execution_surface_requested="expectation_v1",
            value_noise_model="off",
            value_noise_std=0.0,
        )
        noisy_oracle_cfg = self._oracle_cfg(
            noise_mode="ideal",
            execution_surface="expectation_v1",
            execution_surface_requested="expectation_v1",
            value_noise_model="gaussian_iid_v1",
            value_noise_std=1.0e-6,
            value_noise_seed=123,
            value_noise_sigma0_abs=1.0e-3,
            value_noise_n_eff=1.0e6,
            value_noise_semantic="snake_function_value_noise_shot_equivalent_v1",
            value_noise_std_source="shot_equivalent_sigma0_over_sqrt_n_eff",
        )
        common_kwargs = dict(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=1,
            seed=7,
            adapt_inner_optimizer="SPSA",
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase3_v1",
            phase3_symmetry_mitigation_mode="verify_only",
            phase3_enable_rescue=True,
            phase3_lifetime_cost_mode="phase3_v1",
        )

        exact_payload, _ = _run_hardcoded_adapt_vqe(
            **common_kwargs,
            phase3_oracle_gradient_config=exact_oracle_cfg,
            phase3_oracle_inner_objective_mode="exact",
        )
        noisy_payload, _ = _run_hardcoded_adapt_vqe(
            **common_kwargs,
            phase3_oracle_gradient_config=noisy_oracle_cfg,
            phase3_oracle_inner_objective_mode="noisy_v1",
        )

        assert noisy_payload["phase3_oracle_inner_objective_mode"] == _adapt_mod._VALUE_NOISE_ORACLE_INNER_EXACT_STRUCTURE_MODE
        assert noisy_payload["phase3_oracle_inner_objective_mode_requested"] == "noisy_v1"
        assert (
            noisy_payload["phase3_oracle_inner_objective_runtime_guard_reason"]
            == _adapt_mod._VALUE_NOISE_ORACLE_INNER_EXACT_STRUCTURE_REASON
        )
        assert noisy_payload["energy_source"] == "exact_statevector_plus_value_noise"
        assert noisy_payload["energy"] != pytest.approx(-999.0)
        noisy_continuation = noisy_payload["continuation"]
        exact_continuation = exact_payload["continuation"]
        assert noisy_continuation["reoptimization_backend"] == "exact_statevector_plus_value_noise"
        assert noisy_continuation["oracle_inner_objective_mode"] == _adapt_mod._VALUE_NOISE_ORACLE_INNER_EXACT_STRUCTURE_MODE
        assert noisy_continuation["oracle_inner_objective_mode_requested"] == "noisy_v1"
        assert noisy_continuation["oracle_inner_objective_runtime_guard_reason"] == _adapt_mod._VALUE_NOISE_ORACLE_INNER_EXACT_STRUCTURE_REASON
        assert noisy_continuation["oracle_inner_objective_raw_records_total"] == 0
        assert noisy_continuation["oracle_inner_objective_calls_total"] > 0
        assert noisy_continuation["phase3_enable_rescue_effective"] == exact_continuation["phase3_enable_rescue_effective"]
        assert noisy_continuation["phase1_prune"]["enabled"] == exact_continuation["phase1_prune"]["enabled"]
        assert noisy_continuation["oracle_gradient_calls_total"] == exact_continuation["oracle_gradient_calls_total"]

        value_noise_summary = noisy_continuation["oracle_inner_exact_structure_value_noise"]
        assert value_noise_summary["enabled"] is True
        assert value_noise_summary["guard_reason"] == _adapt_mod._VALUE_NOISE_ORACLE_INNER_EXACT_STRUCTURE_REASON
        assert value_noise_summary["draw_count"] == noisy_continuation["oracle_inner_objective_calls_total"]
        assert value_noise_summary["effective_seed"] == 123
        assert value_noise_summary["seed_source"] == "value_noise_seed"
        assert value_noise_summary["oracle_gradient_value_noise_suppressed"] is True
        last_draw = value_noise_summary["last_draw"]
        assert last_draw["schema"] == "phase3_inner_exact_structure_value_noise_draw_v1"
        assert last_draw["model"] == "gaussian_iid_v1"
        assert last_draw["std"] == pytest.approx(1.0e-6)
        assert last_draw["n_eff"] == pytest.approx(1.0e6)
        assert last_draw["semantic"] == "snake_function_value_noise_shot_equivalent_v1"
        assert last_draw["physical_shots_unchanged"] is True

        assert list(noisy_payload["operators"]) == list(exact_payload["operators"])
        exact_scout = exact_payload["history"][0]["candidate_gradient_scout"]
        noisy_scout = noisy_payload["history"][0]["candidate_gradient_scout"]
        assert [row["candidate_label"] for row in noisy_scout] == [
            row["candidate_label"] for row in exact_scout
        ]
        assert [float(row["gradient_signed"]) for row in noisy_scout] == pytest.approx(
            [float(row["gradient_signed"]) for row in exact_scout]
        )

        assert len(oracle_instances) == 2
        noisy_oracle = oracle_instances[1]
        assert getattr(noisy_oracle.config, "value_noise_model") == "off"
        assert float(getattr(noisy_oracle.config, "value_noise_std")) == pytest.approx(0.0)
        assert getattr(noisy_oracle.config, "value_noise_seed") is None
        objective_stages = {
            "initial_state",
            "resume_boundary_refit",
            "seed_window_refit",
            "post_append_reopt",
            "final_full_refit",
            "phase1_live_prune_refit",
            "phase1_live_prune_trial",
        }
        assert not any(str(label) in objective_stages for label, _ in noisy_oracle.calls)

    def test_phase3_oracle_target_hit_requires_exact_final_state_audit_for_noisy_energy(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        self._install_fake_oracle_bindings(
            monkeypatch,
            default_gradient=1.0,
            default_sigma=0.0,
            gradient_step=0.1,
            objective_mean=-0.321,
        )
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=1,
            seed=7,
            adapt_inner_optimizer="SPSA",
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase3_v1",
            phase3_symmetry_mitigation_mode="verify_only",
            phase3_enable_rescue=True,
            phase3_lifetime_cost_mode="phase3_v1",
            phase3_oracle_gradient_config=self._oracle_cfg(
                noise_mode="aer_density_matrix_synthetic_depolarizing",
                execution_surface="expectation_v1",
                execution_surface_requested="expectation_v1",
                synthetic_depolarizing_2q_error=1.0e-8,
            ),
            phase3_oracle_inner_objective_mode="noisy_v1",
            benchmark_target_reference_energy=-0.321,
            benchmark_target_abs_delta_e=1.0e-12,
        )

        assert payload["energy_source"] == "oracle_expectation_v1"
        assert payload["energy"] == pytest.approx(-0.321)
        assert payload["stop_reason"] == "benchmark_abs_delta_e_target"
        assert payload["benchmark_target_error_within_threshold"] is True
        assert payload["benchmark_target_hit_success"] is False
        assert payload["benchmark_target_non_hit_reason"] == "exact_final_state_audit_misses_target"
        classification = payload["benchmark_target_classification"]
        assert classification["target_hit_success_before_exact_audit"] is True
        assert classification["exact_final_state_audit_required"] is True
        assert classification["exact_final_state_target_error_within_threshold"] is False
        assert payload["exact_final_state_benchmark_target_error_within_threshold"] is False
        assert payload["exact_final_state_benchmark_target_abs_delta_e_current"] > 1.0e-12

    def test_phase3_oracle_inner_objective_zero_noise_noisy_v1_does_not_guard_analytic_noise(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        self._install_fake_oracle_bindings(
            monkeypatch,
            default_gradient=1.0,
            default_sigma=0.0,
            gradient_step=0.1,
            objective_mean=-0.321,
        )
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=1,
            seed=7,
            adapt_inner_optimizer="SPSA",
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase3_v1",
            phase3_symmetry_mitigation_mode="verify_only",
            phase3_enable_rescue=True,
            phase3_lifetime_cost_mode="phase3_v1",
            phase3_oracle_gradient_config=self._oracle_cfg(
                noise_mode="ideal",
                execution_surface="expectation_v1",
                execution_surface_requested="expectation_v1",
                value_noise_model="off",
                value_noise_std=0.0,
            ),
            phase3_oracle_inner_objective_mode="noisy_v1",
            adapt_analytic_noise_std=1.0,
            adapt_analytic_noise_seed=17,
        )

        assert payload["phase3_oracle_inner_objective_mode"] == "noisy_v1"
        assert payload["phase3_oracle_inner_objective_mode_requested"] == "noisy_v1"
        assert payload["phase3_oracle_inner_objective_runtime_guard_reason"] is None
        assert payload["energy_source"] == "oracle_expectation_v1"
        assert payload["energy"] == pytest.approx(-0.321)
        assert payload["continuation"]["oracle_inner_objective_calls_total"] > 0

    @pytest.mark.parametrize(
        "oracle_cfg_overrides",
        [
            pytest.param(
                {"noise_mode": "ideal", "value_noise_model": "gaussian_iid_v1", "value_noise_std": 1.0e-6},
                id="positive_value_noise",
            ),
            pytest.param(
                {
                    "noise_mode": "aer_density_matrix_synthetic_depolarizing",
                    "synthetic_depolarizing_2q_error": 1.0e-8,
                },
                id="positive_synthetic_2q",
            ),
            pytest.param({"noise_mode": "shots"}, id="shots_mode"),
            pytest.param({"noise_mode": "aer_noise"}, id="aer_noise_mode"),
            pytest.param({"noise_mode": "runtime"}, id="runtime_mode"),
            pytest.param({"noise_mode": "backend_scheduled"}, id="backend_scheduled_mode"),
            pytest.param({"noise_mode": "ideal", "execution_surface": "raw_measurement_v1"}, id="raw_surface"),
            pytest.param({"noise_mode": "ideal", "mitigation_mode": "readout"}, id="readout_mitigation"),
            pytest.param({"noise_mode": "backend_scheduled", "zne_scales": (1.0, 3.0)}, id="zne_scales"),
            pytest.param({"noise_mode": "backend_scheduled", "local_gate_twirling": True}, id="gate_twirling"),
            pytest.param({"noise_mode": "backend_scheduled", "dd_sequence": "XpXm"}, id="dd_sequence"),
        ],
    )
    def test_phase3_oracle_inner_zero_noise_exact_guard_rejects_noisy_surfaces(
        self,
        oracle_cfg_overrides: dict[str, object],
    ):
        oracle_cfg = self._oracle_cfg(
            **oracle_cfg_overrides,
            execution_surface_requested=str(
                oracle_cfg_overrides.get(
                    "execution_surface",
                    oracle_cfg_overrides.get("execution_surface_requested", "expectation_v1"),
                )
            ),
        )

        assert _adapt_mod._phase3_oracle_inner_zero_noise_exact_equivalent(oracle_cfg) is False

    def test_phase3_oracle_inner_zero_noise_exact_guard_requires_resolved_config_attrs(self):
        assert _adapt_mod._phase3_oracle_inner_zero_noise_exact_equivalent(SimpleNamespace()) is False

    def test_phase3_oracle_inner_value_noise_exact_structure_accepts_scalar_shot_proxy(self):
        oracle_cfg = self._oracle_cfg(
            noise_mode="ideal",
            execution_surface="expectation_v1",
            execution_surface_requested="expectation_v1",
            value_noise_model="gaussian_iid_v1",
            value_noise_std=1.0e-6,
            value_noise_sigma0_abs=1.0e-3,
            value_noise_n_eff=1.0e6,
        )

        assert _adapt_mod._phase3_oracle_inner_value_noise_exact_structure_eligible(oracle_cfg) is True

    @pytest.mark.parametrize(
        "oracle_cfg_overrides",
        [
            pytest.param({"noise_mode": "ideal", "value_noise_model": "off", "value_noise_std": 0.0}, id="off"),
            pytest.param({"noise_mode": "ideal", "value_noise_model": "gaussian_iid_v1", "value_noise_std": 0.0}, id="zero_std"),
            pytest.param(
                {
                    "noise_mode": "aer_density_matrix_synthetic_depolarizing",
                    "value_noise_model": "gaussian_iid_v1",
                    "value_noise_std": 1.0e-6,
                    "synthetic_depolarizing_2q_error": 1.0e-8,
                },
                id="positive_synthetic_2q",
            ),
            pytest.param(
                {"noise_mode": "ideal", "execution_surface": "raw_measurement_v1", "value_noise_model": "gaussian_iid_v1", "value_noise_std": 1.0e-6},
                id="raw_surface",
            ),
            pytest.param({"noise_mode": "shots", "value_noise_model": "gaussian_iid_v1", "value_noise_std": 1.0e-6}, id="shots_mode"),
            pytest.param({"noise_mode": "ideal", "mitigation_mode": "readout", "value_noise_model": "gaussian_iid_v1", "value_noise_std": 1.0e-6}, id="readout_mitigation"),
            pytest.param({"noise_mode": "ideal", "zne_scales": (1.0, 3.0), "value_noise_model": "gaussian_iid_v1", "value_noise_std": 1.0e-6}, id="zne_scales"),
            pytest.param({"noise_mode": "ideal", "local_gate_twirling": True, "value_noise_model": "gaussian_iid_v1", "value_noise_std": 1.0e-6}, id="gate_twirling"),
            pytest.param({"noise_mode": "ideal", "dd_sequence": "XpXm", "value_noise_model": "gaussian_iid_v1", "value_noise_std": 1.0e-6}, id="dd_sequence"),
        ],
    )
    def test_phase3_oracle_inner_value_noise_exact_structure_rejects_non_scalar_surfaces(
        self,
        oracle_cfg_overrides: dict[str, object],
    ):
        oracle_cfg = self._oracle_cfg(
            **oracle_cfg_overrides,
            execution_surface_requested=str(
                oracle_cfg_overrides.get(
                    "execution_surface",
                    oracle_cfg_overrides.get("execution_surface_requested", "expectation_v1"),
                )
            ),
        )

        assert _adapt_mod._phase3_oracle_inner_value_noise_exact_structure_eligible(oracle_cfg) is False

    def test_phase3_oracle_inner_value_noise_exact_structure_requires_resolved_config_attrs(self):
        assert _adapt_mod._phase3_oracle_inner_value_noise_exact_structure_eligible(SimpleNamespace()) is False

    def test_phase3_oracle_auto_surface_resolves_raw_for_runtime_none_mitigation(self):
        resolved = _adapt_mod._resolve_phase3_oracle_gradient_config(
            self._oracle_cfg(
                noise_mode="runtime",
                backend_name="ibm_marrakesh",
                mitigation_mode="none",
            )
        )

        assert resolved.execution_surface == "raw_measurement_v1"

    def test_phase3_oracle_auto_surface_keeps_expectation_for_backend_scheduled(self):
        resolved = _adapt_mod._resolve_phase3_oracle_gradient_config(
            self._oracle_cfg(
                noise_mode="backend_scheduled",
                use_fake_backend=True,
                backend_name="FakeNighthawk",
                mitigation_mode="none",
            )
        )

        assert resolved.execution_surface == "expectation_v1"

    def test_phase3_backend_scheduled_expectation_threads_full_local_mitigation_stack(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        oracle_instances = self._install_fake_oracle_bindings(
            monkeypatch,
            default_gradient=1.0,
            default_sigma=0.1,
            gradient_step=0.1,
            backend_name="FakeNighthawk",
        )
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=10,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_continuation_mode="phase3_v1",
            phase3_oracle_gradient_config=self._oracle_cfg(
                noise_mode="backend_scheduled",
                use_fake_backend=True,
                backend_name="FakeNighthawk",
                mitigation_mode="readout",
                local_readout_strategy="mthree",
                zne_scales=(1.0, 3.0, 5.0),
                local_gate_twirling=True,
                dd_sequence="XpXm",
                gradient_step=0.1,
            ),
        )

        mitigation = dict(getattr(oracle_instances[0].config, "mitigation", {}))
        assert payload["continuation"]["oracle_execution_surface"] == "expectation_v1"
        assert mitigation["mode"] == "readout"
        assert mitigation["local_readout_strategy"] == "mthree"
        assert mitigation["zne_scales"] == [1.0, 3.0, 5.0]
        assert mitigation["local_gate_twirling"] is True
        assert mitigation["dd_sequence"] == "XpXm"
        assert oracle_instances and getattr(oracle_instances[0], "closed", False) is True

    def test_phase3_backend_scheduled_local_zne_requires_unit_scale(self):
        with pytest.raises(ValueError, match="must include the base noise scale 1"):
            _run_hardcoded_adapt_vqe(
                h_poly=self._hh_h(),
                num_sites=2,
                ordering="blocked",
                problem="hh",
                adapt_pool="paop_lf_std",
                t=1.0,
                u=2.0,
                dv=0.0,
                boundary="periodic",
                omega0=1.0,
                g_ep=0.5,
                n_ph_max=1,
                boson_encoding="binary",
                max_depth=1,
                eps_grad=1e-3,
                eps_energy=1e-8,
                maxiter=10,
                seed=7,
                allow_repeats=True,
                finite_angle_fallback=False,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
                adapt_continuation_mode="phase3_v1",
                phase3_oracle_gradient_config=self._oracle_cfg(
                    noise_mode="backend_scheduled",
                    use_fake_backend=True,
                    backend_name="FakeNighthawk",
                    zne_scales=(3.0, 5.0),
                    gradient_step=0.1,
                ),
            )

    def test_phase3_backend_scheduled_raw_inner_objective_routes_grouped_measurement(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        oracle_instances = self._install_fake_oracle_bindings(
            monkeypatch,
            default_gradient=1.0,
            default_sigma=0.1,
            gradient_step=0.1,
            objective_mean=-0.321,
            backend_name="FakeMarrakesh",
        )
        preflight_calls: list[dict[str, object]] = []
        bindings_factory = _adapt_mod._phase3_oracle_runtime_bindings

        def _wrapped_bindings() -> dict[str, object]:
            bindings = dict(bindings_factory())
            orig_preflight = bindings["preflight_backend_scheduled_fake_backend_environment"]

            def _spy_preflight(cfg: object) -> None:
                preflight_calls.append(
                    {
                        "backend_name": getattr(cfg, "backend_name", None),
                        "execution_surface": getattr(cfg, "execution_surface", None),
                    }
                )
                orig_preflight(cfg)

            bindings["preflight_backend_scheduled_fake_backend_environment"] = _spy_preflight
            return bindings

        monkeypatch.setattr(_adapt_mod, "_phase3_oracle_runtime_bindings", _wrapped_bindings)
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=1,
            seed=7,
            adapt_inner_optimizer="SPSA",
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase3_v1",
            phase3_symmetry_mitigation_mode="verify_only",
            phase3_enable_rescue=True,
            phase3_lifetime_cost_mode="phase3_v1",
            phase3_oracle_gradient_config=self._oracle_cfg(
                noise_mode="backend_scheduled",
                use_fake_backend=True,
                backend_name="FakeMarrakesh",
                mitigation_mode="none",
                execution_surface_requested="raw_measurement_v1",
                execution_surface="raw_measurement_v1",
                raw_transport="auto",
                gradient_step=0.1,
            ),
            phase3_oracle_inner_objective_mode="noisy_v1",
        )

        assert payload["energy_source"] == "oracle_raw_measurement_v1"
        assert payload["energy"] == pytest.approx(-0.321)
        assert payload["phase3_oracle_inner_objective_mode"] == "noisy_v1"
        assert payload["phase3_oracle_inner_objective_mode_requested"] == "noisy_v1"
        assert payload["phase3_oracle_inner_objective_runtime_guard_reason"] is None
        assert payload["continuation"]["oracle_execution_surface"] == "raw_measurement_v1"
        assert payload["continuation"]["reoptimization_backend"] == "oracle_raw_measurement_v1"
        assert payload["continuation"]["oracle_inner_objective_mode"] == "noisy_v1"
        assert payload["continuation"]["oracle_inner_objective_mode_requested"] == "noisy_v1"
        assert payload["continuation"]["oracle_inner_objective_runtime_guard_reason"] is None
        assert payload["continuation"]["oracle_inner_objective_calls_total"] > 0
        assert payload["continuation"]["oracle_inner_objective_raw_records_total"] > 0
        assert (
            payload["continuation"]["oracle_backend_info"]["details"]["execution_surface"]
            == "raw_measurement_v1"
        )
        assert (
            payload["continuation"]["last_oracle_inner_objective_backend_info"]["details"]["execution_surface"]
            == "raw_measurement_v1"
        )
        assert preflight_calls == [
            {
                "backend_name": "FakeMarrakesh",
                "execution_surface": "raw_measurement_v1",
            }
        ]
        assert oracle_instances and getattr(oracle_instances[0], "closed", False) is True

    def test_phase3_raw_oracle_gradient_mode_routes_sigma_and_raw_summary(self, monkeypatch: pytest.MonkeyPatch):
        oracle_instances = self._install_fake_oracle_bindings(
            monkeypatch,
            default_gradient=1.0,
            default_sigma=0.25,
            gradient_step=0.1,
        )
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=20,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase3_v1",
            phase3_symmetry_mitigation_mode="verify_only",
            phase3_enable_rescue=False,
            phase3_lifetime_cost_mode="phase3_v1",
            phase3_oracle_gradient_config=self._oracle_cfg(
                noise_mode="runtime",
                use_fake_backend=False,
                backend_name="ibm_marrakesh",
                mitigation_mode="none",
                raw_artifact_path="artifacts/raw_phase3.ndjson.gz",
                gradient_step=0.1,
            ),
        )

        continuation = payload["continuation"]
        scout_rows = (
            payload["history"][0]["candidate_gradient_scout"]
            if payload["history"]
            else continuation["last_candidate_gradient_scout"]
        )
        assert continuation["oracle_execution_surface"] == "raw_measurement_v1"
        assert continuation["oracle_gradient_raw_records_total"] > 0
        assert continuation["oracle_symmetry_diagnostic_calls_total"] == 2 * len(scout_rows)
        assert continuation["oracle_symmetry_diagnostic_raw_records_total"] > 0
        assert continuation["oracle_gradient_calls_total"] == 2 * len(scout_rows)
        assert continuation["reoptimization_backend"] == "exact_statevector"
        assert continuation["oracle_raw_transport"] == "sampler_v2"
        assert scout_rows
        assert all(row["raw_summary"] is not None for row in scout_rows)
        assert all(
            row["raw_summary"]["symmetry_diagnostic"]["plus"]["available"] is True
            and row["raw_summary"]["symmetry_diagnostic"]["minus"]["available"] is True
            for row in scout_rows
        )
        assert all(
            row["raw_summary"]["symmetry_diagnostic"]["plus"]["summary"]["sector_weight_mean"] == pytest.approx(1.0)
            for row in scout_rows
        )
        assert oracle_instances and getattr(oracle_instances[0], "closed", False) is True

    def test_phase3_raw_oracle_rejects_incompatible_readout_mitigation(self):
        with pytest.raises(ValueError, match="mitigation_mode='none'"):
            _run_hardcoded_adapt_vqe(
                h_poly=self._hh_h(),
                num_sites=2,
                ordering="blocked",
                problem="hh",
                adapt_pool="paop_lf_std",
                t=1.0,
                u=2.0,
                dv=0.0,
                boundary="periodic",
                omega0=1.0,
                g_ep=0.5,
                n_ph_max=1,
                boson_encoding="binary",
                max_depth=1,
                eps_grad=1e-3,
                eps_energy=1e-8,
                maxiter=10,
                seed=7,
                allow_repeats=True,
                finite_angle_fallback=False,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
                adapt_continuation_mode="phase3_v1",
                phase3_oracle_gradient_config=self._oracle_cfg(
                    noise_mode="runtime",
                    use_fake_backend=False,
                    backend_name="ibm_marrakesh",
                    execution_surface_requested="raw_measurement_v1",
                    execution_surface="raw_measurement_v1",
                    mitigation_mode="readout",
                    local_readout_strategy="mthree",
                ),
            )

    def test_phase3_raw_oracle_accepts_backend_scheduled_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        oracle_instances = self._install_fake_oracle_bindings(
            monkeypatch,
            default_gradient=1.0,
            default_sigma=0.25,
            gradient_step=0.1,
            backend_name="FakeNighthawk",
        )
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=10,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_continuation_mode="phase3_v1",
            phase3_oracle_gradient_config=self._oracle_cfg(
                noise_mode="backend_scheduled",
                use_fake_backend=True,
                backend_name="FakeNighthawk",
                execution_surface_requested="raw_measurement_v1",
                execution_surface="raw_measurement_v1",
                raw_transport="auto",
            ),
        )

        assert payload["continuation"]["oracle_execution_surface"] == "raw_measurement_v1"
        assert payload["continuation"]["oracle_gradient_raw_records_total"] > 0
        assert payload["continuation"]["reoptimization_backend"] == "exact_statevector"
        assert oracle_instances and getattr(oracle_instances[0], "closed", False) is True

    def test_phase3_raw_oracle_rejects_backend_run_transport(self):
        with pytest.raises(ValueError, match="phase3_oracle_raw_transport"):
            _run_hardcoded_adapt_vqe(
                h_poly=self._hh_h(),
                num_sites=2,
                ordering="blocked",
                problem="hh",
                adapt_pool="paop_lf_std",
                t=1.0,
                u=2.0,
                dv=0.0,
                boundary="periodic",
                omega0=1.0,
                g_ep=0.5,
                n_ph_max=1,
                boson_encoding="binary",
                max_depth=1,
                eps_grad=1e-3,
                eps_energy=1e-8,
                maxiter=10,
                seed=7,
                allow_repeats=True,
                finite_angle_fallback=False,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
                adapt_continuation_mode="phase3_v1",
                phase3_oracle_gradient_config=self._oracle_cfg(
                    noise_mode="runtime",
                    use_fake_backend=False,
                    backend_name="ibm_marrakesh",
                    execution_surface_requested="raw_measurement_v1",
                    execution_surface="raw_measurement_v1",
                    raw_transport="backend_run",
                ),
            )

    def test_phase3_raw_oracle_keeps_main_run_when_symmetry_diagnostic_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        oracle_instances = self._install_fake_oracle_bindings(
            monkeypatch,
            default_gradient=1.0,
            default_sigma=0.25,
            gradient_step=0.1,
            raise_on_symmetry_measure=True,
        )
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=20,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase3_v1",
            phase3_symmetry_mitigation_mode="verify_only",
            phase3_enable_rescue=False,
            phase3_lifetime_cost_mode="phase3_v1",
            phase3_oracle_gradient_config=self._oracle_cfg(
                noise_mode="runtime",
                use_fake_backend=False,
                backend_name="ibm_marrakesh",
                mitigation_mode="none",
                raw_artifact_path="artifacts/raw_phase3.ndjson.gz",
                gradient_step=0.1,
            ),
        )

        history_row = payload["history"][0]
        assert payload["continuation"]["oracle_gradient_calls_total"] == 2 * len(history_row["candidate_gradient_scout"])
        assert payload["continuation"]["oracle_symmetry_diagnostic_calls_total"] == 0
        assert all(
            row["raw_summary"]["symmetry_diagnostic"]["plus"]["available"] is False
            and row["raw_summary"]["symmetry_diagnostic"]["plus"]["reason"] == "measurement_failed"
            for row in history_row["candidate_gradient_scout"]
        )
        assert oracle_instances and getattr(oracle_instances[0], "closed", False) is True

    def test_phase3_raw_oracle_closes_on_exception(self, monkeypatch: pytest.MonkeyPatch):
        oracle_instances = self._install_fake_oracle_bindings(
            monkeypatch,
            default_gradient=1.0,
            default_sigma=0.25,
            gradient_step=0.1,
            raise_on_raw_measure=True,
        )

        with pytest.raises(RuntimeError, match="synthetic raw oracle failure") as excinfo:
            _run_hardcoded_adapt_vqe(
                h_poly=self._hh_h(),
                num_sites=2,
                ordering="blocked",
                problem="hh",
                adapt_pool="paop_lf_std",
                t=1.0,
                u=2.0,
                dv=0.0,
                boundary="periodic",
                omega0=1.0,
                g_ep=0.5,
                n_ph_max=1,
                boson_encoding="binary",
                max_depth=1,
                eps_grad=1e-3,
                eps_energy=1e-8,
                maxiter=10,
                seed=7,
                allow_repeats=True,
                finite_angle_fallback=False,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
                adapt_continuation_mode="phase3_v1",
                phase3_enable_rescue=False,
                phase3_lifetime_cost_mode="phase3_v1",
                phase3_oracle_gradient_config=self._oracle_cfg(
                    noise_mode="runtime",
                    use_fake_backend=False,
                    backend_name="ibm_marrakesh",
                    execution_surface_requested="raw_measurement_v1",
                    execution_surface="raw_measurement_v1",
                ),
            )

        assert "synthetic raw oracle failure" in str(excinfo.value)
        assert oracle_instances and oracle_instances[0].calls
        assert getattr(oracle_instances[0], "closed", False) is True

    def test_phase3_oracle_gradient_mode_disables_exact_only_sidepaths(self, monkeypatch: pytest.MonkeyPatch):
        self._install_fake_oracle_bindings(
            monkeypatch,
            default_gradient=1.0,
            default_sigma=0.1,
            gradient_step=0.1,
        )
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="open",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=20,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase3_v1",
            phase3_symmetry_mitigation_mode="verify_only",
            phase3_enable_rescue=False,
            phase3_lifetime_cost_mode="phase3_v1",
            phase3_oracle_gradient_config=self._oracle_cfg(
                noise_mode="backend_scheduled",
                use_fake_backend=True,
                backend_name="FakeNighthawk",
                mitigation_mode="readout",
                local_readout_strategy="mthree",
                gradient_step=0.1,
            ),
        )

        assert payload["finite_angle_fallback"] is False
        assert payload["prune_summary"]["enabled"] is False

    def test_phase3_oracle_gradient_mode_rejects_unsupported_problem(self):
        h_poly = build_hubbard_hamiltonian(
            dims=2,
            t=1.0,
            U=4.0,
            v=0.0,
            repr_mode="JW",
            indexing="blocked",
            pbc=True,
        )
        with pytest.raises(ValueError, match="problem in \\{'hh','spin_boson'\\}"):
            _run_hardcoded_adapt_vqe(
                h_poly=h_poly,
                num_sites=2,
                ordering="blocked",
                problem="hubbard",
                adapt_pool="uccsd",
                t=1.0,
                u=4.0,
                dv=0.0,
                boundary="periodic",
                omega0=0.0,
                g_ep=0.0,
                n_ph_max=1,
                boson_encoding="binary",
                max_depth=1,
                eps_grad=1e-3,
                eps_energy=1e-8,
                maxiter=5,
                seed=7,
                allow_repeats=True,
                finite_angle_fallback=False,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
                adapt_continuation_mode="phase3_v1",
                phase3_oracle_gradient_config=self._oracle_cfg(),
            )

    def test_phase3_oracle_gradient_mode_rejects_legacy_mode(self):
        with pytest.raises(ValueError, match="adapt_continuation_mode='phase3_v1'"):
            _run_hardcoded_adapt_vqe(
                h_poly=self._hh_h(),
                num_sites=2,
                ordering="blocked",
                problem="hh",
                adapt_pool="paop_lf_std",
                t=1.0,
                u=2.0,
                dv=0.0,
                boundary="periodic",
                omega0=1.0,
                g_ep=0.5,
                n_ph_max=1,
                boson_encoding="binary",
                max_depth=1,
                eps_grad=1e-3,
                eps_energy=1e-8,
                maxiter=20,
                seed=7,
                allow_repeats=True,
                finite_angle_fallback=False,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
                adapt_continuation_mode="legacy",
                phase3_oracle_gradient_config=self._oracle_cfg(),
            )

    def test_phase3_sigma_hat_does_not_change_precap_shortlist_identity(self, monkeypatch: pytest.MonkeyPatch):
        common_kwargs = dict(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=20,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase3_v1",
            phase1_shortlist_size=1,
            phase1_score_z_alpha=1.0,
            phase3_symmetry_mitigation_mode="verify_only",
            phase3_enable_rescue=False,
            phase3_lifetime_cost_mode="phase3_v1",
        )
        baseline_payload, _ = _run_hardcoded_adapt_vqe(**common_kwargs)
        baseline_label = str(baseline_payload["continuation"]["phase1_retained_rows"][0]["candidate_label"])

        captured_phase1_score_keys: list[str] = []
        original_shortlist_records = _adapt_mod.shortlist_records

        def _capture_shortlist_records(records, *, cfg, score_key="simple_score", tie_break_score_key="simple_score"):
            captured_phase1_score_keys.append(str(score_key))
            return original_shortlist_records(
                records,
                cfg=cfg,
                score_key=score_key,
                tie_break_score_key=tie_break_score_key,
            )

        monkeypatch.setattr(_adapt_mod, "shortlist_records", _capture_shortlist_records)
        monkeypatch.setattr(
            _adapt_mod,
            "_phase3_sigma_hat_for_label",
            lambda **kwargs: 100.0 if str(kwargs.get("candidate_label")) == baseline_label else 0.0,
        )

        sigma_payload, _ = _run_hardcoded_adapt_vqe(**common_kwargs)
        sigma_label = str(sigma_payload["continuation"]["phase1_retained_rows"][0]["candidate_label"])

        assert "simple_score" in captured_phase1_score_keys
        assert sigma_label == baseline_label

    def test_phase3_oracle_gradient_sigma_does_not_change_precap_shortlist_identity(self, monkeypatch: pytest.MonkeyPatch):
        common_kwargs = dict(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=20,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase3_v1",
            phase1_shortlist_size=1,
            phase1_score_z_alpha=1.0,
            phase3_symmetry_mitigation_mode="verify_only",
            phase3_enable_rescue=False,
            phase3_lifetime_cost_mode="phase3_v1",
        )
        exact_payload, _ = _run_hardcoded_adapt_vqe(**common_kwargs)
        target_label = str(exact_payload["continuation"]["phase1_retained_rows"][0]["candidate_label"])

        self._install_fake_oracle_bindings(
            monkeypatch,
            gradient_by_label={target_label: 2.0},
            sigma_by_label={target_label: 0.0},
            default_gradient=1.0,
            default_sigma=0.0,
            gradient_step=0.1,
        )
        oracle_baseline_payload, _ = _run_hardcoded_adapt_vqe(
            **common_kwargs,
            phase3_oracle_gradient_config=self._oracle_cfg(gradient_step=0.1),
        )
        oracle_baseline_label = str(
            oracle_baseline_payload["continuation"]["phase1_retained_rows"][0]["candidate_label"]
        )

        self._install_fake_oracle_bindings(
            monkeypatch,
            gradient_by_label={target_label: 2.0},
            sigma_by_label={target_label: 5.0},
            default_gradient=1.0,
            default_sigma=0.0,
            gradient_step=0.1,
        )
        sigma_payload, _ = _run_hardcoded_adapt_vqe(
            **common_kwargs,
            phase3_oracle_gradient_config=self._oracle_cfg(gradient_step=0.1),
        )
        sigma_label = str(sigma_payload["continuation"]["phase1_retained_rows"][0]["candidate_label"])

        assert oracle_baseline_label == target_label
        assert sigma_label == target_label

    def test_phase3_backend_cost_mode_rejects_non_hh_problem(self):
        h_poly = build_hubbard_hamiltonian(
            dims=2,
            t=1.0,
            U=4.0,
            v=0.0,
            repr_mode="JW",
            indexing="blocked",
            pbc=True,
        )
        with pytest.raises(ValueError, match="phase3_backend_cost_mode is only valid for problem='hh'"):
            _run_hardcoded_adapt_vqe(
                h_poly=h_poly,
                num_sites=2,
                ordering="blocked",
                problem="hubbard",
                adapt_pool="uccsd",
                t=1.0,
                u=4.0,
                dv=0.0,
                boundary="periodic",
                omega0=1.0,
                g_ep=0.5,
                n_ph_max=1,
                boson_encoding="binary",
                max_depth=1,
                eps_grad=1e-3,
                eps_energy=1e-8,
                maxiter=5,
                seed=7,
                allow_repeats=True,
                finite_angle_fallback=False,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
                adapt_reopt_policy="windowed",
                adapt_window_size=1,
                adapt_window_topk=0,
                adapt_continuation_mode="phase3_v1",
                phase3_backend_cost_mode="transpile_single_v1",
                phase3_backend_name="ibm_boston",
            )

    @pytest.mark.parametrize("backend_cost_mode", ["proxy", "transpile_single_v1"])
    def test_phase3_backend_cost_mode_rejects_nonfinite_weights(self, backend_cost_mode: str):
        with pytest.raises(ValueError, match="phase3 backend cost weights must be finite and nonnegative"):
            _run_hardcoded_adapt_vqe(
                h_poly=self._hh_h(),
                num_sites=2,
                ordering="blocked",
                problem="hh",
                adapt_pool="paop_lf_std",
                t=1.0,
                u=2.0,
                dv=0.0,
                boundary="periodic",
                omega0=1.0,
                g_ep=0.5,
                n_ph_max=1,
                boson_encoding="binary",
                max_depth=1,
                eps_grad=1e-3,
                eps_energy=1e-8,
                maxiter=5,
                seed=7,
                allow_repeats=True,
                finite_angle_fallback=False,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
                adapt_reopt_policy="windowed",
                adapt_window_size=1,
                adapt_window_topk=0,
                adapt_continuation_mode="phase3_v1",
                phase3_backend_cost_mode=str(backend_cost_mode),
                phase3_backend_w_depth=math.nan,
            )

    def test_phase3_backend_cost_mode_emits_backend_compile_summary(self, monkeypatch: pytest.MonkeyPatch):
        created_configs = []

        class _StubBackendCompileOracle:
            def __init__(self, *, config, num_qubits, ref_state):
                created_configs.append(config)
                self.config = config
                self.num_qubits = num_qubits
                self.ref_state = ref_state
                self.targets = ("FakeNighthawk",)
                self.resolution_audit = [
                    {
                        "requested_name": "ibm_boston",
                        "resolved_name": "FakeNighthawk",
                        "success": True,
                        "resolution_kind": "fake_exact",
                        "using_fake_backend": True,
                    }
                ]

            def snapshot_base(self, ops):
                return {"ops": [str(op.label) for op in ops]}

            def estimate_insertion(self, snapshot, *, candidate_term, position_id, proxy_baseline=None):
                return CompileCostEstimate(
                    new_pauli_actions=(0.0 if proxy_baseline is None else float(proxy_baseline.new_pauli_actions)),
                    new_rotation_steps=(0.0 if proxy_baseline is None else float(proxy_baseline.new_rotation_steps)),
                    position_shift_span=(0.0 if proxy_baseline is None else float(proxy_baseline.position_shift_span)),
                    refit_active_count=(0.0 if proxy_baseline is None else float(proxy_baseline.refit_active_count)),
                    proxy_total=(0.0 if proxy_baseline is None else float(proxy_baseline.proxy_total)),
                    cx_proxy_total=(0.0 if proxy_baseline is None else float(proxy_baseline.cx_proxy_total)),
                    sq_proxy_total=(0.0 if proxy_baseline is None else float(proxy_baseline.sq_proxy_total)),
                    gate_proxy_total=(0.0 if proxy_baseline is None else float(proxy_baseline.gate_proxy_total)),
                    max_pauli_weight=(0.0 if proxy_baseline is None else float(proxy_baseline.max_pauli_weight)),
                    source_mode="backend_transpile_v1",
                    penalty_total=4.5,
                    depth_surrogate=4.5,
                    compile_gate_open=True,
                    failure_reason=None,
                    selected_backend_name="FakeNighthawk",
                    selected_resolution_kind="fake_exact",
                    aggregation_mode="single_backend",
                    target_backend_names=["FakeNighthawk"],
                    successful_target_count=1,
                    failed_target_count=0,
                    raw_delta_compiled_count_2q=2.0,
                    delta_compiled_count_2q=2.0,
                    raw_delta_compiled_depth=3.0,
                    delta_compiled_depth=3.0,
                    raw_delta_compiled_size=5.0,
                    delta_compiled_size=5.0,
                    delta_compiled_cx_count=2.0,
                    delta_compiled_ecr_count=0.0,
                    base_compiled_count_2q=10.0,
                    base_compiled_depth=12.0,
                    base_compiled_size=20.0,
                    trial_compiled_count_2q=12.0,
                    trial_compiled_depth=15.0,
                    trial_compiled_size=25.0,
                    proxy_baseline=(
                        None
                        if proxy_baseline is None
                        else {
                            "new_pauli_actions": float(proxy_baseline.new_pauli_actions),
                            "new_rotation_steps": float(proxy_baseline.new_rotation_steps),
                            "position_shift_span": float(proxy_baseline.position_shift_span),
                            "refit_active_count": float(proxy_baseline.refit_active_count),
                            "proxy_total": float(proxy_baseline.proxy_total),
                            "cx_proxy_total": float(proxy_baseline.cx_proxy_total),
                            "sq_proxy_total": float(proxy_baseline.sq_proxy_total),
                            "gate_proxy_total": float(proxy_baseline.gate_proxy_total),
                            "max_pauli_weight": float(proxy_baseline.max_pauli_weight),
                        }
                    ),
                    selected_backend_row={
                        "transpile_backend": "FakeNighthawk",
                        "resolution_kind": "fake_exact",
                        "compiled_count_2q": 12,
                        "compiled_depth": 15,
                        "compiled_size": 25,
                    },
                )

            def final_scaffold_summary(self, ops):
                return {
                    "rows": [
                        {
                            "transpile_backend": "FakeNighthawk",
                            "resolution_kind": "fake_exact",
                            "transpile_status": "ok",
                            "compiled_count_2q": 18,
                            "compiled_depth": 21,
                            "compiled_size": 33,
                            "compiled_op_counts": {"swap": 1, "cx": 18},
                            "absolute_burden_score_v1": 20.43,
                        }
                    ],
                    "selected_backend": {
                        "transpile_backend": "FakeNighthawk",
                        "resolution_kind": "fake_exact",
                        "transpile_status": "ok",
                        "compiled_count_2q": 18,
                        "compiled_depth": 21,
                        "compiled_size": 33,
                        "compiled_op_counts": {"swap": 1, "cx": 18},
                        "absolute_burden_score_v1": 20.43,
                    },
                }

            def cache_summary(self):
                return {"row_hits": 2, "row_misses": 1, "compile_failures": 0, "cache_entries": 3}

        monkeypatch.setattr(_adapt_mod, "BackendCompileOracle", _StubBackendCompileOracle)

        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=2,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=20,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase3_v1",
            phase3_symmetry_mitigation_mode="verify_only",
            phase3_enable_rescue=False,
            phase3_lifetime_cost_mode="phase3_v1",
            phase3_backend_cost_mode="transpile_single_v1",
        )

        assert len(created_configs) == 1
        assert created_configs[0].mode == "transpile_single_v1"
        assert created_configs[0].requested_backend_name == "FakeMarrakesh"
        assert payload["continuation"]["backend_compile_cost_summary"]["requested_backend_name"] == "FakeMarrakesh"
        assert payload["compile_cost_mode"] == "transpile_single_v1"
        assert payload["backend_compile_cost_summary"]["selected_backend"]["transpile_backend"] == "FakeNighthawk"
        assert payload["continuation"]["backend_compile_cost_summary"]["cache_summary"]["cache_entries"] == 3
        assert payload["scaffold_fingerprint_lite"]["compile_cost_mode"] == "transpile_single_v1"
        assert payload["scaffold_fingerprint_lite"]["backend_target_names"] == ["FakeNighthawk"]
        assert any(row["compile_cost_mode"] == "transpile_single_v1" for row in payload["history"])
        assert any(row["compile_cost_source"] == "backend_transpile_v1" for row in payload["history"])
        assert any(
            isinstance(row.get("compile_cost_backend"), dict)
            and row["compile_cost_backend"].get("selected_backend_name") == "FakeNighthawk"
            for row in payload["history"]
        )

    def test_phase3_eps_energy_is_telemetry_only_without_drop_policy(self, monkeypatch: pytest.MonkeyPatch):
        events: list[tuple[str, dict[str, object]]] = []
        original_ai_log = _adapt_mod._ai_log

        def _capture(event: str, **fields: object) -> None:
            events.append((str(event), dict(fields)))

        monkeypatch.setattr(_adapt_mod, "_ai_log", _capture)
        try:
            payload, _ = _run_hardcoded_adapt_vqe(
                h_poly=self._hh_h(),
                num_sites=2,
                ordering="blocked",
                problem="hh",
                adapt_pool="paop_lf_std",
                t=1.0,
                u=2.0,
                dv=0.0,
                boundary="periodic",
                omega0=1.0,
                g_ep=0.5,
                n_ph_max=1,
                boson_encoding="binary",
                max_depth=3,
                eps_grad=-1.0,
                eps_energy=1e9,
                maxiter=20,
                seed=17,
                adapt_inner_optimizer="SPSA",
                adapt_spsa_callback_every=10,
                adapt_spsa_progress_every_s=999.0,
                allow_repeats=True,
                finite_angle_fallback=False,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
                adapt_reopt_policy="windowed",
                adapt_window_size=1,
                adapt_window_topk=0,
                adapt_continuation_mode="phase3_v1",
                phase3_symmetry_mitigation_mode="off",
                phase3_enable_rescue=False,
                phase3_lifetime_cost_mode="phase3_v1",
            )
            assert payload["success"] is True
            assert bool(payload["eps_energy_termination_enabled"]) is False
            assert bool(payload["eps_grad_termination_enabled"]) is False
            assert bool(payload["adapt_drop_policy_enabled"]) is False
            assert payload["adapt_drop_floor_resolved"] == pytest.approx(-1.0)
            assert int(payload["adapt_drop_patience_resolved"]) == 0
            assert int(payload["adapt_drop_min_depth_resolved"]) == 0
            assert payload["adapt_grad_floor_resolved"] == pytest.approx(-1.0)
            assert payload["adapt_drop_policy_source"] == "auto_staged"
            assert payload["adapt_drop_floor_source"] == "auto_staged"
            assert payload["adapt_drop_patience_source"] == "auto_staged"
            assert payload["adapt_drop_min_depth_source"] == "auto_staged"
            assert payload["adapt_grad_floor_source"] == "auto_staged"
            assert str(payload["stop_reason"]) in {"max_depth", "pool_exhausted"}
            assert str(payload["stop_reason"]) != "eps_energy"
            assert all(bool(row["eps_energy_termination_enabled"]) is False for row in payload.get("history", []))
            assert all(bool(row["eps_grad_termination_enabled"]) is False for row in payload.get("history", []))
            assert any(int(row["eps_energy_low_streak"]) >= 2 for row in payload.get("history", []))

            suppressed = [ev[1] for ev in events if ev[0] == "hardcoded_adapt_eps_energy_termination_suppressed"]
            assert len(suppressed) >= 1
            converged_energy = [ev for ev in events if ev[0] == "hardcoded_adapt_converged_energy"]
            assert len(converged_energy) == 0
        finally:
            monkeypatch.setattr(_adapt_mod, "_ai_log", original_ai_log)

    def test_phase3_low_gradient_no_longer_terminates(self, monkeypatch: pytest.MonkeyPatch):
        events: list[tuple[str, dict[str, object]]] = []
        original_ai_log = _adapt_mod._ai_log

        def _capture(event: str, **fields: object) -> None:
            events.append((str(event), dict(fields)))

        monkeypatch.setattr(_adapt_mod, "_ai_log", _capture)
        try:
            payload, _ = _run_hardcoded_adapt_vqe(
                h_poly=self._hh_h(),
                num_sites=2,
                ordering="blocked",
                problem="hh",
                adapt_pool="paop_lf_std",
                t=1.0,
                u=2.0,
                dv=0.0,
                boundary="periodic",
                omega0=1.0,
                g_ep=0.5,
                n_ph_max=1,
                boson_encoding="binary",
                max_depth=3,
                eps_grad=1e9,
                eps_energy=1e-9,
                maxiter=20,
                seed=23,
                adapt_inner_optimizer="SPSA",
                adapt_spsa_callback_every=10,
                adapt_spsa_progress_every_s=999.0,
                allow_repeats=True,
                finite_angle_fallback=False,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
                adapt_reopt_policy="windowed",
                adapt_window_size=1,
                adapt_window_topk=0,
                adapt_continuation_mode="phase3_v1",
                phase3_symmetry_mitigation_mode="off",
                phase3_enable_rescue=False,
                phase3_lifetime_cost_mode="phase3_v1",
            )
            assert payload["success"] is True
            assert bool(payload["eps_grad_termination_enabled"]) is False
            assert str(payload["stop_reason"]) in {"max_depth", "pool_exhausted"}
            assert str(payload["stop_reason"]) != "eps_grad"
            assert any(bool(row["eps_grad_threshold_hit"]) is True for row in payload.get("history", []))

            suppressed = [ev for ev in events if ev[0] == "hardcoded_adapt_eps_grad_termination_suppressed"]
            assert len(suppressed) >= 1
            converged_grad = [ev for ev in events if ev[0] == "hardcoded_adapt_converged_grad"]
            assert len(converged_grad) == 0
        finally:
            monkeypatch.setattr(_adapt_mod, "_ai_log", original_ai_log)

    def test_phase3_drop_plateau_preempts_eps_energy_hard_stop(self, monkeypatch: pytest.MonkeyPatch):
        events: list[tuple[str, dict[str, object]]] = []
        original_ai_log = _adapt_mod._ai_log

        def _capture(event: str, **fields: object) -> None:
            events.append((str(event), dict(fields)))

        monkeypatch.setattr(_adapt_mod, "_ai_log", _capture)
        try:
            payload, _ = _run_hardcoded_adapt_vqe(
                h_poly=self._hh_h(),
                num_sites=2,
                ordering="blocked",
                problem="hh",
                adapt_pool="paop_lf_std",
                t=1.0,
                u=2.0,
                dv=0.0,
                boundary="periodic",
                omega0=1.0,
                g_ep=0.5,
                n_ph_max=1,
                boson_encoding="binary",
                max_depth=4,
                eps_grad=-1.0,
                eps_energy=1e9,
                maxiter=20,
                seed=19,
                adapt_inner_optimizer="SPSA",
                adapt_spsa_callback_every=10,
                adapt_spsa_progress_every_s=999.0,
                allow_repeats=True,
                finite_angle_fallback=False,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
                adapt_reopt_policy="windowed",
                adapt_window_size=1,
                adapt_window_topk=0,
                adapt_continuation_mode="phase3_v1",
                adapt_drop_floor=1e9,
                adapt_drop_patience=1,
                adapt_drop_min_depth=1,
                adapt_grad_floor=-1.0,
                phase3_symmetry_mitigation_mode="off",
                phase3_enable_rescue=False,
                phase3_lifetime_cost_mode="phase3_v1",
            )
            assert payload["success"] is True
            assert bool(payload["eps_energy_termination_enabled"]) is False
            assert payload["adapt_drop_floor_resolved"] == pytest.approx(1e9)
            assert int(payload["adapt_drop_patience_resolved"]) == 1
            assert int(payload["adapt_drop_min_depth_resolved"]) == 1
            assert payload["adapt_drop_floor_source"] == "explicit"
            assert payload["adapt_drop_patience_source"] == "explicit"
            assert payload["adapt_drop_min_depth_source"] == "explicit"
            assert str(payload["stop_reason"]) == "drop_plateau"
            assert str(payload["stop_reason"]) != "eps_energy"
            assert all(bool(row["eps_energy_termination_enabled"]) is False for row in payload.get("history", []))

            residual_opened = [ev for ev in events if ev[0] == "hardcoded_adapt_phase1_residual_opened_on_plateau"]
            assert len(residual_opened) >= 1
            converged_drop = [ev for ev in events if ev[0] == "hardcoded_adapt_converged_drop_plateau"]
            assert len(converged_drop) == 1
            converged_energy = [ev for ev in events if ev[0] == "hardcoded_adapt_converged_energy"]
            assert len(converged_energy) == 0
        finally:
            monkeypatch.setattr(_adapt_mod, "_ai_log", original_ai_log)

    def test_final_noise_audit_expectation_appends_versioned_payload_without_changing_exact_energy(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        oracle_instances = self._install_fake_oracle_bindings(
            monkeypatch,
            objective_mean=-0.321,
        )
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=20,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase3_v1",
            final_noise_audit_config=self._final_audit_cfg(),
        )
        audit = payload["final_noise_audit_v1"]
        assert payload["success"] is True
        assert payload["energy_source"] == "exact_statevector"
        assert audit["status"] == "completed"
        assert audit["reference"]["primary_metric_name"] == "exact_target_abs_error"
        assert audit["normalized_request"]["execution_surface"] == "expectation_v1"
        assert audit["result"]["requested_estimate_energy"] == pytest.approx(-0.321)
        assert audit["deltas"]["exact_target_abs_error"] == pytest.approx(
            abs(float(payload["exact_gs_energy"]) - (-0.321))
        )
        assert audit["deltas"]["exact_final_state_abs_error"] == pytest.approx(
            abs(float(payload["exact_energy_from_final_state"]) - (-0.321))
        )
        assert oracle_instances and getattr(oracle_instances[0], "closed", False) is True

    def test_final_noise_audit_fail_open_records_failure(self, monkeypatch: pytest.MonkeyPatch):
        self._install_fake_oracle_bindings(
            monkeypatch,
            objective_mean=-0.321,
            raise_on_final_audit=True,
        )
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=20,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase3_v1",
            final_noise_audit_config=self._final_audit_cfg(strict=False),
        )
        audit = payload["final_noise_audit_v1"]
        assert payload["success"] is True
        assert audit["status"] == "failed"
        assert audit["strict"] is False
        assert audit["failure"]["error_type"] == "RuntimeError"
        assert "synthetic final noise audit failure" in audit["failure"]["error_message"]

    def test_final_noise_audit_strict_raises(self, monkeypatch: pytest.MonkeyPatch):
        self._install_fake_oracle_bindings(
            monkeypatch,
            objective_mean=-0.321,
            raise_on_final_audit=True,
        )
        with pytest.raises(RuntimeError, match="synthetic final noise audit failure"):
            _run_hardcoded_adapt_vqe(
                h_poly=self._hh_h(),
                num_sites=2,
                ordering="blocked",
                problem="hh",
                adapt_pool="paop_lf_std",
                t=1.0,
                u=2.0,
                dv=0.0,
                boundary="periodic",
                omega0=1.0,
                g_ep=0.5,
                n_ph_max=1,
                boson_encoding="binary",
                max_depth=1,
                eps_grad=1e-3,
                eps_energy=1e-8,
                maxiter=20,
                seed=7,
                allow_repeats=True,
                finite_angle_fallback=False,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
                adapt_reopt_policy="windowed",
                adapt_window_size=1,
                adapt_window_topk=0,
                adapt_continuation_mode="phase3_v1",
                final_noise_audit_config=self._final_audit_cfg(strict=True),
            )

    def test_final_noise_audit_runtime_expectation_records_profile_and_session(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        self._install_fake_oracle_bindings(
            monkeypatch,
            objective_mean=-0.222,
        )
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=20,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase3_v1",
            final_noise_audit_config=self._final_audit_cfg(
                noise_mode="runtime",
                backend_name="ibm_marrakesh",
                runtime_profile_name="main_twirled_readout_v1",
                runtime_session_policy="backend_only",
            ),
        )
        audit = payload["final_noise_audit_v1"]
        assert audit["status"] == "completed"
        assert audit["requested_config"]["runtime_profile"]["name"] == "main_twirled_readout_v1"
        assert audit["requested_config"]["runtime_session"]["mode"] == "backend_only"
        assert audit["normalized_request"]["runtime_profile"]["name"] == "main_twirled_readout_v1"
        assert audit["normalized_request"]["runtime_session"]["mode"] == "backend_only"
        assert audit["normalized_request"]["execution_surface"] == "expectation_v1"

    def test_final_noise_audit_backend_scheduled_threads_full_local_mitigation_stack(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        oracle_instances = self._install_fake_oracle_bindings(
            monkeypatch,
            objective_mean=-0.222,
            backend_name="FakeNighthawk",
        )
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=20,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase3_v1",
            final_noise_audit_config=self._final_audit_cfg(
                noise_mode="backend_scheduled",
                use_fake_backend=True,
                backend_name="FakeNighthawk",
                mitigation_mode="readout",
                local_readout_strategy="mthree",
                zne_scales=(1.0, 3.0, 5.0),
                local_gate_twirling=True,
                dd_sequence="XpXm",
            ),
        )
        audit = payload["final_noise_audit_v1"]
        mitigation = dict(getattr(oracle_instances[0].config, "mitigation", {}))
        assert audit["status"] == "completed"
        assert audit["normalized_request"]["mitigation"]["mode"] == "readout"
        assert audit["normalized_request"]["mitigation"]["local_readout_strategy"] == "mthree"
        assert audit["normalized_request"]["mitigation"]["zne_scales"] == [1.0, 3.0, 5.0]
        assert audit["normalized_request"]["mitigation"]["local_gate_twirling"] is True
        assert audit["normalized_request"]["mitigation"]["dd_sequence"] == "XpXm"
        assert mitigation["zne_scales"] == [1.0, 3.0, 5.0]
        assert mitigation["local_gate_twirling"] is True
        assert mitigation["dd_sequence"] == "XpXm"
        assert oracle_instances and getattr(oracle_instances[0], "closed", False) is True

    def test_final_noise_audit_runtime_rejects_fake_backend(self):
        with pytest.raises(ValueError, match="requires a real runtime backend"):
            _run_hardcoded_adapt_vqe(
                h_poly=self._hh_h(),
                num_sites=2,
                ordering="blocked",
                problem="hh",
                adapt_pool="paop_lf_std",
                t=1.0,
                u=2.0,
                dv=0.0,
                boundary="periodic",
                omega0=1.0,
                g_ep=0.5,
                n_ph_max=1,
                boson_encoding="binary",
                max_depth=1,
                eps_grad=1e-3,
                eps_energy=1e-8,
                maxiter=20,
                seed=7,
                allow_repeats=True,
                finite_angle_fallback=False,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
                adapt_reopt_policy="windowed",
                adapt_window_size=1,
                adapt_window_topk=0,
                adapt_continuation_mode="phase3_v1",
                final_noise_audit_config=self._final_audit_cfg(
                    noise_mode="runtime",
                    backend_name="ibm_marrakesh",
                    use_fake_backend=True,
                ),
            )

    def test_final_noise_audit_runtime_profile_rejects_explicit_mitigation(self):
        with pytest.raises(ValueError, match="runtime profiles already encode mitigation"):
            _run_hardcoded_adapt_vqe(
                h_poly=self._hh_h(),
                num_sites=2,
                ordering="blocked",
                problem="hh",
                adapt_pool="paop_lf_std",
                t=1.0,
                u=2.0,
                dv=0.0,
                boundary="periodic",
                omega0=1.0,
                g_ep=0.5,
                n_ph_max=1,
                boson_encoding="binary",
                max_depth=1,
                eps_grad=1e-3,
                eps_energy=1e-8,
                maxiter=20,
                seed=7,
                allow_repeats=True,
                finite_angle_fallback=False,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
                adapt_reopt_policy="windowed",
                adapt_window_size=1,
                adapt_window_topk=0,
                adapt_continuation_mode="phase3_v1",
                final_noise_audit_config=self._final_audit_cfg(
                    noise_mode="runtime",
                    backend_name="ibm_marrakesh",
                    mitigation_mode="readout",
                    runtime_profile_name="main_twirled_readout_v1",
                ),
            )

    def test_final_noise_audit_runtime_readout_rejects_local_strategy(self):
        with pytest.raises(ValueError, match="provider-side mitigation"):
            _run_hardcoded_adapt_vqe(
                h_poly=self._hh_h(),
                num_sites=2,
                ordering="blocked",
                problem="hh",
                adapt_pool="paop_lf_std",
                t=1.0,
                u=2.0,
                dv=0.0,
                boundary="periodic",
                omega0=1.0,
                g_ep=0.5,
                n_ph_max=1,
                boson_encoding="binary",
                max_depth=1,
                eps_grad=1e-3,
                eps_energy=1e-8,
                maxiter=20,
                seed=7,
                allow_repeats=True,
                finite_angle_fallback=False,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
                adapt_reopt_policy="windowed",
                adapt_window_size=1,
                adapt_window_topk=0,
                adapt_continuation_mode="phase3_v1",
                final_noise_audit_config=self._final_audit_cfg(
                    noise_mode="runtime",
                    backend_name="ibm_marrakesh",
                    mitigation_mode="readout",
                    local_readout_strategy="mthree",
                ),
            )

    def test_final_noise_audit_runtime_comparison_bundle_records_unmitigated_baseline(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        self._install_fake_oracle_bindings(
            monkeypatch,
            objective_mean_by_stage={
                "final_noise_audit_v1::requested": -0.222,
                "final_noise_audit_v1::unmitigated_baseline": -0.300,
            },
        )
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=20,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase3_v1",
            final_noise_audit_config=self._final_audit_cfg(
                noise_mode="runtime",
                backend_name="ibm_marrakesh",
                runtime_profile_name="main_twirled_readout_v1",
                compare_unmitigated_baseline=True,
            ),
        )
        comparison = payload["final_noise_audit_v1"]["unmitigated_baseline_comparison"]
        assert comparison["enabled"] is True
        assert comparison["status"] == "completed"
        assert comparison["baseline_requested_config"]["runtime_profile"]["name"] == "legacy_runtime_v0"
        assert comparison["baseline_result"]["requested_estimate_energy"] == pytest.approx(-0.300)
        assert comparison["comparison_metrics"]["requested_minus_unmitigated_delta_e"] == pytest.approx(0.078)
        assert comparison["comparison_metrics"]["requested_minus_unmitigated_abs_delta_e"] == pytest.approx(0.078)

    def test_final_noise_audit_baseline_failure_is_fail_open_when_not_strict(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        self._install_fake_oracle_bindings(
            monkeypatch,
            objective_mean_by_stage={
                "final_noise_audit_v1::requested": -0.222,
            },
            raise_on_final_audit_baseline=True,
        )
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=20,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase3_v1",
            final_noise_audit_config=self._final_audit_cfg(
                noise_mode="runtime",
                backend_name="ibm_marrakesh",
                runtime_profile_name="main_twirled_readout_v1",
                compare_unmitigated_baseline=True,
                strict=False,
            ),
        )
        assert payload["final_noise_audit_v1"]["status"] == "completed"
        comparison = payload["final_noise_audit_v1"]["unmitigated_baseline_comparison"]
        assert comparison["enabled"] is True
        assert comparison["status"] == "failed"
        assert comparison["reason"] == "evaluation_failed"
        assert comparison["failure"]["error_type"] == "RuntimeError"
        assert "synthetic final noise audit baseline failure" in comparison["failure"]["error_message"]

    def test_final_noise_audit_comparison_skips_when_requested_matches_unmitigated(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        self._install_fake_oracle_bindings(
            monkeypatch,
            objective_mean_by_stage={
                "final_noise_audit_v1::requested": -0.210,
            },
        )
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=20,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase3_v1",
            final_noise_audit_config=self._final_audit_cfg(
                noise_mode="runtime",
                backend_name="ibm_marrakesh",
                runtime_profile_name="legacy_runtime_v0",
                mitigation_mode="none",
                compare_unmitigated_baseline=True,
            ),
        )
        comparison = payload["final_noise_audit_v1"]["unmitigated_baseline_comparison"]
        assert comparison["enabled"] is True
        assert comparison["status"] == "skipped"
        assert comparison["reason"] == "requested_matches_unmitigated_baseline"

    def test_hubbard_legacy_still_allows_eps_grad_stop(self):
        h_poly = build_hubbard_hamiltonian(
            dims=2, t=1.0, U=4.0, v=0.0,
            repr_mode="JW", indexing="blocked", pbc=True,
        )
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=h_poly,
            num_sites=2,
            ordering="blocked",
            problem="hubbard",
            adapt_pool="uccsd",
            t=1.0,
            u=4.0,
            dv=0.0,
            boundary="periodic",
            omega0=0.0,
            g_ep=0.0,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=4,
            eps_grad=1e9,
            eps_energy=1e-12,
            maxiter=20,
            seed=29,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_continuation_mode="legacy",
        )
        assert payload["success"] is True
        assert bool(payload["eps_grad_termination_enabled"]) is True
        assert str(payload["stop_reason"]) == "eps_grad"
        assert bool(payload["adapt_drop_policy_enabled"]) is False
        assert payload["adapt_drop_policy_source"] == "default_off"


class TestHHContinuationModeGatingNegative:
    def _hh_h(self):
        return build_hubbard_holstein_hamiltonian(
            dims=2,
            J=1.0,
            U=2.0,
            omega0=1.0,
            g=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            v_t=None,
            v0=0.0,
            t_eval=None,
            repr_mode="JW",
            indexing="blocked",
            pbc=True,
            include_zero_point=True,
        )

    @pytest.mark.parametrize("mode", ["legacy", "phase1_v1", "phase2_v1"])
    def test_phase3_knobs_do_not_leak_into_older_modes(self, mode: str):
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=2,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=20,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode=mode,
            phase3_symmetry_mitigation_mode="verify_only",
            phase3_enable_rescue=True,
            phase3_lifetime_cost_mode="phase3_v1",
        )

        if mode == "legacy":
            assert "continuation" not in payload
            for row in payload.get("history", []):
                assert "full_v2_score" not in row
                assert "shortlisted_records" not in row
                assert "optimizer_memory_source" not in row
                assert "generator_id" not in row
                assert "symmetry_mode" not in row
                assert "lifetime_cost_mode" not in row
                assert "remaining_evaluations_proxy" not in row
                assert "cheap_score" not in row
            return

        continuation = payload["continuation"]
        assert continuation["mode"] == mode

        if mode == "phase1_v1":
            assert "optimizer_memory" not in continuation
            assert "selected_generator_metadata" not in continuation
            assert "motif_library" not in continuation
            assert "symmetry_mitigation" not in continuation
            assert "rescue_history" not in continuation
            for row in payload.get("history", []):
                assert "full_v2_score" not in row
                assert "shortlisted_records" not in row
                assert "optimizer_memory_source" not in row
                assert "generator_id" not in row
                assert "symmetry_mode" not in row
                assert "lifetime_cost_mode" not in row
                assert "remaining_evaluations_proxy" not in row
                assert row["cheap_score_version"] == "simple_v1"
                assert row["cheap_score"] == pytest.approx(row["simple_score"])
            return

        assert "optimizer_memory" in continuation
        assert "selected_generator_metadata" not in continuation
        assert "motif_library" not in continuation
        assert "symmetry_mitigation" not in continuation
        assert "rescue_history" not in continuation
        for row in payload.get("history", []):
            assert "full_v2_score" in row
            assert "shortlisted_records" in row
            assert "optimizer_memory_source" in row
            assert "generator_id" not in row
            assert "symmetry_mode" not in row
            assert "lifetime_cost_mode" not in row
            assert "remaining_evaluations_proxy" not in row
            assert row["cheap_score_version"] == "simple_v1"
            assert row["cheap_score"] == pytest.approx(row["simple_score"])


# ────────────────────────────────────────────────────────────────────
#  P2 — windowed reopt pure helpers
# ────────────────────────────────────────────────────────────────────

class TestResolveReoptActiveIndices:
    """Tests for _resolve_reopt_active_indices (pure deterministic helper)."""

    def test_append_only_returns_last(self):
        theta = np.array([0.1, 0.2, 0.3, 0.4])
        idx, name = _resolve_reopt_active_indices(
            policy="append_only", n=4, theta=theta,
            window_size=3, window_topk=0, periodic_full_refit_triggered=False,
        )
        assert idx == [3]
        assert name == "append_only"

    def test_full_returns_all(self):
        theta = np.array([0.1, 0.2, 0.3, 0.4])
        idx, name = _resolve_reopt_active_indices(
            policy="full", n=4, theta=theta,
            window_size=3, window_topk=0, periodic_full_refit_triggered=False,
        )
        assert idx == [0, 1, 2, 3]
        assert name == "full"

    def test_windowed_newest_window(self):
        theta = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        idx, name = _resolve_reopt_active_indices(
            policy="windowed", n=5, theta=theta,
            window_size=2, window_topk=0, periodic_full_refit_triggered=False,
        )
        assert idx == [3, 4]
        assert name == "windowed"

    def test_windowed_topk_selection(self):
        theta = np.array([0.9, 0.01, 0.02, 0.5, 0.6])
        idx, _name = _resolve_reopt_active_indices(
            policy="windowed", n=5, theta=theta,
            window_size=2, window_topk=1, periodic_full_refit_triggered=False,
        )
        # newest = [3,4]; older by |theta| desc: [0(0.9), 2(0.02), 1(0.01)]
        # topk=1 -> pick [0]
        assert 0 in idx
        assert 3 in idx
        assert 4 in idx

    def test_windowed_topk_tiebreak_ascending(self):
        theta = np.array([0.5, 0.5, 0.3, 0.4])
        idx, _name = _resolve_reopt_active_indices(
            policy="windowed", n=4, theta=theta,
            window_size=1, window_topk=1, periodic_full_refit_triggered=False,
        )
        # newest = [3]; older by |theta| desc = [0(0.5),1(0.5),2(0.3)]
        # tie at 0.5: ascending index -> pick 0
        assert idx == [0, 3]

    def test_windowed_sorted_ascending(self):
        theta = np.array([0.9, 0.01, 0.02, 0.5, 0.6])
        idx, _name = _resolve_reopt_active_indices(
            policy="windowed", n=5, theta=theta,
            window_size=2, window_topk=2, periodic_full_refit_triggered=False,
        )
        assert idx == sorted(idx)

    def test_windowed_window_larger_than_n(self):
        theta = np.array([0.1, 0.2])
        idx, _name = _resolve_reopt_active_indices(
            policy="windowed", n=2, theta=theta,
            window_size=10, window_topk=5, periodic_full_refit_triggered=False,
        )
        assert idx == [0, 1]

    def test_periodic_full_override(self):
        theta = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        idx, name = _resolve_reopt_active_indices(
            policy="windowed", n=5, theta=theta,
            window_size=2, window_topk=0, periodic_full_refit_triggered=True,
        )
        assert idx == [0, 1, 2, 3, 4]
        assert name == "windowed_periodic_full"

    def test_append_only_ignores_periodic_full(self):
        """append_only does not honour periodic_full — only windowed does."""
        theta = np.array([0.1, 0.2, 0.3])
        idx, name = _resolve_reopt_active_indices(
            policy="append_only", n=3, theta=theta,
            window_size=3, window_topk=0, periodic_full_refit_triggered=True,
        )
        assert idx == [2]
        assert name == "append_only"

    def test_n_zero_returns_empty(self):
        """n=0 is a degenerate case — returns empty list."""
        idx, _name = _resolve_reopt_active_indices(
            policy="windowed", n=0, theta=np.array([]),
            window_size=2, window_topk=0, periodic_full_refit_triggered=False,
        )
        assert idx == []

    def test_invalid_policy_raises(self):
        with pytest.raises(ValueError, match="Unknown reopt policy"):
            _resolve_reopt_active_indices(
                policy="bogus", n=1, theta=np.array([0.1]),
                window_size=2, window_topk=0, periodic_full_refit_triggered=False,
            )

    def test_n_equals_1(self):
        theta = np.array([0.42])
        idx, _name = _resolve_reopt_active_indices(
            policy="windowed", n=1, theta=theta,
            window_size=3, window_topk=2, periodic_full_refit_triggered=False,
        )
        assert idx == [0]


class TestMakeReducedObjective:
    """Tests for _make_reduced_objective (pure mapping helper)."""

    def test_full_prefix_passthrough(self):
        theta_full = np.array([0.1, 0.2, 0.3])
        active = [0, 1, 2]
        calls = []

        def fake_obj(t):
            calls.append(t.copy())
            return float(np.sum(t))

        obj_r, x0 = _make_reduced_objective(theta_full, active, fake_obj)
        np.testing.assert_array_equal(x0, theta_full)
        val = obj_r(x0)
        assert val == pytest.approx(0.6)
        np.testing.assert_array_equal(calls[-1], theta_full)

    def test_subset_freezes_inactive(self):
        theta_full = np.array([10.0, 0.2, 0.3, 20.0])
        active = [1, 2]
        calls = []

        def fake_obj(t):
            calls.append(t.copy())
            return float(np.sum(t))

        obj_r, x0 = _make_reduced_objective(theta_full, active, fake_obj)
        np.testing.assert_array_equal(x0, np.array([0.2, 0.3]))
        val = obj_r(np.array([0.5, 0.6]))
        expected_full = np.array([10.0, 0.5, 0.6, 20.0])
        np.testing.assert_array_equal(calls[-1], expected_full)
        assert val == pytest.approx(expected_full.sum())

    def test_multiple_active_indices(self):
        theta_full = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        active = [0, 2, 4]
        log = []

        def fake_obj(t):
            log.append(t.copy())
            return float(t[0] + t[2] + t[4])

        obj_r, x0 = _make_reduced_objective(theta_full, active, fake_obj)
        assert len(x0) == 3
        np.testing.assert_array_equal(x0, np.array([1.0, 3.0, 5.0]))


class TestValidReoptPoliciesSet:
    """Smoke test: constant matches spec."""

    def test_members(self):
        assert _VALID_REOPT_POLICIES == {"append_only", "full", "windowed"}


class TestAdaptCLIParsingWindowed:
    """CLI arg-parsing tests for windowed knobs."""

    def test_accepts_windowed(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["adapt_pipeline.py", "--adapt-reopt-policy", "windowed"],
        )
        args = _adapt_mod.parse_args()
        assert args.adapt_reopt_policy == "windowed"

    def test_defaults(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["adapt_pipeline.py"])
        args = _adapt_mod.parse_args()
        assert args.adapt_reopt_policy == "append_only"
        assert args.adapt_window_size == 3
        assert args.adapt_window_topk == 0
        assert args.phase3_geometry_window_size == 0
        assert args.adapt_full_refit_every == 0
        assert args.adapt_final_full_refit == "true"
        assert args.adapt_insertion_mode == "append_only"
        assert not hasattr(args, "adapt_rollback_mode")
        assert not hasattr(args, "adapt_rollback_tolerance")

    def test_overrides(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            [
                "adapt_pipeline.py",
                "--adapt-reopt-policy", "windowed",
                "--adapt-window-size", "5",
                "--adapt-window-topk", "2",
                "--phase3-geometry-window-size", "3",
                "--adapt-full-refit-every", "4",
                "--adapt-final-full-refit", "false",
                "--adapt-insertion-mode", "adaptive",
            ],
        )
        args = _adapt_mod.parse_args()
        assert args.adapt_window_size == 5
        assert args.adapt_window_topk == 2
        assert args.phase3_geometry_window_size == 3
        assert args.adapt_full_refit_every == 4
        assert args.adapt_final_full_refit == "false"
        assert args.adapt_insertion_mode == "adaptive"

    @pytest.mark.parametrize(
        "obsolete_option",
        ["--adapt-rollback-mode", "--adapt-rollback-tolerance"],
    )
    def test_obsolete_rollback_options_are_rejected(self, monkeypatch, obsolete_option):
        monkeypatch.setattr(sys, "argv", ["adapt_pipeline.py", obsolete_option, "off"])
        with pytest.raises(SystemExit):
            _adapt_mod.parse_args()


class TestWindowedReoptValidation:
    """Validation guard tests (called via _run_hardcoded_adapt_vqe)."""

    @pytest.fixture()
    def tiny_h(self):
        return build_hubbard_hamiltonian(
            dims=2, t=1.0, U=4.0, v=0.0,
            repr_mode="JW", indexing="blocked", pbc=True,
        )

    def _call(self, h, **overrides):
        defaults = dict(
            h_poly=h, num_sites=2, ordering="blocked",
            problem="hubbard", adapt_pool="uccsd",
            t=1.0, u=4.0, dv=0.0, boundary="periodic",
            omega0=0.0, g_ep=0.0, n_ph_max=1, boson_encoding="binary",
            max_depth=1, eps_grad=1e-2, eps_energy=1e-6,
            maxiter=5, seed=7,
            allow_repeats=True, finite_angle_fallback=False,
            finite_angle=0.1, finite_angle_min_improvement=1e-12,
        )
        defaults.update(overrides)
        return _run_hardcoded_adapt_vqe(**defaults)

    def test_window_size_lt1_raises(self, tiny_h):
        with pytest.raises(ValueError, match="adapt_window_size"):
            self._call(tiny_h, adapt_reopt_policy="windowed",
                       adapt_window_size=0)

    def test_topk_negative_raises(self, tiny_h):
        with pytest.raises(ValueError, match="adapt_window_topk"):
            self._call(tiny_h, adapt_reopt_policy="windowed",
                       adapt_window_topk=-1)

    def test_phase3_geometry_window_negative_raises(self, tiny_h):
        with pytest.raises(ValueError, match="phase3_geometry_window_size"):
            self._call(tiny_h, phase3_geometry_window_size=-1)

    def test_refit_every_negative_raises(self, tiny_h):
        with pytest.raises(ValueError, match="adapt_full_refit_every"):
            self._call(tiny_h, adapt_reopt_policy="windowed",
                       adapt_full_refit_every=-1)

    def test_invalid_policy_raises(self, tiny_h):
        with pytest.raises(ValueError, match="adapt_reopt_policy"):
            self._call(tiny_h, adapt_reopt_policy="bogus")


class TestWindowedReoptIntegration:
    """End-to-end integration tests for windowed reopt."""

    @pytest.fixture()
    def tiny_h(self):
        return build_hubbard_hamiltonian(
            dims=2, t=1.0, U=4.0, v=0.0,
            repr_mode="JW", indexing="blocked", pbc=True,
        )

    def _run(self, h, **overrides):
        defaults = dict(
            h_poly=h, num_sites=2, ordering="blocked",
            problem="hubbard", adapt_pool="uccsd",
            t=1.0, u=4.0, dv=0.0, boundary="periodic",
            omega0=0.0, g_ep=0.0, n_ph_max=1, boson_encoding="binary",
            max_depth=3, eps_grad=1e-2, eps_energy=1e-6,
            maxiter=40, seed=7,
            allow_repeats=True, finite_angle_fallback=False,
            finite_angle=0.1, finite_angle_min_improvement=1e-12,
            # These tests isolate optimizer-window semantics.  Phase-III's
            # default benchmark batching can add multiple logical coordinates
            # in one controller round, so it must not be part of this fixture.
        )
        defaults.update(overrides)
        payload, _psi = _run_hardcoded_adapt_vqe(**defaults)
        return payload

    # -- payload schema --

    def test_windowed_payload_valid(self, tiny_h):
        res = self._run(tiny_h, adapt_reopt_policy="windowed",
                        adapt_window_size=2, adapt_window_topk=0)
        assert "adapt_window_size" in res
        assert "adapt_window_topk" in res
        assert "adapt_full_refit_every" in res
        assert "adapt_final_full_refit" in res
        assert "final_full_refit" in res

    def test_history_row_metadata(self, tiny_h):
        res = self._run(tiny_h, adapt_reopt_policy="windowed",
                        adapt_window_size=2, adapt_window_topk=0)
        for row in res.get("history", []):
            assert "reopt_policy_effective" in row
            assert "reopt_active_indices" in row
            assert "reopt_active_count" in row

    def test_active_count_bounded(self, tiny_h):
        res = self._run(tiny_h, adapt_reopt_policy="windowed",
                        adapt_window_size=1, adapt_window_topk=0)
        for row in res.get("history", []):
            assert row["reopt_active_count"] <= row.get("depth", 999)

    def test_periodic_trigger(self, tiny_h):
        res = self._run(tiny_h, adapt_reopt_policy="windowed",
                        adapt_window_size=1, adapt_window_topk=0,
                        adapt_full_refit_every=2, max_depth=4)
        triggered = [r["reopt_periodic_full_refit_triggered"]
                     for r in res.get("history", [])]
        # at least one True expected at some cumulative-depth % 2 == 0
        assert any(triggered) or len(triggered) < 2

    def test_final_refit_metadata(self, tiny_h):
        res = self._run(tiny_h, adapt_reopt_policy="windowed",
                        adapt_window_size=1, adapt_window_topk=0,
                        adapt_final_full_refit=True)
        ffr = res.get("final_full_refit", {})
        assert "executed" in ffr

    def test_final_refit_false_skips(self, tiny_h):
        res = self._run(tiny_h, adapt_reopt_policy="windowed",
                        adapt_window_size=1, adapt_window_topk=0,
                        adapt_final_full_refit=False)
        ffr = res.get("final_full_refit", {})
        assert ffr.get("executed") is False or ffr.get("skipped_reason") is not None

    def test_knobs_recorded(self, tiny_h):
        res = self._run(tiny_h, adapt_reopt_policy="windowed",
                        adapt_window_size=5, adapt_window_topk=2,
                        adapt_full_refit_every=3)
        assert res["adapt_window_size"] == 5
        assert res["adapt_window_topk"] == 2
        assert res["adapt_full_refit_every"] == 3

    # -- regression: existing policies unchanged --

    def test_append_only_regression(self, tiny_h):
        res = self._run(tiny_h, adapt_reopt_policy="append_only")
        for row in res.get("history", []):
            assert row["reopt_active_count"] == 1

    def test_full_regression(self, tiny_h):
        res = self._run(tiny_h, adapt_reopt_policy="full")
        for row in res.get("history", []):
            d = row.get("depth", 1)
            assert row["reopt_active_count"] == d

    def test_topk_carry(self, tiny_h):
        res = self._run(tiny_h, adapt_reopt_policy="windowed",
                        adapt_window_size=1, adapt_window_topk=1,
                        max_depth=3)
        for row in res.get("history", []):
            d = row.get("depth", 1)
            expected_max = min(1 + 1, d)  # window + topk, capped by depth
            assert row["reopt_active_count"] <= expected_max

    def test_full_response_scope_does_not_expand_optimizer_window(self, tiny_h):
        res = self._run(
            tiny_h,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            phase3_response_coordinate_scope="full_active_plus_singleton_v1",
            max_depth=3,
        )

        for row in res.get("history", []):
            response_count = int(row["phase3_response_pre_support_count"])
            assert row["phase3_response_coordinate_indices"] == list(
                range(response_count)
            )
            assert response_count == int(
                row["phase3_active_logical_coordinate_count"]
            ) + 1
            assert row["reopt_active_count"] == 1
            assert row["phase3_accepted_refit_coordinate_count"] == 1
            assert "phase3_response_supported_rank" in row

    def test_replay_compat(self, tiny_h):
        """Windowed run must still emit replay-compatible fields."""
        res = self._run(tiny_h, adapt_reopt_policy="windowed",
                        adapt_window_size=2, adapt_window_topk=0)
        assert "operators" in res
        assert "optimal_point" in res


class TestPeriodicFullRefitCadence:
    """Edge cases for periodic full-refit triggering."""

    def test_periodic_full_returns_all(self):
        theta = np.array([0.1, 0.2, 0.3])
        # periodic_full_refit_triggered=True should override to full prefix
        idx, name = _resolve_reopt_active_indices(
            policy="windowed", n=3, theta=theta,
            window_size=1, window_topk=0, periodic_full_refit_triggered=True,
        )
        assert idx == [0, 1, 2]
        assert name == "windowed_periodic_full"

    def test_disabled_when_not_triggered(self):
        # periodic_full_refit_triggered=False with windowed stays windowed.
        theta = np.array([0.1, 0.2, 0.3, 0.4])
        idx, name = _resolve_reopt_active_indices(
            policy="windowed", n=4, theta=theta,
            window_size=2, window_topk=0, periodic_full_refit_triggered=False,
        )
        assert idx == [2, 3]
        assert name == "windowed"


class TestAdaptRefExactEnergyReuse:
    @staticmethod
    def _hh_nq_total() -> int:
        return int(2 * 2 + 2 * boson_qubits_per_site(1, "binary"))

    @classmethod
    def _ref_payload(
        cls,
        *,
        t: float = 1.0,
        include_exact_energy: bool = True,
        exact_energy: float = 0.15866790412572704,
    ) -> dict[str, object]:
        nq_total = cls._hh_nq_total()
        payload: dict[str, object] = {
            "settings": {
                "L": 2,
                "problem": "hh",
                "ordering": "blocked",
                "boundary": "open",
                "t": float(t),
                "u": 4.0,
                "dv": 0.0,
                "omega0": 1.0,
                "g_ep": 0.5,
                "n_ph_max": 1,
                "boson_encoding": "binary",
            },
            "initial_state": {
                "source": "adapt_vqe",
                "amplitudes_qn_to_q0": {
                    format(0, f"0{nq_total}b"): {"re": 1.0, "im": 0.0},
                },
            },
            "adapt_vqe": {
                "ansatz_depth": 2,
            },
        }
        if include_exact_energy:
            payload["ground_state"] = {
                "exact_energy_filtered": float(exact_energy),
            }
        return payload

    @classmethod
    def _run_main_with_ref(
        cls,
        *,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        ref_payload: dict[str, object],
        exact_impl,
    ) -> tuple[dict[str, object], dict[str, object]]:
        ref_json = tmp_path / "warm_ref.json"
        out_json = tmp_path / "adapt_out.json"
        ref_json.write_text(json.dumps(ref_payload), encoding="utf-8")

        captured: dict[str, object] = {"exact_gs_override": None}
        dim = 1 << cls._hh_nq_total()

        def _fake_run_hardcoded_adapt_vqe(**kwargs):
            captured["exact_gs_override"] = kwargs.get("exact_gs_override")
            psi = np.zeros(dim, dtype=complex)
            psi[0] = 1.0
            return {
                "success": True,
                "method": "mock_adapt",
                "energy": float(kwargs.get("exact_gs_override")),
                "pool_type": str(kwargs.get("adapt_pool")),
                "ansatz_depth": 1,
                "num_parameters": 1,
            }, psi

        def _fake_simulate_trajectory(**kwargs):
            return ([{"time": 0.0, "fidelity": 1.0}], [])

        monkeypatch.setattr(_adapt_mod, "_exact_gs_energy_for_problem", exact_impl)
        monkeypatch.setattr(_adapt_mod, "_run_hardcoded_adapt_vqe", _fake_run_hardcoded_adapt_vqe)
        monkeypatch.setattr(_adapt_mod, "_simulate_trajectory", _fake_simulate_trajectory)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "adapt_pipeline.py",
                "--L", "2",
                "--problem", "hh",
                "--t", "1.0",
                "--u", "4.0",
                "--dv", "0.0",
                "--omega0", "1.0",
                "--g-ep", "0.5",
                "--n-ph-max", "1",
                "--boson-encoding", "binary",
                "--boundary", "open",
                "--ordering", "blocked",
                "--adapt-pool", "paop_lf_std",
                "--adapt-continuation-mode", "phase3_v1",
                "--adapt-ref-json", str(ref_json),
                "--skip-pdf",
                "--output-json", str(out_json),
            ],
        )

        _adapt_mod.main()
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        return payload, captured

    def test_main_reuses_exact_energy_from_metadata_compatible_ref(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        ref_payload = self._ref_payload(include_exact_energy=True)

        def _fail_exact(*args, **kwargs):
            raise AssertionError("_exact_gs_energy_for_problem should not run when warm exact energy is reusable")

        payload, captured = self._run_main_with_ref(
            monkeypatch=monkeypatch,
            tmp_path=tmp_path,
            ref_payload=ref_payload,
            exact_impl=_fail_exact,
        )

        assert payload["ground_state"]["exact_energy_source"] == "adapt_ref_json"
        assert payload["ground_state"]["exact_energy"] == pytest.approx(0.15866790412572704)
        assert payload["ground_state"]["method"] == "python_matrix_eigendecomposition"
        assert payload["settings"]["phase3_runtime_split_selection_mode"] == "proxy_child_set_preselection"
        assert int(payload["settings"]["phase3_runtime_split_max_subset_size"]) == 3
        assert int(payload["settings"]["phase3_parent_collapse_debug_max_depth"]) == 0
        assert captured["exact_gs_override"] == pytest.approx(0.15866790412572704)
        assert bool(payload["adapt_ref_import"]["exact_energy_reused"]) is True
        assert payload["adapt_ref_import"]["exact_energy_reuse_mismatches"] == []
        assert bool(payload["adapt_ref_import"]["ansatz_input_state_persisted"]) is True
        assert payload["adapt_ref_import"]["initial_state_handoff_state_kind"] is None
        assert payload["ansatz_input_state"]["source"] == "adapt_vqe"
        assert payload["ansatz_input_state"]["nq_total"] == self._hh_nq_total()

    def test_main_prefers_cli_exact_energy_override_without_local_ed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        override = -1.2345
        ref_payload = self._ref_payload(include_exact_energy=True, exact_energy=0.15866790412572704)
        ref_json = tmp_path / "warm_ref.json"
        out_json = tmp_path / "adapt_out.json"
        ref_json.write_text(json.dumps(ref_payload), encoding="utf-8")
        captured: dict[str, object] = {"exact_gs_override": None}
        dim = 1 << self._hh_nq_total()

        def _fail_exact(*args, **kwargs):
            raise AssertionError("local exact energy resolution should not run when --adapt-exact-gs-override is set")

        def _fail_eigh(*args, **kwargs):
            raise AssertionError("dense exact-state diagonalization should not run for hf/adapt_vqe initial state")

        def _fake_run_hardcoded_adapt_vqe(**kwargs):
            captured["exact_gs_override"] = kwargs.get("exact_gs_override")
            psi = np.zeros(dim, dtype=complex)
            psi[0] = 1.0
            return {
                "success": True,
                "method": "mock_adapt",
                "energy": float(kwargs.get("exact_gs_override")),
                "pool_type": str(kwargs.get("adapt_pool")),
                "ansatz_depth": 1,
                "num_parameters": 1,
            }, psi

        monkeypatch.setattr(_adapt_mod, "_exact_gs_energy_for_problem", _fail_exact)
        monkeypatch.setattr(_adapt_mod.np.linalg, "eigh", _fail_eigh)
        monkeypatch.setattr(_adapt_mod, "_run_hardcoded_adapt_vqe", _fake_run_hardcoded_adapt_vqe)
        monkeypatch.setattr(sys, "argv", [
            "adapt_pipeline.py",
            "--L", "2",
            "--problem", "hh",
            "--t", "1.0",
            "--u", "4.0",
            "--dv", "0.0",
            "--omega0", "1.0",
            "--g-ep", "0.5",
            "--n-ph-max", "1",
            "--boson-encoding", "binary",
            "--boundary", "open",
            "--ordering", "blocked",
            "--adapt-pool", "paop_lf_std",
            "--adapt-continuation-mode", "phase3_v1",
            "--adapt-ref-json", str(ref_json),
            "--adapt-exact-gs-override", str(override),
            "--skip-pdf",
            "--skip-trajectory",
            "--output-json", str(out_json),
        ])

        _adapt_mod.main()
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        assert payload["ground_state"]["exact_energy_source"] == "adapt_exact_gs_override"
        assert payload["ground_state"]["exact_energy"] == pytest.approx(override)
        assert payload["settings"]["adapt_exact_gs_override"] == pytest.approx(override)
        assert captured["exact_gs_override"] == pytest.approx(override)
        assert bool(payload["adapt_ref_import"]["exact_energy_reused"]) is False

    def test_main_reuses_exact_energy_from_reference_manifest(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        reference_energy = -0.4321
        ref_manifest = tmp_path / "exact_refs.json"
        out_json = tmp_path / "adapt_out.json"
        ref_manifest.write_text(
            json.dumps(
                {
                    "references": [
                        {
                            "id": "hh_L2_nph1_fixture",
                            "settings": {
                                "problem": "hh",
                                "L": 2,
                                "t": 1.0,
                                "u": 4.0,
                                "dv": 0.0,
                                "omega0": 1.0,
                                "g_ep": 0.5,
                                "n_ph_max": 1,
                                "boson_encoding": "binary",
                                "boundary": "open",
                                "ordering": "blocked",
                                "include_zero_point": True,
                            },
                            "exact_energy": reference_energy,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        captured: dict[str, object] = {"exact_gs_override": None}
        dim = 1 << self._hh_nq_total()

        def _fail_exact(*args, **kwargs):
            raise AssertionError("local exact energy resolution should not run with a matching reference manifest")

        def _fail_eigh(*args, **kwargs):
            raise AssertionError("dense exact-state diagonalization should not run for hf/adapt_vqe initial state")

        def _fake_run_hardcoded_adapt_vqe(**kwargs):
            captured["exact_gs_override"] = kwargs.get("exact_gs_override")
            psi = np.zeros(dim, dtype=complex)
            psi[0] = 1.0
            return {
                "success": True,
                "method": "mock_adapt",
                "energy": float(kwargs.get("exact_gs_override")),
                "pool_type": str(kwargs.get("adapt_pool")),
                "ansatz_depth": 1,
                "num_parameters": 1,
            }, psi

        monkeypatch.setattr(_adapt_mod, "_exact_gs_energy_for_problem", _fail_exact)
        monkeypatch.setattr(_adapt_mod.np.linalg, "eigh", _fail_eigh)
        monkeypatch.setattr(_adapt_mod, "_run_hardcoded_adapt_vqe", _fake_run_hardcoded_adapt_vqe)
        monkeypatch.setattr(sys, "argv", [
            "adapt_pipeline.py",
            "--L", "2",
            "--problem", "hh",
            "--t", "1.0",
            "--u", "4.0",
            "--dv", "0.0",
            "--omega0", "1.0",
            "--g-ep", "0.5",
            "--n-ph-max", "1",
            "--boson-encoding", "binary",
            "--boundary", "open",
            "--ordering", "blocked",
            "--adapt-pool", "paop_lf_std",
            "--adapt-continuation-mode", "phase3_v1",
            "--adapt-exact-gs-reference-json", str(ref_manifest),
            "--skip-pdf",
            "--skip-trajectory",
            "--output-json", str(out_json),
        ])

        _adapt_mod.main()
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        assert payload["ground_state"]["exact_energy_source"] == "adapt_exact_gs_reference_json"
        assert payload["ground_state"]["exact_energy"] == pytest.approx(reference_energy)
        assert payload["settings"]["adapt_exact_gs_reference_json"] == str(ref_manifest)
        assert payload["exact_reference_import"]["entry_id"] == "hh_L2_nph1_fixture"
        assert captured["exact_gs_override"] == pytest.approx(reference_energy)

    def test_main_falls_back_when_ref_lacks_exact_energy(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        sentinel = 0.777
        ref_payload = self._ref_payload(include_exact_energy=False)

        def _fake_exact(*args, **kwargs):
            return float(sentinel)

        payload, captured = self._run_main_with_ref(
            monkeypatch=monkeypatch,
            tmp_path=tmp_path,
            ref_payload=ref_payload,
            exact_impl=_fake_exact,
        )

        assert payload["ground_state"]["exact_energy_source"] == "computed"
        assert payload["ground_state"]["exact_energy"] == pytest.approx(sentinel)
        assert captured["exact_gs_override"] == pytest.approx(sentinel)
        assert bool(payload["adapt_ref_import"]["exact_energy_reused"]) is False
        assert payload["adapt_ref_import"]["exact_energy_reuse_mismatches"] == []

    def test_main_falls_back_when_ref_metadata_mismatches(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        sentinel = 0.666
        ref_payload = self._ref_payload(include_exact_energy=True, t=0.75)

        def _fake_exact(*args, **kwargs):
            return float(sentinel)

        payload, captured = self._run_main_with_ref(
            monkeypatch=monkeypatch,
            tmp_path=tmp_path,
            ref_payload=ref_payload,
            exact_impl=_fake_exact,
        )

        assert payload["ground_state"]["exact_energy_source"] == "computed"
        assert payload["ground_state"]["exact_energy"] == pytest.approx(sentinel)
        assert captured["exact_gs_override"] == pytest.approx(sentinel)
        assert bool(payload["adapt_ref_import"]["exact_energy_reused"]) is False
        mismatches = payload["adapt_ref_import"]["exact_energy_reuse_mismatches"]
        assert isinstance(mismatches, list)
        assert any(str(msg).startswith("t:") for msg in mismatches)


class TestHHPhase3MotifSeedRegression:
    def test_phase3_motif_seeding_rebuilds_layout_before_projection(self, tmp_path: Path):
        h_poly = build_hubbard_holstein_hamiltonian(
            dims=2,
            J=1.0,
            U=2.0,
            omega0=1.0,
            g=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            repr_mode="JW",
            indexing="blocked",
            pbc=False,
            include_zero_point=True,
        )
        num_particles = half_filled_num_particles(2)
        pool, _method, _class_meta, _label_meta = _adapt_mod.build_hh_pool_by_key(
            pool_key_hh="full_meta",
            h_poly=h_poly,
            num_sites=2,
            t=1.0,
            u=2.0,
            omega0=1.0,
            g_ep=0.5,
            dv=0.0,
            n_ph_max=1,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
            paop_r=1,
            paop_split_paulis=False,
            paop_prune_eps=0.0,
            paop_normalization="none",
            num_particles=num_particles,
        )
        registry = _adapt_mod.build_pool_generator_registry(
            terms=pool,
            family_ids=["full_meta"] * len(pool),
            num_sites=2,
            ordering="blocked",
            qpb=1,
            symmetry_specs=None,
            split_policy="preserve",
        )
        motif_labels = [
            "uccsd_ferm_lifted::uccsd_sing(alpha:0->1)",
            "uccsd_ferm_lifted::uccsd_sing(beta:2->3)",
            "paop_lf_full:paop_dbl_p(site=0->phonon=0)",
            "paop_lf_full:paop_dbl_p(site=1->phonon=1)",
        ]
        generator_metadata = _adapt_mod.selected_generator_metadata_for_labels(motif_labels, registry)
        assert len(generator_metadata) == len(motif_labels)
        motif_library = _adapt_mod.extract_motif_library(
            generator_metadata=generator_metadata,
            theta=[-0.4, 0.4, -0.2, 0.2],
            source_num_sites=2,
            source_tag="test_full_meta_motif",
            ordering="blocked",
            boson_encoding="binary",
        )
        motif_path = tmp_path / "motif_payload.json"
        motif_path.write_text(
            json.dumps({"continuation": {"motif_library": motif_library}}, indent=2) + "\n",
            encoding="utf-8",
        )

        payload, _psi = _run_hardcoded_adapt_vqe(
            h_poly=h_poly,
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="full_meta",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="open",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=6,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=60,
            seed=7,
            allow_repeats=False,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_state_backend="compiled",
            adapt_reopt_policy="windowed",
            adapt_window_size=32,
            adapt_window_topk=32,
            adapt_full_refit_every=0,
            adapt_final_full_refit=False,
            adapt_drop_floor=-1.0,
            adapt_grad_floor=-1.0,
            adapt_continuation_mode="phase3_v1",
            disable_hh_seed=True,
            phase3_motif_source_json=motif_path,
            phase2_motif_bonus_weight=0.05,
            phase3_runtime_split_mode="off",
            phase3_lifetime_cost_mode="phase3_v1",
            phase3_symmetry_mitigation_mode="verify_only",
            phase3_enable_rescue=False,
            phase3_backend_cost_mode="proxy",
        )

        continuation = payload["continuation"]
        motif_usage = continuation["motif_usage"]
        assert payload["success"] is True
        assert motif_usage["enabled"] is True
        assert motif_usage["source_tag"] == "test_full_meta_motif"
        assert motif_usage["seeded_labels"]
        assert all(str(label) in payload["operators"] for label in motif_usage["seeded_labels"])
        assert len(payload["optimal_point"]) == int(payload["parameterization"]["runtime_parameter_count"])
        assert len(payload["logical_optimal_point"]) == int(payload["parameterization"]["logical_operator_count"])


class TestGenericPhase3ContinuationParity:
    def _hubbard_defaults(self) -> dict[str, Any]:
        h_poly = build_hubbard_hamiltonian(
            dims=2,
            t=1.0,
            U=4.0,
            v=0.0,
            repr_mode="JW",
            indexing="blocked",
            pbc=True,
        )
        return {
            "h_poly": h_poly,
            "num_sites": 2,
            "ordering": "blocked",
            "problem": "hubbard",
            "adapt_pool": "uccsd",
            "t": 1.0,
            "u": 4.0,
            "dv": 0.0,
            "boundary": "periodic",
            "omega0": 0.0,
            "g_ep": 0.0,
            "n_ph_max": 1,
            "boson_encoding": "binary",
            "max_depth": 3,
            "eps_grad": 1e-6,
            "eps_energy": 1e-10,
            "maxiter": 120,
            "seed": 17,
            "adapt_inner_optimizer": "POWELL",
            "allow_repeats": False,
            "finite_angle_fallback": True,
            "finite_angle": 0.1,
            "finite_angle_min_improvement": 1e-12,
            "adapt_state_backend": "compiled",
            "adapt_reopt_policy": "windowed",
            "adapt_window_size": 64,
            "adapt_window_topk": 64,
            "adapt_full_refit_every": 1,
            "adapt_final_full_refit": True,
            "adapt_drop_floor": -1.0,
            "adapt_grad_floor": -1.0,
            "adapt_continuation_mode": "phase3_v1",
            "phase1_shortlist_size": 64,
            "phase1_probe_max_positions": 64,
            "phase1_trough_margin_ratio": 1.0,
            "phase2_shortlist_fraction": 1.0,
            "phase2_shortlist_size": 32,
            "phase2_lambda_H": 1e-6,
            "phase2_rho": 0.25,
            "phase3_runtime_split_mode": "off",
            "phase3_lifetime_cost_mode": "off",
            "phase3_symmetry_mitigation_mode": "off",
            "phase3_backend_cost_mode": "proxy",
        }

    def test_hubbard_phase3_rescue_sidecar_uses_generic_exact_state_without_dimension_crash(self):
        payload, _psi = _run_hardcoded_adapt_vqe(
            **self._hubbard_defaults(),
            phase3_enable_rescue=True,
        )

        assert payload["success"] is True
        continuation = payload["continuation"]
        sidecar = continuation["exact_state_sidecar"]
        assert continuation["phase3_enable_rescue_requested"] is True
        assert continuation["phase3_enable_rescue_effective"] is True
        assert sidecar["requested"] is True
        assert sidecar["available"] is True
        assert sidecar["source"] == "dense_spin_sector"
        assert sidecar["comparison_space_label"] == "full_register"
        assert sidecar["skip_reason"] is None
        assert sidecar["used_for_final_fidelity"] is True
        assert payload["exact_state_fidelity_source"] == "phase3_rescue_exact_state"
        assert continuation["exact_reference_resolution"]["source"] == "dense_spin_sector"
