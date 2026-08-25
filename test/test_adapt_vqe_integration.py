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



# ============================================================================
# Pool builder tests
# ============================================================================

class TestAdaptResolvedProblemContext:



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



class TestAdaptQNSPSAOptIn:
    """QNSPSA is opt-in and uses native ADAPT state fidelity without Qiskit."""



class TestAdaptEnergyStopGate:
    """eps_energy stop must honor min-extra-depth and patience gates."""




# ============================================================================
# ADAPT re-optimization policy tests
# ============================================================================

class TestAdaptReoptPolicyAppendOnly:
    """append_only policy must freeze the theta prefix and only optimize the newest param."""




class TestAdaptReoptPolicyFull:
    """Full (legacy) re-optimization policy must allow all parameters to change."""




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



    def test_oracle_fd_gradient_stderr_combines_stderr_in_quadrature(self):
        stderr = _adapt_mod._oracle_fd_gradient_stderr(
            SimpleNamespace(stderr=0.3),
            {"stderr": 0.4},
            grad_step=0.2,
        )
        assert stderr == pytest.approx(math.sqrt(0.3 ** 2 + 0.4 ** 2) / 0.4)



















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








class TestWindowedReoptIntegration:
    """End-to-end integration tests for windowed reopt."""

    @pytest.fixture()
    def tiny_h(self):
        return build_hubbard_hamiltonian(
            dims=2, t=1.0, U=4.0, v=0.0,
            repr_mode="JW", indexing="blocked", pbc=True,
        )


    # -- payload schema --








    # -- regression: existing policies unchanged --







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

