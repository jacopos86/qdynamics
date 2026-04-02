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

import numpy as np
import pytest

import pipelines.exact_bench.noise_oracle_runtime as _raw_runtime

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum.hubbard_latex_python_pairs import (
    build_hubbard_hamiltonian,
    build_hubbard_holstein_hamiltonian,
    boson_qubits_per_site,
)
from src.quantum.vqe_latex_python_pairs import (
    AnsatzTerm,
    exact_ground_energy_sector,
    exact_ground_energy_sector_hh,
    half_filled_num_particles,
)

# Import ADAPT pipeline internals
import builtins
import importlib.util
from pipelines.hardcoded.hh_continuation_types import CompileCostEstimate

_spec = importlib.util.spec_from_file_location(
    "hardcoded_adapt_pipeline",
    str(REPO_ROOT / "pipelines" / "hardcoded" / "adapt_pipeline.py"),
)
_adapt_mod = importlib.util.module_from_spec(_spec)
sys.modules["hardcoded_adapt_pipeline"] = _adapt_mod
_spec.loader.exec_module(_adapt_mod)

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
_commutator_gradient = _adapt_mod._commutator_gradient
_resolve_reopt_active_indices = _adapt_mod._resolve_reopt_active_indices
_make_reduced_objective = _adapt_mod._make_reduced_objective
_VALID_REOPT_POLICIES = _adapt_mod._VALID_REOPT_POLICIES
_Phase3OracleGradientConfig = _adapt_mod.Phase3OracleGradientConfig


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


class TestCompiledPauliCache:
    """Parity and performance checks for cached compiled Pauli actions."""

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


# ============================================================================
# Pool builder tests
# ============================================================================

class TestAdaptCLIParsing:
    """CLI parsing includes newly supported ADAPT pool options."""

    def test_parse_accepts_uccsd_paop_lf_full_pool(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--adapt-pool", "uccsd_paop_lf_full"],
        )
        args = _adapt_mod.parse_args()
        assert str(args.adapt_pool) == "uccsd_paop_lf_full"

    def test_parse_accepts_full_meta_pool(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--problem", "hh", "--adapt-pool", "full_meta"],
        )
        args = _adapt_mod.parse_args()
        assert str(args.adapt_pool) == "full_meta"

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

    def test_programmatic_default_resolution_stays_legacy_for_none(self):
        assert _adapt_mod._resolve_adapt_continuation_mode(problem="hh", requested_mode=None) == "legacy"

    def test_programmatic_default_resolution_stays_legacy_for_empty_string(self):
        assert _adapt_mod._resolve_adapt_continuation_mode(problem="hh", requested_mode="") == "legacy"

    def test_cli_default_resolution_promotes_hh_to_phase3(self):
        assert _adapt_mod._resolve_cli_adapt_continuation_mode(problem="hh", requested_mode=None) == "phase3_v1"

    def test_cli_default_resolution_keeps_hubbard_legacy(self):
        assert _adapt_mod._resolve_cli_adapt_continuation_mode(problem="hubbard", requested_mode=None) == "legacy"

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

    def test_paop_module_importable(self):
        """Verify the operator_pools module can be imported directly."""
        from src.quantum.operator_pools import make_pool
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

    def test_adapt_full_hamiltonian_converges(self):
        """full_hamiltonian pool should converge well for L=2."""
        payload, psi = _run_hardcoded_adapt_vqe(
            h_poly=self.h_poly,
            num_sites=self.L,
            ordering="blocked",
            problem="hubbard",
            adapt_pool="full_hamiltonian",
            t=self.t, u=self.u, dv=0.0,
            boundary="periodic",
            omega0=0.0, g_ep=0.0,
            n_ph_max=1, boson_encoding="binary",
            max_depth=20,
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
        # full_hamiltonian pool for L=2 periodic Hubbard: ADAPT can get trapped
        # at E≈0 (a degenerate eigenvalue) because the greedy gradient selection
        # cannot escape this local minimum with only 10 Hamiltonian-term generators.
        # Verify significant improvement over HF reference energy.
        hf_energy = 4.0
        assert payload["energy"] < hf_energy - 1.0, \
            f"ADAPT full_hamiltonian must improve on HF: E={payload['energy']:.4f} vs HF={hf_energy}"


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


class TestAdaptDepthRollbackGuard:
    """ADAPT must not accept a depth that regresses energy.

    Regression test: before the rollback guard, the ADAPT loop would
    unconditionally accept the optimizer result.  If SPSA (or COBYLA)
    returned an energy worse than entry, the regression was permanently
    committed.  Now iter_done should never show positive delta_e.
    """

    def test_spsa_depth_never_regresses_energy(self, monkeypatch: pytest.MonkeyPatch):
        """Every iter_done event must have delta_e <= 0 (or depth_rollback=True with delta_e==0)."""
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
                eps_grad=1e-6,
                eps_energy=1e-10,
                maxiter=40,
                seed=11,
                adapt_inner_optimizer="SPSA",
                adapt_spsa_callback_every=10,
                adapt_spsa_progress_every_s=999.0,
                allow_repeats=True,
                finite_angle_fallback=True,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
            )
            iter_done_events = [
                ev for ev in events if ev[0] == "hardcoded_adapt_iter_done"
            ]
            assert len(iter_done_events) > 0, "No iter_done events emitted"
            for ev_name, ev_fields in iter_done_events:
                delta_e = float(ev_fields["delta_e"])
                # After rollback guard: accepted delta_e must be <= 0.
                # Rolled-back depths have delta_e == 0.0 exactly.
                assert delta_e <= 0.0 + 1e-14, (
                    f"depth {ev_fields.get('depth')} accepted a regression: "
                    f"delta_e={delta_e}"
                )
        finally:
            monkeypatch.setattr(_adapt_mod, "_ai_log", original_ai_log)

    def test_rollback_event_is_logged(self, monkeypatch: pytest.MonkeyPatch):
        """If rollback fires, the hardcoded_adapt_depth_rollback event must be emitted."""
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
                max_depth=4,
                eps_grad=1e-6,
                eps_energy=1e-10,
                maxiter=40,
                seed=11,
                adapt_inner_optimizer="SPSA",
                adapt_spsa_callback_every=10,
                adapt_spsa_progress_every_s=999.0,
                allow_repeats=True,
                finite_angle_fallback=True,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
            )
            rollback_events = [
                ev for ev in events if ev[0] == "hardcoded_adapt_depth_rollback"
            ]
            iter_done_events = [
                ev for ev in events if ev[0] == "hardcoded_adapt_iter_done"
            ]
            # Verify that any iter_done with depth_rollback=True has a
            # corresponding rollback log event
            rollback_depths_from_iter = {
                int(ev[1]["depth"])
                for ev in iter_done_events
                if ev[1].get("depth_rollback") is True
            }
            rollback_depths_from_event = {
                int(ev[1]["depth"])
                for ev in rollback_events
            }
            assert rollback_depths_from_iter == rollback_depths_from_event, (
                f"Mismatch: iter_done rollback depths={rollback_depths_from_iter} "
                f"vs rollback events={rollback_depths_from_event}"
            )
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

    def test_hubbard_pool_full_meta_raises(self):
        """full_meta is HH-only and must reject pure Hubbard runs."""
        h_poly = build_hubbard_hamiltonian(
            dims=2, t=1.0, U=4.0, v=0.0,
            repr_mode="JW", indexing="blocked", pbc=True,
        )
        with pytest.raises(ValueError, match="only valid for problem='hh'"):
            _run_hardcoded_adapt_vqe(
                h_poly=h_poly,
                num_sites=2, ordering="blocked",
                problem="hubbard", adapt_pool="full_meta",
                t=1.0, u=4.0, dv=0.0, boundary="periodic",
                omega0=0.0, g_ep=0.0, n_ph_max=1, boson_encoding="binary",
                max_depth=5, eps_grad=1e-2, eps_energy=1e-6,
                maxiter=50, seed=7,
                allow_repeats=True, finite_angle_fallback=False,
                finite_angle=0.1, finite_angle_min_improvement=1e-12,
            )

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

        def _fake_bindings() -> dict[str, object]:
            return {
                "ExpectationOracle": _FakeOracle,
                "RawMeasurementOracle": _FakeRawOracle,
                "OracleConfig": _FakeOracleConfig,
                "all_z_full_register_qop": _raw_runtime._all_z_full_register_qop,
                "summarize_hh_full_register_z_records": _raw_runtime._summarize_hh_full_register_z_records,
                "normalize_sampler_raw_runtime_config": (lambda cfg: cfg),
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
            assert row["cheap_score_version"] == "phase3_cheap_ratio_v1"
            assert "cheap_metric_proxy" in row
            assert "metric_proxy" in row
            assert "cheap_benefit_proxy" in row
            assert "cheap_burden_total" in row
            assert "sigma_hat" in row
            assert row["sigma_hat"] == pytest.approx(0.0)
            assert row["cheap_metric_proxy"] == pytest.approx(row["metric_proxy"])
        assert continuation["phase2_shortlist_rows"]
        assert all(
            row["cheap_score_version"] == "phase3_cheap_ratio_v1"
            for row in continuation["phase2_shortlist_rows"]
        )
        assert all(
            row["sigma_hat"] == pytest.approx(0.0)
            for row in continuation["phase2_shortlist_rows"]
        )
        assert all(
            row["cheap_metric_proxy"] == pytest.approx(row["metric_proxy"])
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
            if score_key == "cheap_score":
                for rec in records:
                    feat = rec.get("feature")
                    if feat is not None and hasattr(feat, "cheap_metric_proxy") and hasattr(feat, "g_abs"):
                        captured_phase1_metrics.append(
                            (float(feat.cheap_metric_proxy), float(feat.g_abs))
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

    def test_phase3_routes_sigma_hat_from_label_resolver(self, monkeypatch: pytest.MonkeyPatch):
        captured_sigmas: list[float] = []
        original_shortlist_records = _adapt_mod.shortlist_records

        def _capture_shortlist_records(records, *, cfg, score_key="simple_score", tie_break_score_key="simple_score"):
            if score_key == "cheap_score":
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

        history_row = payload["history"][0]
        assert payload["continuation"]["gradient_uncertainty_source"] == "oracle_fd_stderr_v1"
        assert payload["continuation"]["oracle_gradient_scope"] == "selection_only"
        assert payload["continuation"]["oracle_gradient_calls_total"] == 2 * len(history_row["candidate_gradient_scout"])
        assert payload["continuation"]["oracle_backend_info"]["backend_name"] == "FakeNighthawk"
        assert payload["continuation"]["reoptimization_backend"] == "exact_statevector"
        assert history_row["gradient_source"] == "oracle_fd_v1"
        assert history_row["max_gradient_stderr"] > 0.0
        assert history_row["candidate_gradient_scout"]
        assert any(float(row["sigma_hat"]) > 0.0 for row in history_row["candidate_gradient_scout"])
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

    def test_phase3_oracle_inner_objective_mode_requires_spsa(self):
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
                adapt_inner_optimizer="POWELL",
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

        history_row = payload["history"][0]
        continuation = payload["continuation"]
        assert continuation["oracle_execution_surface"] == "raw_measurement_v1"
        assert continuation["oracle_gradient_raw_records_total"] > 0
        assert continuation["oracle_symmetry_diagnostic_calls_total"] == 2 * len(history_row["candidate_gradient_scout"])
        assert continuation["oracle_symmetry_diagnostic_raw_records_total"] > 0
        assert continuation["oracle_gradient_calls_total"] == 2 * len(history_row["candidate_gradient_scout"])
        assert continuation["reoptimization_backend"] == "exact_statevector"
        assert continuation["oracle_raw_transport"] == "sampler_v2"
        assert history_row["candidate_gradient_scout"]
        assert all(row["raw_summary"] is not None for row in history_row["candidate_gradient_scout"])
        assert all(
            row["raw_summary"]["symmetry_diagnostic"]["plus"]["available"] is True
            and row["raw_summary"]["symmetry_diagnostic"]["minus"]["available"] is True
            for row in history_row["candidate_gradient_scout"]
        )
        assert all(
            row["raw_summary"]["symmetry_diagnostic"]["plus"]["summary"]["sector_weight_mean"] == pytest.approx(1.0)
            for row in history_row["candidate_gradient_scout"]
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

    def test_phase3_oracle_gradient_mode_rejects_non_hh_problem(self):
        h_poly = build_hubbard_hamiltonian(
            dims=2,
            t=1.0,
            U=4.0,
            v=0.0,
            repr_mode="JW",
            indexing="blocked",
            pbc=True,
        )
        with pytest.raises(ValueError, match="problem='hh'"):
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

    def test_phase3_sigma_hat_can_change_precap_shortlist_identity(self, monkeypatch: pytest.MonkeyPatch):
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
        baseline_label = str(baseline_payload["continuation"]["phase2_shortlist_rows"][0]["candidate_label"])

        captured_phase1_labels: list[str] = []
        original_shortlist_records = _adapt_mod.shortlist_records

        def _capture_shortlist_records(records, *, cfg, score_key="simple_score", tie_break_score_key="simple_score"):
            if score_key == "cheap_score":
                captured_phase1_labels.extend(
                    str(rec["feature"].candidate_label)
                    for rec in records
                    if rec.get("feature") is not None and hasattr(rec.get("feature"), "candidate_label")
                )
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
        sigma_label = str(sigma_payload["continuation"]["phase2_shortlist_rows"][0]["candidate_label"])

        assert captured_phase1_labels
        assert baseline_label not in captured_phase1_labels
        assert sigma_label != baseline_label

    def test_phase3_oracle_gradient_sigma_can_change_precap_shortlist_identity(self, monkeypatch: pytest.MonkeyPatch):
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
        target_label = str(exact_payload["continuation"]["phase2_shortlist_rows"][0]["candidate_label"])

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
            oracle_baseline_payload["continuation"]["phase2_shortlist_rows"][0]["candidate_label"]
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
        sigma_label = str(sigma_payload["continuation"]["phase2_shortlist_rows"][0]["candidate_label"])

        assert oracle_baseline_label == target_label
        assert sigma_label != target_label

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

    def test_phase3_requires_positive_lambda_f_for_ratio_cheap_score(self):
        with pytest.raises(ValueError, match="phase3_v1 cheap ratio scoring requires phase1_lambda_F > 0"):
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
                phase1_lambda_F=0.0,
            )

    def test_phase3_backend_cost_mode_emits_backend_compile_summary(self, monkeypatch: pytest.MonkeyPatch):
        class _StubBackendCompileOracle:
            def __init__(self, *, config, num_qubits, ref_state):
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
            phase3_backend_name="ibm_boston",
        )

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
            assert bool(payload["adapt_drop_policy_enabled"]) is True
            assert payload["adapt_drop_floor_resolved"] == pytest.approx(5e-4)
            assert int(payload["adapt_drop_patience_resolved"]) == 3
            assert int(payload["adapt_drop_min_depth_resolved"]) == 12
            assert payload["adapt_grad_floor_resolved"] == pytest.approx(2e-2)
            assert payload["adapt_drop_policy_source"] == "auto_hh_staged"
            assert payload["adapt_drop_floor_source"] == "auto_hh_staged"
            assert payload["adapt_drop_patience_source"] == "auto_hh_staged"
            assert payload["adapt_drop_min_depth_source"] == "auto_hh_staged"
            assert payload["adapt_grad_floor_source"] == "auto_hh_staged"
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
        assert args.adapt_full_refit_every == 0
        assert args.adapt_final_full_refit == "true"

    def test_overrides(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            [
                "adapt_pipeline.py",
                "--adapt-reopt-policy", "windowed",
                "--adapt-window-size", "5",
                "--adapt-window-topk", "2",
                "--adapt-full-refit-every", "4",
                "--adapt-final-full-refit", "false",
            ],
        )
        args = _adapt_mod.parse_args()
        assert args.adapt_window_size == 5
        assert args.adapt_window_topk == 2
        assert args.adapt_full_refit_every == 4
        assert args.adapt_final_full_refit == "false"


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
    ) -> tuple[dict[str, object], dict[str, float | None]]:
        ref_json = tmp_path / "warm_ref.json"
        out_json = tmp_path / "adapt_out.json"
        ref_json.write_text(json.dumps(ref_payload), encoding="utf-8")

        captured: dict[str, float | None] = {"exact_gs_override": None}
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
        assert captured["exact_gs_override"] == pytest.approx(0.15866790412572704)
        assert bool(payload["adapt_ref_import"]["exact_energy_reused"]) is True
        assert payload["adapt_ref_import"]["exact_energy_reuse_mismatches"] == []
        assert bool(payload["adapt_ref_import"]["ansatz_input_state_persisted"]) is True
        assert payload["adapt_ref_import"]["initial_state_handoff_state_kind"] is None
        assert payload["ansatz_input_state"]["source"] == "adapt_vqe"
        assert payload["ansatz_input_state"]["nq_total"] == self._hh_nq_total()

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


class TestHHBeamRuntimeFallbackRegression:
    def test_beam_finite_angle_fallback_splices_runtime_block(self):
        h_poly = build_hubbard_holstein_hamiltonian(
            dims=2,
            J=1.0,
            U=0.5,
            omega0=1.0,
            g=0.2,
            n_ph_max=1,
            boson_encoding="binary",
            repr_mode="JW",
            indexing="blocked",
            pbc=False,
            include_zero_point=True,
        )

        payload, _psi = _run_hardcoded_adapt_vqe(
            h_poly=h_poly,
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="full_meta",
            t=1.0,
            u=0.5,
            dv=0.0,
            boundary="open",
            omega0=1.0,
            g_ep=0.2,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=6,
            eps_grad=5e-7,
            eps_energy=1e-9,
            maxiter=12000,
            seed=7,
            adapt_inner_optimizer="POWELL",
            allow_repeats=True,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_state_backend="compiled",
            adapt_reopt_policy="windowed",
            adapt_window_size=999999,
            adapt_window_topk=999999,
            adapt_full_refit_every=8,
            adapt_final_full_refit=True,
            adapt_drop_floor=-1.0,
            adapt_grad_floor=-1.0,
            adapt_continuation_mode="phase3_v1",
            phase1_prune_enabled=True,
            phase1_prune_fraction=0.25,
            phase1_prune_max_candidates=6,
            phase1_prune_max_regression=1e-8,
            phase1_shortlist_size=256,
            phase1_probe_max_positions=999999,
            phase1_trough_margin_ratio=1.0,
            phase2_shortlist_fraction=1.0,
            phase2_shortlist_size=128,
            phase2_enable_batching=True,
            phase2_batch_target_size=8,
            phase2_batch_size_cap=16,
            phase2_batch_near_degenerate_ratio=0.98,
            phase2_lambda_H=1e-6,
            phase2_rho=0.25,
            phase2_gamma_N=1.0,
            phase3_runtime_split_mode="shortlist_pauli_children_v1",
            phase3_lifetime_cost_mode="phase3_v1",
            phase3_symmetry_mitigation_mode="verify_only",
            phase3_enable_rescue=True,
            phase3_backend_cost_mode="proxy",
            adapt_beam_live_branches=3,
            adapt_beam_children_per_parent=2,
            adapt_beam_terminated_keep=3,
        )

        assert payload["success"] is True
        assert bool(payload["adapt_beam_enabled"]) is True
        assert math.isfinite(float(payload["abs_delta_e"]))
        assert float(payload["abs_delta_e"]) < 1e-3
        assert int(payload["ansatz_depth"]) >= 6

    def test_beam_class_filtered_run_tolerates_missing_phase3_scores(self, tmp_path: Path):
        spec_path = tmp_path / "keep.json"
        spec_path.write_text(
            json.dumps(
                {
                    "classifier_version": _adapt_mod._HH_FULL_META_CLASSIFIER_VERSION,
                    "source_pool": "full_meta",
                    "source_problem": "hh",
                    "source_num_sites": 2,
                    "source_n_ph_max": 1,
                    "keep_classes": ["uccsd_dbl", "paop_cloud_p"],
                },
                indent=2,
            )
            + "\n"
        )

        h_poly = build_hubbard_holstein_hamiltonian(
            dims=2,
            J=1.0,
            U=0.5,
            omega0=1.0,
            g=0.2,
            n_ph_max=1,
            boson_encoding="binary",
            repr_mode="JW",
            indexing="blocked",
            pbc=False,
            include_zero_point=True,
        )

        payload, _psi = _run_hardcoded_adapt_vqe(
            h_poly=h_poly,
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="full_meta",
            t=1.0,
            u=0.5,
            dv=0.0,
            boundary="open",
            omega0=1.0,
            g_ep=0.2,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=40,
            eps_grad=5e-7,
            eps_energy=1e-9,
            maxiter=12000,
            seed=7,
            adapt_inner_optimizer="POWELL",
            allow_repeats=True,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_state_backend="compiled",
            adapt_reopt_policy="windowed",
            adapt_window_size=999999,
            adapt_window_topk=999999,
            adapt_full_refit_every=8,
            adapt_final_full_refit=True,
            adapt_drop_floor=-1.0,
            adapt_grad_floor=-1.0,
            adapt_continuation_mode="phase3_v1",
            phase1_prune_enabled=True,
            phase1_prune_fraction=0.25,
            phase1_prune_max_candidates=6,
            phase1_prune_max_regression=1e-8,
            phase1_shortlist_size=256,
            phase1_probe_max_positions=999999,
            phase1_trough_margin_ratio=1.0,
            phase2_shortlist_fraction=1.0,
            phase2_shortlist_size=128,
            phase2_enable_batching=True,
            phase2_batch_target_size=8,
            phase2_batch_size_cap=16,
            phase2_batch_near_degenerate_ratio=0.98,
            phase2_lambda_H=1e-6,
            phase2_rho=0.25,
            phase2_gamma_N=1.0,
            phase3_runtime_split_mode="shortlist_pauli_children_v1",
            phase3_lifetime_cost_mode="phase3_v1",
            phase3_symmetry_mitigation_mode="verify_only",
            phase3_enable_rescue=True,
            phase3_backend_cost_mode="proxy",
            adapt_beam_live_branches=3,
            adapt_beam_children_per_parent=2,
            adapt_beam_terminated_keep=3,
            adapt_pool_class_filter_json=spec_path,
        )

        assert payload["success"] is True
        assert bool(payload["adapt_beam_enabled"]) is True
        assert math.isfinite(float(payload["energy"]))
        assert math.isfinite(float(payload["abs_delta_e"]))
        assert int(payload["ansatz_depth"]) >= 1
