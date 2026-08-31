#!/usr/bin/env python3
"""Exact/oracle HH adaptive realtime checkpoint controller."""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.hardcoded.hh_continuation_scoring import (
    FullScoreConfig,
    MeasurementCacheAudit,
    Phase1CompileCostOracle,
    Phase2NoveltyOracle,
    shortlist_records,
)
from pipelines.hardcoded.hh_continuation_stage_control import allowed_positions
from pipelines.hardcoded.hh_realtime_checkpoint_types import (
    BaselineGeometrySummary,
    CandidateProbeSummary,
    CheckpointLedgerEntry,
    GeometryValueKey,
    OracleValueKey,
    RealtimeCheckpointConfig,
    dataclass_to_payload,
    make_checkpoint_context,
)
from pipelines.hardcoded.hh_realtime_measurement import (
    ExactCheckpointValueCache,
    OracleCheckpointValueCache,
    build_controller_oracle_tier_configs,
    planning_group_keys_for_term,
    planning_stats_for_term,
    validate_controller_oracle_base_config,
    validate_controller_tiers_mean_only,
)
from pipelines.hardcoded.hh_vqe_from_adapt_family import ReplayScaffoldContext
from src.quantum.ansatz_parameterization import (
    AnsatzParameterLayout,
    GeneratorParameterBlock,
    RotationTermSpec,
    build_parameter_layout,
    runtime_insert_position,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.compiled_polynomial import (
    apply_compiled_polynomial,
    compile_polynomial_action,
)
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, PauliPolynomial


@dataclass(frozen=True)
class RuntimeTermCarrier:
    label: str
    source_label: str
    polynomial: PauliPolynomial
    runtime_specs: tuple[RotationTermSpec, ...]
    repetition_index: int | None = None
    source_logical_index: int | None = None
    candidate_pool_index: int | None = None


@dataclass(frozen=True)
class ControllerRunArtifacts:
    trajectory: list[dict[str, Any]]
    ledger: list[dict[str, Any]]
    summary: dict[str, Any]
    reference: dict[str, Any]


def _carrier_to_term(carrier: RuntimeTermCarrier) -> AnsatzTerm:
    return AnsatzTerm(label=str(carrier.label), polynomial=carrier.polynomial)


def _layout_from_carriers(
    carriers: Sequence[RuntimeTermCarrier],
    *,
    template: AnsatzParameterLayout,
) -> AnsatzParameterLayout:
    runtime_start = 0
    blocks: list[GeneratorParameterBlock] = []
    for logical_index, carrier in enumerate(carriers):
        blocks.append(
            GeneratorParameterBlock(
                candidate_label=str(carrier.label),
                logical_index=int(logical_index),
                runtime_start=int(runtime_start),
                terms=tuple(carrier.runtime_specs),
            )
        )
        runtime_start += int(len(carrier.runtime_specs))
    return AnsatzParameterLayout(
        mode=str(template.mode),
        term_order=str(template.term_order),
        ignore_identity=bool(template.ignore_identity),
        coefficient_tolerance=float(template.coefficient_tolerance),
        blocks=tuple(blocks),
    )


def _build_candidate_carrier(
    term: AnsatzTerm,
    *,
    logical_index: int,
    unique_label: str,
    template_layout: AnsatzParameterLayout,
    candidate_pool_index: int,
) -> RuntimeTermCarrier:
    block_layout = build_parameter_layout(
        [term],
        ignore_identity=bool(template_layout.ignore_identity),
        coefficient_tolerance=float(template_layout.coefficient_tolerance),
        sort_terms=(str(template_layout.term_order).strip().lower() == "sorted"),
    )
    block = block_layout.blocks[0] if block_layout.blocks else GeneratorParameterBlock(
        candidate_label=str(unique_label),
        logical_index=int(logical_index),
        runtime_start=0,
        terms=tuple(),
    )
    return RuntimeTermCarrier(
        label=str(unique_label),
        source_label=str(term.label),
        polynomial=term.polynomial,
        runtime_specs=tuple(block.terms),
        repetition_index=None,
        source_logical_index=None,
        candidate_pool_index=int(candidate_pool_index),
    )


def _build_replay_runtime_terms(
    replay_context: ReplayScaffoldContext,
    *,
    reps: int,
) -> tuple[list[RuntimeTermCarrier], AnsatzParameterLayout]:
    carriers: list[RuntimeTermCarrier] = []
    for rep_idx in range(int(reps)):
        for logical_index, (term, block) in enumerate(
            zip(replay_context.replay_terms, replay_context.base_layout.blocks)
        ):
            carriers.append(
                RuntimeTermCarrier(
                    label=f"{block.candidate_label}__r{rep_idx}",
                    source_label=str(block.candidate_label),
                    polynomial=term.polynomial,
                    runtime_specs=tuple(block.terms),
                    repetition_index=int(rep_idx),
                    source_logical_index=int(logical_index),
                    candidate_pool_index=None,
                )
            )
    layout = _layout_from_carriers(carriers, template=replay_context.base_layout)
    return carriers, layout


def _insert_theta_block(theta: np.ndarray, *, runtime_position: int, width: int) -> np.ndarray:
    arr = np.asarray(theta, dtype=float).reshape(-1)
    return np.concatenate(
        [arr[: int(runtime_position)], np.zeros(int(width), dtype=float), arr[int(runtime_position) :]]
    )


def _overlap_l2(lhs: np.ndarray, rhs: np.ndarray | None) -> float | None:
    if rhs is None:
        return None
    lhs_arr = np.asarray(lhs, dtype=float).reshape(-1)
    rhs_arr = np.asarray(rhs, dtype=float).reshape(-1)
    overlap = min(int(lhs_arr.size), int(rhs_arr.size))
    total = 0.0
    if overlap > 0:
        total += float(np.vdot(lhs_arr[:overlap] - rhs_arr[:overlap], lhs_arr[:overlap] - rhs_arr[:overlap]).real)
    if int(lhs_arr.size) > overlap:
        tail = lhs_arr[overlap:]
        total += float(np.vdot(tail, tail).real)
    if int(rhs_arr.size) > overlap:
        tail = rhs_arr[overlap:]
        total += float(np.vdot(tail, tail).real)
    return float(np.sqrt(max(total, 0.0)))


class RealtimeCheckpointController:
    """Exact/oracle horizon-1 stay-vs-append controller."""

    def __init__(
        self,
        *,
        cfg: RealtimeCheckpointConfig,
        replay_context: ReplayScaffoldContext,
        h_poly: Any,
        hmat: np.ndarray,
        psi_initial: np.ndarray,
        best_theta: Sequence[float],
        allow_repeats: bool,
        t_final: float,
        num_times: int,
        oracle_base_config: Any | None = None,
        wallclock_cap_s: int | None = None,
    ) -> None:
        validate_controller_tiers_mean_only(cfg.tiers)
        self.cfg = cfg
        self.replay_context = replay_context
        self.h_poly = h_poly
        self.hmat = np.asarray(hmat, dtype=complex)
        self.psi_initial = np.asarray(psi_initial, dtype=complex).reshape(-1)
        self._num_qubits = int(round(np.log2(int(self.psi_initial.size))))
        self.allow_repeats = bool(allow_repeats)
        self.times = np.linspace(0.0, float(t_final), int(num_times), dtype=float)
        self._pauli_action_cache: dict[str, Any] = {}
        self._compiled_poly_cache: dict[str, Any] = {}
        self._compiled_h = compile_polynomial_action(
            h_poly,
            tol=1e-12,
            pauli_action_cache=self._pauli_action_cache,
        )
        self._planning_audit = MeasurementCacheAudit(
            nominal_shots_per_group=1,
            plan_version="phase1_qwc_basis_cover_reuse",
            grouping_mode=str(cfg.grouping_mode),
        )
        self._compile_oracle = Phase1CompileCostOracle()
        self._novelty_oracle = Phase2NoveltyOracle()
        self._shortlist_cfg = FullScoreConfig(
            shortlist_fraction=float(cfg.shortlist_fraction),
            shortlist_size=int(cfg.shortlist_size),
        )
        self._append_counter = 0
        self._trajectory: list[dict[str, Any]] = []
        self._ledger: list[dict[str, Any]] = []
        self._previous_theta_dot: np.ndarray | None = None
        self._previous_append_position: int | None = None
        self._run_wallclock_start: float | None = None
        self._wallclock_cap_s = (None if wallclock_cap_s is None else int(wallclock_cap_s))
        self._oracle_base_config = None
        self._oracle_tier_configs: dict[str, Any] = {}
        self._oracle_qop = None
        self._oracle_instances: dict[str, Any] = {}
        self._degraded_checkpoint_count = 0

        mode = str(cfg.mode)
        if mode not in {"exact_v1", "oracle_v1"}:
            raise ValueError(f"Unsupported realtime checkpoint controller mode {mode!r}.")
        if mode == "oracle_v1":
            if oracle_base_config is None:
                raise ValueError("checkpoint controller oracle_v1 requires oracle_base_config.")
            validate_controller_oracle_base_config(oracle_base_config)
            from pipelines.exact_bench.noise_oracle_runtime import pauli_poly_to_sparse_pauli_op

            self._oracle_base_config = oracle_base_config
            self._oracle_tier_configs = build_controller_oracle_tier_configs(
                oracle_base_config,
                cfg.tiers,
            )
            self._oracle_qop = pauli_poly_to_sparse_pauli_op(h_poly)

        if not bool(replay_context.pool_meta.get("candidate_pool_complete", True)):
            raise ValueError(
                "checkpoint controller exact_v1 requires a complete candidate family pool; sparse full_meta replay context is unsupported."
            )

        self.current_terms, self.current_layout = _build_replay_runtime_terms(
            replay_context,
            reps=int(replay_context.cfg.reps),
        )
        self.current_theta = np.asarray(best_theta, dtype=float).reshape(-1)
        self.current_executor = self._build_executor(self.current_terms, self.current_layout)
        if int(self.current_theta.size) != int(self.current_layout.runtime_parameter_count):
            raise ValueError(
                f"Replay best_theta length mismatch: {self.current_theta.size} vs expected {self.current_layout.runtime_parameter_count}."
            )
        psi_reconstructed = self.current_executor.prepare_state(
            self.current_theta,
            np.asarray(replay_context.psi_ref, dtype=complex).reshape(-1),
        )
        reconstruction_error = float(np.linalg.norm(psi_reconstructed - self.psi_initial))
        if reconstruction_error > float(cfg.reconstruction_tol):
            raise ValueError(
                f"Replay reconstruction mismatch: ||psi_reconstructed - psi_initial||={reconstruction_error:.3e} > {cfg.reconstruction_tol:.3e}."
            )

        for carrier in self.current_terms:
            self._planning_audit.commit(planning_group_keys_for_term(_carrier_to_term(carrier)))

        evals, evecs = np.linalg.eigh(self.hmat)
        self._exact_evals = np.asarray(evals, dtype=float)
        self._exact_evecs = np.asarray(evecs, dtype=complex)
        self._exact_coeffs0 = np.asarray(self._exact_evecs.conj().T @ self.psi_initial, dtype=complex)

    def _build_executor(
        self,
        carriers: Sequence[RuntimeTermCarrier],
        layout: AnsatzParameterLayout,
    ) -> CompiledAnsatzExecutor:
        return CompiledAnsatzExecutor(
            [_carrier_to_term(carrier) for carrier in carriers],
            coefficient_tolerance=float(layout.coefficient_tolerance),
            ignore_identity=bool(layout.ignore_identity),
            sort_terms=(str(layout.term_order).strip().lower() == "sorted"),
            pauli_action_cache=self._pauli_action_cache,
            parameterization_mode="per_pauli_term",
            parameterization_layout=layout,
        )

    def _exact_state_at(self, time_value: float) -> np.ndarray:
        phases = np.exp(-1.0j * np.asarray(self._exact_evals, dtype=float) * float(time_value))
        return np.asarray(self._exact_evecs @ (phases * self._exact_coeffs0), dtype=complex)

    def _current_scaffold_labels(self) -> list[str]:
        return [str(carrier.label) for carrier in self.current_terms]

    def _current_source_labels(self) -> set[str]:
        return {str(carrier.source_label) for carrier in self.current_terms}

    def _oracle_wallclock_hit(self) -> bool:
        if self._wallclock_cap_s is None or self._run_wallclock_start is None:
            return False
        return bool((time.perf_counter() - float(self._run_wallclock_start)) >= float(self._wallclock_cap_s))

    def _oracle_estimate_kind(self) -> str | None:
        if self._oracle_base_config is None:
            return None
        return f"oracle_{str(self._oracle_base_config.noise_mode).strip().lower()}"

    def _oracle_for_tier(self, tier_name: str) -> Any:
        if str(self.cfg.mode) != "oracle_v1":
            raise ValueError("Oracle tier access requested while controller mode is not oracle_v1.")
        tier_key = str(tier_name)
        oracle = self._oracle_instances.get(tier_key)
        if oracle is None:
            from pipelines.exact_bench.noise_oracle_runtime import ExpectationOracle

            oracle = ExpectationOracle(self._oracle_tier_configs[tier_key])
            self._oracle_instances[tier_key] = oracle
        return oracle

    def _close_oracles(self) -> None:
        for oracle in self._oracle_instances.values():
            try:
                oracle.close()
            except Exception:
                pass
        self._oracle_instances.clear()

    def _build_runtime_circuit(
        self,
        *,
        layout: AnsatzParameterLayout,
        theta_runtime: np.ndarray | Sequence[float],
    ) -> Any:
        from pipelines.exact_bench.noise_oracle_runtime import build_runtime_layout_circuit

        return build_runtime_layout_circuit(
            layout,
            theta_runtime,
            int(self._num_qubits),
            reference_state=np.asarray(self.replay_context.psi_ref, dtype=complex).reshape(-1),
        )

    def _oracle_energy_estimate(
        self,
        *,
        checkpoint_ctx: Any,
        cache: OracleCheckpointValueCache,
        tier_name: str,
        observable_family: str,
        candidate_label: str | None,
        position_id: int | None,
        layout: AnsatzParameterLayout,
        theta_runtime: np.ndarray | Sequence[float],
    ) -> tuple[dict[str, Any], bool]:
        value_key = OracleValueKey(
            checkpoint_id=str(checkpoint_ctx.checkpoint_id),
            tier_name=str(tier_name),
            observable_family=str(observable_family),
            candidate_label=(None if candidate_label is None else str(candidate_label)),
            position_id=(None if position_id is None else int(position_id)),
        )

        def _compute() -> dict[str, Any]:
            if self._oracle_wallclock_hit():
                raise TimeoutError("checkpoint controller oracle_v1 wallclock cap reached")
            oracle = self._oracle_for_tier(str(tier_name))
            circuit = self._build_runtime_circuit(layout=layout, theta_runtime=theta_runtime)
            est = oracle.evaluate(circuit, self._oracle_qop)
            backend_info = {
                "noise_mode": str(oracle.backend_info.noise_mode),
                "estimator_kind": str(oracle.backend_info.estimator_kind),
                "backend_name": oracle.backend_info.backend_name,
                "using_fake_backend": bool(oracle.backend_info.using_fake_backend),
                "details": dict(oracle.backend_info.details),
            }
            return {
                "mean": float(est.mean),
                "stderr": float(est.stderr),
                "std": float(est.std),
                "stdev": float(est.stdev),
                "n_samples": int(est.n_samples),
                "aggregate": str(est.aggregate),
                "backend_info": backend_info,
            }

        return cache.get_or_compute(value_key, compute=_compute)

    def _confirm_candidates_oracle(
        self,
        *,
        checkpoint_ctx: Any,
        baseline: Mapping[str, Any],
        confirmed: Sequence[Mapping[str, Any]],
        dt: float,
        oracle_cache: OracleCheckpointValueCache,
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str | None]:
        stay_theta = np.asarray(
            self.current_theta + float(dt) * np.asarray(baseline["theta_dot_step"], dtype=float),
            dtype=float,
        ).reshape(-1)
        try:
            stay_estimate, _ = self._oracle_energy_estimate(
                checkpoint_ctx=checkpoint_ctx,
                cache=oracle_cache,
                tier_name="confirm",
                observable_family="stay_step_energy",
                candidate_label=None,
                position_id=None,
                layout=self.current_layout,
                theta_runtime=stay_theta,
            )
        except Exception as exc:
            return [dict(rec) for rec in confirmed], None, str(exc)

        confirmed_oracle: list[dict[str, Any]] = []
        for record in confirmed:
            rec = dict(record)
            try:
                est, _ = self._oracle_energy_estimate(
                    checkpoint_ctx=checkpoint_ctx,
                    cache=oracle_cache,
                    tier_name="confirm",
                    observable_family="candidate_step_energy",
                    candidate_label=str(rec["candidate_label"]),
                    position_id=int(rec["position_id"]),
                    layout=rec["candidate_data"]["aug_layout"],
                    theta_runtime=np.asarray(
                        rec["candidate_data"]["theta_aug"] + float(dt) * np.asarray(rec["theta_dot_aug"], dtype=float),
                        dtype=float,
                    ).reshape(-1),
                )
                improvement_abs = float(stay_estimate["mean"] - est["mean"])
                improvement_ratio = float(improvement_abs / max(abs(float(stay_estimate["mean"])), 1e-14))
                improvement_stderr = float(
                    np.sqrt(float(stay_estimate["stderr"]) ** 2 + float(est["stderr"]) ** 2)
                )
                directional_penalty = 0.0 if rec["candidate_summary"].directional_change_l2 is None else float(rec["candidate_summary"].directional_change_l2)
                adjusted_noisy_improvement = float(
                    improvement_ratio
                    - float(self.cfg.directional_penalty_weight) * directional_penalty
                    - float(self.cfg.measurement_penalty_weight) * float(rec.get("groups_new", 0.0))
                )
                rec["predicted_noisy_energy_mean"] = float(est["mean"])
                rec["predicted_noisy_energy_stderr"] = float(est["stderr"])
                rec["predicted_noisy_improvement_abs"] = float(improvement_abs)
                rec["predicted_noisy_improvement_ratio"] = float(improvement_ratio)
                rec["predicted_noisy_improvement_stderr"] = float(improvement_stderr)
                rec["adjusted_noisy_improvement"] = float(adjusted_noisy_improvement)
                rec["confirm_backend_info"] = dict(est.get("backend_info", {}))
                rec["confirm_error"] = None
                rec["candidate_summary"] = replace(
                    rec["candidate_summary"],
                    decision_metric="oracle_energy_improvement",
                    oracle_estimate_kind=self._oracle_estimate_kind(),
                    predicted_noisy_energy_mean=float(est["mean"]),
                    predicted_noisy_energy_stderr=float(est["stderr"]),
                    predicted_noisy_improvement_abs=float(improvement_abs),
                    predicted_noisy_improvement_ratio=float(improvement_ratio),
                )
            except Exception as exc:
                rec["predicted_noisy_energy_mean"] = None
                rec["predicted_noisy_energy_stderr"] = None
                rec["predicted_noisy_improvement_abs"] = None
                rec["predicted_noisy_improvement_ratio"] = None
                rec["predicted_noisy_improvement_stderr"] = None
                rec["adjusted_noisy_improvement"] = float("-inf")
                rec["confirm_backend_info"] = None
                rec["confirm_error"] = str(exc)
            confirmed_oracle.append(rec)
        return confirmed_oracle, stay_estimate, None

    def _select_action_oracle(
        self,
        *,
        baseline: Mapping[str, Any],
        confirmed: Sequence[Mapping[str, Any]],
    ) -> tuple[str, Mapping[str, Any] | None]:
        if float(baseline["summary"].rho_miss) <= float(self.cfg.miss_threshold):
            return "stay", None
        viable = [rec for rec in confirmed if rec.get("predicted_noisy_improvement_abs") is not None]
        if not viable:
            return "stay", None
        ordered = sorted(
            viable,
            key=lambda rec: (
                -float(rec.get("adjusted_noisy_improvement", float("-inf"))),
                float(rec["candidate_summary"].position_jump_penalty),
                float(rec["candidate_summary"].compile_proxy_total),
                float(rec["candidate_summary"].groups_new),
                int(rec["candidate_summary"].candidate_pool_index),
                int(rec["candidate_summary"].position_id),
            ),
        )
        best = ordered[0]
        if float(best.get("predicted_noisy_improvement_ratio", 0.0)) < float(self.cfg.gain_ratio_threshold):
            return "stay", None
        if float(best.get("predicted_noisy_improvement_abs", 0.0)) < float(self.cfg.append_margin_abs):
            return "stay", None
        return "append_candidate", best

    def _oracle_commit_payload(
        self,
        *,
        checkpoint_ctx: Any,
        oracle_cache: OracleCheckpointValueCache,
        baseline: Mapping[str, Any],
        selected: Mapping[str, Any] | None,
        action_kind: str,
        dt: float,
    ) -> tuple[dict[str, Any], str | None]:
        stay_theta = np.asarray(
            self.current_theta + float(dt) * np.asarray(baseline["theta_dot_step"], dtype=float),
            dtype=float,
        ).reshape(-1)
        out: dict[str, Any] = {
            "stay_noisy_energy_mean": None,
            "stay_noisy_energy_stderr": None,
            "selected_noisy_energy_mean": None,
            "selected_noisy_energy_stderr": None,
            "selected_noisy_improvement_abs": None,
            "selected_noisy_improvement_ratio": None,
        }
        degraded_reason: str | None = None
        try:
            stay_est, _ = self._oracle_energy_estimate(
                checkpoint_ctx=checkpoint_ctx,
                cache=oracle_cache,
                tier_name="commit",
                observable_family="stay_step_energy",
                candidate_label=None,
                position_id=None,
                layout=self.current_layout,
                theta_runtime=stay_theta,
            )
            out["stay_noisy_energy_mean"] = float(stay_est["mean"])
            out["stay_noisy_energy_stderr"] = float(stay_est["stderr"])
            if str(action_kind) == "stay" or selected is None:
                out["selected_noisy_energy_mean"] = float(stay_est["mean"])
                out["selected_noisy_energy_stderr"] = float(stay_est["stderr"])
                out["selected_noisy_improvement_abs"] = 0.0
                out["selected_noisy_improvement_ratio"] = 0.0
                return out, None
            selected_est, _ = self._oracle_energy_estimate(
                checkpoint_ctx=checkpoint_ctx,
                cache=oracle_cache,
                tier_name="commit",
                observable_family="candidate_step_energy",
                candidate_label=str(selected["candidate_label"]),
                position_id=int(selected["position_id"]),
                layout=selected["candidate_data"]["aug_layout"],
                theta_runtime=np.asarray(
                    selected["candidate_data"]["theta_aug"] + float(dt) * np.asarray(selected["theta_dot_aug"], dtype=float),
                    dtype=float,
                ).reshape(-1),
            )
            improvement_abs = float(stay_est["mean"] - selected_est["mean"])
            out["selected_noisy_energy_mean"] = float(selected_est["mean"])
            out["selected_noisy_energy_stderr"] = float(selected_est["stderr"])
            out["selected_noisy_improvement_abs"] = float(improvement_abs)
            out["selected_noisy_improvement_ratio"] = float(
                improvement_abs / max(abs(float(stay_est["mean"])), 1e-14)
            )
        except Exception as exc:
            degraded_reason = str(exc)
        return out, degraded_reason

    def _baseline_geometry(
        self,
        checkpoint_ctx: Any,
        cache: ExactCheckpointValueCache,
    ) -> dict[str, Any]:
        runtime_indices = tuple(range(int(self.current_layout.runtime_parameter_count)))
        psi, raw_tangents = cache.get_or_compute(
            GeometryValueKey(
                checkpoint_id=str(checkpoint_ctx.checkpoint_id),
                observable_family="baseline_runtime_tangents",
                candidate_label=None,
                position_id=None,
                runtime_indices=runtime_indices,
                group_key=None,
                grouping_mode=str(self.cfg.grouping_mode),
            ),
            tier_name="scout",
            compute=lambda: self.current_executor.prepare_state_with_runtime_tangents(
                self.current_theta,
                self.replay_context.psi_ref,
                runtime_indices=runtime_indices,
            ),
        )[0]
        energy, hpsi, variance = cache.get_or_compute(
            GeometryValueKey(
                checkpoint_id=str(checkpoint_ctx.checkpoint_id),
                observable_family="baseline_h_apply",
                candidate_label=None,
                position_id=None,
                runtime_indices=tuple(),
                group_key=None,
                grouping_mode=str(self.cfg.grouping_mode),
            ),
            tier_name="scout",
            compute=lambda: self._energy_hpsi_variance(psi),
        )[0]

        psi_vec = np.asarray(psi, dtype=complex).reshape(-1)
        tangents_matrix: np.ndarray
        if int(self.current_layout.runtime_parameter_count) <= 0:
            tangents_matrix = np.zeros((psi_vec.size, 0), dtype=complex)
        else:
            centered_cols: list[np.ndarray] = []
            for runtime_idx in runtime_indices:
                tangent = np.asarray(raw_tangents[int(runtime_idx)], dtype=complex).reshape(-1)
                centered = tangent - complex(np.vdot(psi_vec, tangent)) * psi_vec
                centered_cols.append(np.asarray(centered, dtype=complex))
            tangents_matrix = np.column_stack(centered_cols) if centered_cols else np.zeros((psi_vec.size, 0), dtype=complex)

        b_bar = -1.0j * (np.asarray(hpsi, dtype=complex).reshape(-1) - float(energy) * psi_vec)
        norm_b_sq = float(max(0.0, np.real(np.vdot(b_bar, b_bar))))
        G = np.asarray(np.real(tangents_matrix.conj().T @ tangents_matrix), dtype=float)
        f = np.asarray(np.real(tangents_matrix.conj().T @ b_bar), dtype=float).reshape(-1)
        G_pinv = np.linalg.pinv(G, rcond=float(self.cfg.pinv_rcond)) if G.size else np.zeros((0, 0), dtype=float)
        K = np.asarray(G + float(self.cfg.regularization_lambda) * np.eye(int(G.shape[0])), dtype=float)
        K_pinv = np.linalg.pinv(K, rcond=float(self.cfg.pinv_rcond)) if K.size else np.zeros((0, 0), dtype=float)
        theta_dot_proj = np.asarray(G_pinv @ f, dtype=float).reshape(-1) if G.size else np.zeros(0, dtype=float)
        theta_dot_step = np.asarray(K_pinv @ f, dtype=float).reshape(-1) if K.size else np.zeros(0, dtype=float)
        epsilon_proj_sq = float(max(0.0, norm_b_sq - float(f @ theta_dot_proj))) if f.size else float(norm_b_sq)
        residual_step = np.asarray(tangents_matrix @ theta_dot_step - b_bar, dtype=complex).reshape(-1)
        epsilon_step_sq = float(max(0.0, np.real(np.vdot(residual_step, residual_step))))
        rho_miss = float(epsilon_proj_sq / max(norm_b_sq, 1e-14))
        rank = int(np.linalg.matrix_rank(K, tol=float(self.cfg.pinv_rcond))) if K.size else 0
        cond = float(np.linalg.cond(K)) if K.size else 1.0
        theta_dot_l2 = float(np.linalg.norm(theta_dot_step))
        baseline_summary = BaselineGeometrySummary(
            energy=float(energy),
            variance=float(variance),
            epsilon_proj_sq=float(epsilon_proj_sq),
            epsilon_step_sq=float(epsilon_step_sq),
            rho_miss=float(rho_miss),
            theta_dot_l2=float(theta_dot_l2),
            matrix_rank=int(rank),
            condition_number=float(cond),
            regularization_lambda=float(self.cfg.regularization_lambda),
            solve_mode="pinv_reg",
            logical_block_count=int(self.current_layout.logical_parameter_count),
            runtime_parameter_count=int(self.current_layout.runtime_parameter_count),
            planning_summary=dict(self._planning_audit.summary()),
            exact_cache_summary=dict(cache.summary()),
        )
        return {
            "psi": psi_vec,
            "energy": float(energy),
            "variance": float(variance),
            "Hpsi": np.asarray(hpsi, dtype=complex).reshape(-1),
            "b_bar": b_bar,
            "norm_b_sq": float(norm_b_sq),
            "T": tangents_matrix,
            "G": G,
            "f": f,
            "K": K,
            "K_pinv": K_pinv,
            "theta_dot_proj": theta_dot_proj,
            "theta_dot_step": theta_dot_step,
            "residual_step": residual_step,
            "summary": baseline_summary,
        }

    def _energy_hpsi_variance(self, psi: np.ndarray) -> tuple[float, np.ndarray, float]:
        psi_vec = np.asarray(psi, dtype=complex).reshape(-1)
        hpsi = apply_compiled_polynomial(psi_vec, self._compiled_h)
        energy = float(np.real(np.vdot(psi_vec, hpsi)))
        variance = float(max(0.0, np.real(np.vdot(hpsi, hpsi)) - energy * energy))
        return float(energy), np.asarray(hpsi, dtype=complex), float(variance)

    def _position_jump_penalty(self, position_id: int) -> float:
        if self._previous_append_position is None:
            return 0.0
        return float(
            abs(int(position_id) - int(self._previous_append_position))
            / max(int(self.current_layout.logical_parameter_count), 1)
        )

    def _candidate_pool_terms(self) -> list[tuple[int, AnsatzTerm]]:
        current_source_labels = self._current_source_labels()
        out: list[tuple[int, AnsatzTerm]] = []
        for pool_index, term in enumerate(self.replay_context.family_pool):
            source_label = str(term.label)
            if (not self.allow_repeats) and source_label in current_source_labels:
                continue
            out.append((int(pool_index), term))
        return out

    def _candidate_positions(self) -> list[int]:
        logical_count = int(self.current_layout.logical_parameter_count)
        active_window = range(max(0, logical_count - int(self.cfg.active_window_size)), logical_count)
        return allowed_positions(
            n_params=int(logical_count),
            append_position=int(logical_count),
            active_window_indices=active_window,
            max_positions=int(self.cfg.max_probe_positions),
        )

    def _candidate_executor_data(
        self,
        *,
        checkpoint_ctx: Any,
        cache: ExactCheckpointValueCache,
        candidate_term: AnsatzTerm,
        candidate_pool_index: int,
        position_id: int,
    ) -> dict[str, Any]:
        unique_label = f"{candidate_term.label}__append{self._append_counter}_p{int(position_id)}"
        candidate_carrier = _build_candidate_carrier(
            candidate_term,
            logical_index=int(position_id),
            unique_label=str(unique_label),
            template_layout=self.current_layout,
            candidate_pool_index=int(candidate_pool_index),
        )
        aug_terms = list(self.current_terms)
        aug_terms.insert(int(position_id), candidate_carrier)
        aug_layout = _layout_from_carriers(aug_terms, template=self.current_layout)
        runtime_pos = int(runtime_insert_position(self.current_layout, int(position_id)))
        theta_aug = _insert_theta_block(
            self.current_theta,
            runtime_position=int(runtime_pos),
            width=int(len(candidate_carrier.runtime_specs)),
        )
        aug_executor = self._build_executor(aug_terms, aug_layout)
        block_indices = tuple(range(int(runtime_pos), int(runtime_pos + len(candidate_carrier.runtime_specs))))
        value, _ = cache.get_or_compute(
            GeometryValueKey(
                checkpoint_id=str(checkpoint_ctx.checkpoint_id),
                observable_family="candidate_insert_tangent_block",
                candidate_label=str(candidate_term.label),
                position_id=int(position_id),
                runtime_indices=block_indices,
                group_key=None,
                grouping_mode=str(self.cfg.grouping_mode),
            ),
            tier_name="scout",
            compute=lambda: aug_executor.prepare_state_with_runtime_tangents(
                theta_aug,
                self.replay_context.psi_ref,
                runtime_indices=block_indices,
            ),
        )
        aug_psi, raw_tangents = value
        return {
            "candidate_carrier": candidate_carrier,
            "aug_terms": aug_terms,
            "aug_layout": aug_layout,
            "aug_executor": aug_executor,
            "theta_aug": theta_aug,
            "runtime_insert_position": int(runtime_pos),
            "runtime_block_indices": [int(x) for x in block_indices],
            "aug_psi": np.asarray(aug_psi, dtype=complex).reshape(-1),
            "raw_tangents": {int(k): np.asarray(v, dtype=complex).reshape(-1) for k, v in raw_tangents.items()},
        }

    def _scout_candidates(
        self,
        *,
        checkpoint_ctx: Any,
        cache: ExactCheckpointValueCache,
        baseline: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        current_terms_window = self.current_terms[
            max(0, len(self.current_terms) - int(self.cfg.active_window_size)) :
        ]
        for candidate_pool_index, candidate_term in self._candidate_pool_terms():
            for position_id in self._candidate_positions():
                candidate_data = self._candidate_executor_data(
                    checkpoint_ctx=checkpoint_ctx,
                    cache=cache,
                    candidate_term=candidate_term,
                    candidate_pool_index=int(candidate_pool_index),
                    position_id=int(position_id),
                )
                centered_cols: list[np.ndarray] = []
                for runtime_idx in candidate_data["runtime_block_indices"]:
                    tangent = np.asarray(candidate_data["raw_tangents"][int(runtime_idx)], dtype=complex)
                    centered = tangent - complex(np.vdot(baseline["psi"], tangent)) * np.asarray(baseline["psi"], dtype=complex)
                    centered_cols.append(np.asarray(centered, dtype=complex))
                u_block = np.column_stack(centered_cols) if centered_cols else np.zeros((baseline["psi"].size, 0), dtype=complex)
                residual_overlap_vec = np.asarray(np.real(u_block.conj().T @ baseline["residual_step"]), dtype=float).reshape(-1)
                residual_overlap_l2 = float(np.linalg.norm(residual_overlap_vec))
                planning_stats = planning_stats_for_term(candidate_term, self._planning_audit)
                compile_est = self._compile_oracle.estimate(
                    candidate_term_count=max(1, len(candidate_data["runtime_block_indices"])),
                    position_id=int(position_id),
                    append_position=int(self.current_layout.logical_parameter_count),
                    refit_active_count=max(0, int(self.current_layout.logical_parameter_count) - int(position_id)),
                    candidate_term=candidate_term,
                )
                novelty = None
                if current_terms_window:
                    try:
                        novelty_info = self._novelty_oracle.estimate(
                            psi_state=np.asarray(baseline["psi"], dtype=complex),
                            candidate_label=str(candidate_term.label),
                            candidate_term=candidate_term,
                            window_terms=[_carrier_to_term(term) for term in current_terms_window],
                            window_labels=[str(term.label) for term in current_terms_window],
                            compiled_cache=self._compiled_poly_cache,
                            pauli_action_cache=self._pauli_action_cache,
                            novelty_eps=1e-6,
                        )
                        novelty = float(novelty_info.get("novelty", 0.0))
                    except Exception:
                        novelty = None
                position_jump_penalty = self._position_jump_penalty(int(position_id))
                scout_score = float(
                    residual_overlap_l2
                    - float(self.cfg.compile_penalty_weight) * float(compile_est.proxy_total)
                    - float(self.cfg.measurement_penalty_weight) * float(planning_stats.groups_new)
                    - float(self.cfg.directional_penalty_weight) * float(position_jump_penalty)
                )
                records.append(
                    {
                        "candidate_label": str(candidate_term.label),
                        "candidate_pool_index": int(candidate_pool_index),
                        "position_id": int(position_id),
                        "runtime_insert_position": int(candidate_data["runtime_insert_position"]),
                        "runtime_block_indices": list(candidate_data["runtime_block_indices"]),
                        "residual_overlap_l2": float(residual_overlap_l2),
                        "compile_proxy_total": float(compile_est.proxy_total),
                        "groups_new": float(planning_stats.groups_new),
                        "novelty": novelty,
                        "position_jump_penalty": float(position_jump_penalty),
                        "simple_score": float(scout_score),
                        "candidate_data": candidate_data,
                        "candidate_term": candidate_term,
                    }
                )
        return shortlist_records(records, cfg=self._shortlist_cfg, score_key="simple_score")

    def _confirm_candidates(
        self,
        *,
        checkpoint_ctx: Any,
        cache: ExactCheckpointValueCache,
        baseline: Mapping[str, Any],
        shortlist: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        confirmed: list[dict[str, Any]] = []
        T = np.asarray(baseline["T"], dtype=complex)
        b_bar = np.asarray(baseline["b_bar"], dtype=complex)
        K_pinv = np.asarray(baseline["K_pinv"], dtype=float)
        theta_dot_step = np.asarray(baseline["theta_dot_step"], dtype=float)
        norm_b_sq = float(baseline["norm_b_sq"])
        for record in shortlist:
            candidate_data = self._candidate_executor_data(
                checkpoint_ctx=checkpoint_ctx,
                cache=cache,
                candidate_term=record["candidate_term"],
                candidate_pool_index=int(record["candidate_pool_index"]),
                position_id=int(record["position_id"]),
            )
            candidate_tangents = [
                np.asarray(candidate_data["raw_tangents"][idx], dtype=complex)
                - complex(np.vdot(baseline["psi"], candidate_data["raw_tangents"][idx])) * np.asarray(baseline["psi"], dtype=complex)
                for idx in candidate_data["runtime_block_indices"]
            ]
            U = np.column_stack(candidate_tangents) if candidate_tangents else np.zeros((baseline["psi"].size, 0), dtype=complex)
            B = np.asarray(np.real(T.conj().T @ U), dtype=float)
            C = np.asarray(np.real(U.conj().T @ U), dtype=float)
            q = np.asarray(np.real(U.conj().T @ b_bar), dtype=float).reshape(-1)
            S = np.asarray(
                C
                + float(self.cfg.candidate_regularization_lambda) * np.eye(int(C.shape[0]))
                - B.T @ K_pinv @ B,
                dtype=float,
            )
            S_pinv = np.linalg.pinv(S, rcond=float(self.cfg.pinv_rcond)) if S.size else np.zeros((0, 0), dtype=float)
            w = np.asarray(q - B.T @ theta_dot_step, dtype=float).reshape(-1)
            eta_dot = np.asarray(S_pinv @ w, dtype=float).reshape(-1) if S.size else np.zeros(0, dtype=float)
            gain_exact = float(max(0.0, float(w @ eta_dot))) if w.size else 0.0
            gain_ratio = float(gain_exact / max(norm_b_sq, 1e-14))
            theta_dot_aug_existing = np.asarray(theta_dot_step - K_pinv @ B @ eta_dot, dtype=float).reshape(-1)
            runtime_pos = int(candidate_data["runtime_insert_position"])
            theta_dot_aug = np.concatenate(
                [
                    theta_dot_aug_existing[:runtime_pos],
                    eta_dot,
                    theta_dot_aug_existing[runtime_pos:],
                ]
            )
            directional_change_l2 = _overlap_l2(theta_dot_aug, self._previous_theta_dot)
            candidate_summary = CandidateProbeSummary(
                candidate_label=str(record["candidate_label"]),
                candidate_pool_index=int(record["candidate_pool_index"]),
                position_id=int(record["position_id"]),
                runtime_insert_position=int(runtime_pos),
                runtime_block_indices=list(candidate_data["runtime_block_indices"]),
                residual_overlap_l2=float(record["residual_overlap_l2"]),
                gain_exact=float(gain_exact),
                gain_ratio=float(gain_ratio),
                compile_proxy_total=float(record["compile_proxy_total"]),
                groups_new=float(record["groups_new"]),
                novelty=(None if record.get("novelty") is None else float(record["novelty"])),
                position_jump_penalty=float(record["position_jump_penalty"]),
                directional_change_l2=(None if directional_change_l2 is None else float(directional_change_l2)),
                tier_reached="confirm",
                admissible=True,
                rejection_reason=None,
                oracle_estimate_kind=self._oracle_estimate_kind(),
            )
            directional_penalty = 0.0 if directional_change_l2 is None else float(directional_change_l2)
            adjusted_gain = float(
                gain_ratio
                - float(self.cfg.directional_penalty_weight) * directional_penalty
                - float(self.cfg.measurement_penalty_weight) * float(record["groups_new"])
            )
            confirmed.append(
                {
                    **dict(record),
                    "gain_exact": float(gain_exact),
                    "gain_ratio": float(gain_ratio),
                    "adjusted_gain": float(adjusted_gain),
                    "theta_dot_aug": theta_dot_aug,
                    "theta_dot_aug_existing": theta_dot_aug_existing,
                    "eta_dot": eta_dot,
                    "candidate_summary": candidate_summary,
                }
            )
        return confirmed

    def _select_action(
        self,
        *,
        baseline: Mapping[str, Any],
        confirmed: Sequence[Mapping[str, Any]],
    ) -> tuple[str, Mapping[str, Any] | None]:
        if float(baseline["summary"].rho_miss) <= float(self.cfg.miss_threshold):
            return "stay", None
        if not confirmed:
            return "stay", None
        ordered = sorted(
            confirmed,
            key=lambda rec: (
                -float(rec["adjusted_gain"]),
                float(rec["candidate_summary"].position_jump_penalty),
                float(rec["candidate_summary"].compile_proxy_total),
                float(rec["candidate_summary"].groups_new),
                int(rec["candidate_summary"].candidate_pool_index),
                int(rec["candidate_summary"].position_id),
            ),
        )
        best = ordered[0]
        if float(best["gain_ratio"]) < float(self.cfg.gain_ratio_threshold):
            return "stay", None
        if float(best["gain_exact"]) < float(self.cfg.append_margin_abs):
            return "stay", None
        return "append_candidate", best

    def _controller_state_payload(self) -> dict[str, Any]:
        return {
            "logical_block_count": int(self.current_layout.logical_parameter_count),
            "runtime_parameter_count": int(self.current_layout.runtime_parameter_count),
            "labels": self._current_scaffold_labels(),
        }

    def run(self) -> ControllerRunArtifacts:
        self._run_wallclock_start = time.perf_counter()
        try:
            for checkpoint_index, time_value in enumerate(self.times):
                time_stop = None
                if int(checkpoint_index) + 1 < int(len(self.times)):
                    time_stop = float(self.times[int(checkpoint_index) + 1])
                psi_current = self.current_executor.prepare_state(self.current_theta, self.replay_context.psi_ref)
                checkpoint_ctx = make_checkpoint_context(
                    checkpoint_index=int(checkpoint_index),
                    time_start=float(time_value),
                    time_stop=(None if time_stop is None else float(time_stop)),
                    scaffold_labels=self._current_scaffold_labels(),
                    theta=self.current_theta,
                    psi=psi_current,
                    logical_count=int(self.current_layout.logical_parameter_count),
                    runtime_count=int(self.current_layout.runtime_parameter_count),
                    resolved_family=str(self.replay_context.family_info.get("resolved", "unknown")),
                    grouping_mode=str(self.cfg.grouping_mode),
                    structure_locked=False,
                )
                cache = ExactCheckpointValueCache(
                    checkpoint_id=str(checkpoint_ctx.checkpoint_id),
                    grouping_mode=str(self.cfg.grouping_mode),
                )
                oracle_cache = OracleCheckpointValueCache(
                    checkpoint_id=str(checkpoint_ctx.checkpoint_id),
                ) if str(self.cfg.mode) == "oracle_v1" else None
                baseline = self._baseline_geometry(checkpoint_ctx, cache)
                degraded_reason: str | None = None
                decision_backend = "exact"
                decision_noise_mode: str | None = None
                oracle_attempted = False
                oracle_decision_used = False
                oracle_estimate_kind = None
                oracle_commit_payload = {
                    "stay_noisy_energy_mean": None,
                    "stay_noisy_energy_stderr": None,
                    "selected_noisy_energy_mean": None,
                    "selected_noisy_energy_stderr": None,
                    "selected_noisy_improvement_abs": None,
                    "selected_noisy_improvement_ratio": None,
                }
                if time_stop is None:
                    shortlist = []
                    confirmed = []
                    action_kind, selected = "stay", None
                else:
                    shortlist = self._scout_candidates(
                        checkpoint_ctx=checkpoint_ctx,
                        cache=cache,
                        baseline=baseline,
                    ) if float(baseline["summary"].rho_miss) > float(self.cfg.miss_threshold) else []
                    confirmed = self._confirm_candidates(
                        checkpoint_ctx=checkpoint_ctx,
                        cache=cache,
                        baseline=baseline,
                        shortlist=shortlist,
                    ) if shortlist else []
                    if str(self.cfg.mode) == "oracle_v1" and shortlist and oracle_cache is not None:
                        oracle_attempted = True
                        confirmed, _, degraded_reason = self._confirm_candidates_oracle(
                            checkpoint_ctx=checkpoint_ctx,
                            baseline=baseline,
                            confirmed=confirmed,
                            dt=float(time_stop - float(time_value)),
                            oracle_cache=oracle_cache,
                        )
                        if degraded_reason is None:
                            decision_backend = "oracle"
                            decision_noise_mode = (
                                None if self._oracle_base_config is None else str(self._oracle_base_config.noise_mode)
                            )
                            oracle_decision_used = True
                            oracle_estimate_kind = self._oracle_estimate_kind()
                            action_kind, selected = self._select_action_oracle(
                                baseline=baseline,
                                confirmed=confirmed,
                            )
                            oracle_commit_payload, commit_degraded_reason = self._oracle_commit_payload(
                                checkpoint_ctx=checkpoint_ctx,
                                oracle_cache=oracle_cache,
                                baseline=baseline,
                                selected=selected,
                                action_kind=str(action_kind),
                                dt=float(time_stop - float(time_value)),
                            )
                            if commit_degraded_reason is not None:
                                degraded_reason = str(commit_degraded_reason)
                        else:
                            action_kind, selected = "stay", None
                    else:
                        action_kind, selected = self._select_action(
                            baseline=baseline,
                            confirmed=confirmed,
                        )

                if degraded_reason is not None:
                    self._degraded_checkpoint_count += 1
                dt = 0.0 if time_stop is None else float(time_stop - float(time_value))
                logical_before = int(self.current_layout.logical_parameter_count)
                runtime_before = int(self.current_layout.runtime_parameter_count)
                selected_groups_new = 0.0
                selected_gain_ratio = 0.0
                selected_candidate_label: str | None = None
                selected_position_id: int | None = None
                if selected is not None:
                    selected_candidate_label = str(selected["candidate_label"])
                    selected_position_id = int(selected["position_id"])
                    selected_groups_new = float(selected.get("groups_new", 0.0))
                    selected_gain_ratio = float(selected.get("gain_ratio", 0.0))
                tier_reached = "scout"
                rate_change_l2 = _overlap_l2(np.asarray(baseline["theta_dot_step"], dtype=float), self._previous_theta_dot)

                psi_exact = self._exact_state_at(float(time_value))
                energy_exact, _, _ = self._energy_hpsi_variance(psi_exact)
                energy_controller = float(baseline["summary"].energy)
                fidelity_exact = float(abs(np.vdot(psi_exact, baseline["psi"])) ** 2)
                abs_energy_total_error = float(abs(float(energy_controller) - float(energy_exact)))

                shortlist_payload = [
                    {
                        "candidate_label": str(item["candidate_label"]),
                        "candidate_pool_index": int(item["candidate_pool_index"]),
                        "position_id": int(item["position_id"]),
                        "runtime_insert_position": int(item["runtime_insert_position"]),
                        "runtime_block_indices": list(item["runtime_block_indices"]),
                        "residual_overlap_l2": float(item["residual_overlap_l2"]),
                        "compile_proxy_total": float(item["compile_proxy_total"]),
                        "groups_new": float(item["groups_new"]),
                        "novelty": (None if item.get("novelty") is None else float(item["novelty"])),
                        "position_jump_penalty": float(item["position_jump_penalty"]),
                        "simple_score": float(item["simple_score"]),
                    }
                    for item in shortlist
                ]
                confirmed_payload = [
                    {
                        "candidate_label": str(rec["candidate_label"]),
                        "candidate_pool_index": int(rec["candidate_pool_index"]),
                        "position_id": int(rec["position_id"]),
                        "gain_exact": float(rec["gain_exact"]),
                        "gain_ratio": float(rec["gain_ratio"]),
                        "adjusted_gain": float(rec["adjusted_gain"]),
                        "adjusted_noisy_improvement": (
                            None if rec.get("adjusted_noisy_improvement") is None or not np.isfinite(rec.get("adjusted_noisy_improvement", float("nan"))) else float(rec.get("adjusted_noisy_improvement"))
                        ),
                        "confirm_error": rec.get("confirm_error", None),
                        "candidate_summary": dataclass_to_payload(rec["candidate_summary"]),
                    }
                    for rec in confirmed
                ]

                self._trajectory.append(
                    {
                        "checkpoint_index": int(checkpoint_index),
                        "time": float(time_value),
                        "action_kind": str(action_kind),
                        "candidate_label": selected_candidate_label,
                        "requested_mode": str(self.cfg.mode),
                        "decision_backend": str(decision_backend),
                        "decision_noise_mode": decision_noise_mode,
                        "oracle_attempted": bool(oracle_attempted),
                        "oracle_decision_used": bool(oracle_decision_used),
                        "oracle_estimate_kind": oracle_estimate_kind,
                        "rho_miss": float(baseline["summary"].rho_miss),
                        "epsilon_proj_sq": float(baseline["summary"].epsilon_proj_sq),
                        "epsilon_step_sq": float(baseline["summary"].epsilon_step_sq),
                        "energy_total_controller": float(energy_controller),
                        "energy_total_exact": float(energy_exact),
                        "abs_energy_total_error": float(abs_energy_total_error),
                        "fidelity_exact": float(fidelity_exact),
                        "logical_block_count": int(logical_before),
                        "runtime_parameter_count": int(runtime_before),
                        "selected_noisy_energy_mean": oracle_commit_payload.get("selected_noisy_energy_mean", None),
                        "selected_noisy_energy_stderr": oracle_commit_payload.get("selected_noisy_energy_stderr", None),
                        "stay_noisy_energy_mean": oracle_commit_payload.get("stay_noisy_energy_mean", None),
                        "stay_noisy_energy_stderr": oracle_commit_payload.get("stay_noisy_energy_stderr", None),
                        "selected_noisy_improvement_abs": oracle_commit_payload.get("selected_noisy_improvement_abs", None),
                        "selected_noisy_improvement_ratio": oracle_commit_payload.get("selected_noisy_improvement_ratio", None),
                        "degraded_reason": degraded_reason,
                        "baseline_geometry": dataclass_to_payload(baseline["summary"]),
                        "shortlist": shortlist_payload,
                        "confirmed": confirmed_payload,
                    }
                )

                if str(action_kind) == "append_candidate" and selected is not None:
                    tier_reached = "commit"
                    candidate_data = dict(selected["candidate_data"])
                    self.current_terms = list(candidate_data["aug_terms"])
                    self.current_layout = candidate_data["aug_layout"]
                    self.current_executor = candidate_data["aug_executor"]
                    self.current_theta = np.asarray(
                        candidate_data["theta_aug"] + float(dt) * np.asarray(selected["theta_dot_aug"], dtype=float),
                        dtype=float,
                    ).reshape(-1)
                    self._append_counter += 1
                    self._previous_append_position = int(selected_position_id)
                    self._planning_audit.commit(planning_group_keys_for_term(selected["candidate_term"]))
                    self._previous_theta_dot = np.asarray(selected["theta_dot_aug"], dtype=float).reshape(-1)
                else:
                    self.current_theta = np.asarray(
                        self.current_theta + float(dt) * np.asarray(baseline["theta_dot_step"], dtype=float),
                        dtype=float,
                    ).reshape(-1)
                    self._previous_theta_dot = np.asarray(baseline["theta_dot_step"], dtype=float).reshape(-1)
                    if shortlist:
                        tier_reached = "confirm"

                ledger_entry = CheckpointLedgerEntry(
                    checkpoint_index=int(checkpoint_index),
                    time=float(time_value),
                    action_kind=str(action_kind),
                    candidate_label=selected_candidate_label,
                    position_id=selected_position_id,
                    rho_miss=float(baseline["summary"].rho_miss),
                    gain_ratio_selected=float(selected_gain_ratio),
                    shortlist_size=int(len(shortlist)),
                    tier_reached=str(tier_reached),
                    logical_block_count_before=int(logical_before),
                    logical_block_count_after=int(self.current_layout.logical_parameter_count),
                    runtime_parameter_count_before=int(runtime_before),
                    runtime_parameter_count_after=int(self.current_layout.runtime_parameter_count),
                    rate_change_l2=(None if rate_change_l2 is None else float(rate_change_l2)),
                    exact_cache_hits=int(cache.summary()["hits"]),
                    exact_cache_misses=int(cache.summary()["misses"]),
                    planning_groups_new_selected=float(selected_groups_new),
                    energy_total_controller=float(energy_controller),
                    energy_total_exact=float(energy_exact),
                    abs_energy_total_error=float(abs_energy_total_error),
                    fidelity_exact=float(fidelity_exact),
                    requested_mode=str(self.cfg.mode),
                    decision_backend=str(decision_backend),
                    decision_noise_mode=decision_noise_mode,
                    oracle_decision_used=bool(oracle_decision_used),
                    oracle_attempted=bool(oracle_attempted),
                    oracle_estimate_kind=oracle_estimate_kind,
                    selected_noisy_energy_mean=(None if oracle_commit_payload.get("selected_noisy_energy_mean", None) is None else float(oracle_commit_payload["selected_noisy_energy_mean"])),
                    selected_noisy_energy_stderr=(None if oracle_commit_payload.get("selected_noisy_energy_stderr", None) is None else float(oracle_commit_payload["selected_noisy_energy_stderr"])),
                    stay_noisy_energy_mean=(None if oracle_commit_payload.get("stay_noisy_energy_mean", None) is None else float(oracle_commit_payload["stay_noisy_energy_mean"])),
                    stay_noisy_energy_stderr=(None if oracle_commit_payload.get("stay_noisy_energy_stderr", None) is None else float(oracle_commit_payload["stay_noisy_energy_stderr"])),
                    selected_noisy_improvement_abs=(None if oracle_commit_payload.get("selected_noisy_improvement_abs", None) is None else float(oracle_commit_payload["selected_noisy_improvement_abs"])),
                    selected_noisy_improvement_ratio=(None if oracle_commit_payload.get("selected_noisy_improvement_ratio", None) is None else float(oracle_commit_payload["selected_noisy_improvement_ratio"])),
                    oracle_cache_hits=(0 if oracle_cache is None else int(oracle_cache.summary()["hits"])),
                    oracle_cache_misses=(0 if oracle_cache is None else int(oracle_cache.summary()["misses"])),
                    degraded_reason=degraded_reason,
                )
                self._ledger.append(dataclass_to_payload(ledger_entry))

            append_count = int(sum(1 for row in self._ledger if str(row.get("action_kind")) == "append_candidate"))
            stay_count = int(sum(1 for row in self._ledger if str(row.get("action_kind")) == "stay"))
            exact_decision_checkpoints = int(sum(1 for row in self._ledger if str(row.get("decision_backend")) == "exact"))
            oracle_decision_checkpoints = int(sum(1 for row in self._ledger if str(row.get("decision_backend")) == "oracle"))
            oracle_attempted_checkpoints = int(sum(1 for row in self._ledger if bool(row.get("oracle_attempted", False))))
            executed_backends = sorted({str(row.get("decision_backend", "exact")) for row in self._ledger}) or ["exact"]
            final_row = self._trajectory[-1] if self._trajectory else {}
            summary = {
                "mode": str(self.cfg.mode),
                "requested_decision_backend": ("oracle" if str(self.cfg.mode) == "oracle_v1" else "exact"),
                "status": ("completed_with_fallback" if int(self._degraded_checkpoint_count) > 0 else "completed"),
                "decision_backend": (
                    executed_backends[0]
                    if len(executed_backends) == 1
                    else "mixed"
                ),
                "executed_decision_backends": list(executed_backends),
                "decision_noise_mode": (None if oracle_decision_checkpoints <= 0 or self._oracle_base_config is None else str(self._oracle_base_config.noise_mode)),
                "oracle_estimate_kind": (None if oracle_decision_checkpoints <= 0 else self._oracle_estimate_kind()),
                "append_count": int(append_count),
                "stay_count": int(stay_count),
                "exact_decision_checkpoints": int(exact_decision_checkpoints),
                "oracle_decision_checkpoints": int(oracle_decision_checkpoints),
                "oracle_attempted_checkpoints": int(oracle_attempted_checkpoints),
                "degraded_checkpoints": int(self._degraded_checkpoint_count),
                "final_logical_block_count": int(self.current_layout.logical_parameter_count),
                "final_runtime_parameter_count": int(self.current_layout.runtime_parameter_count),
                "final_fidelity_exact": float(final_row.get("fidelity_exact", float("nan"))),
                "final_abs_energy_total_error": float(final_row.get("abs_energy_total_error", float("nan"))),
                "planning_audit": dict(self._planning_audit.summary()),
            }
            reference = {
                "kind": "static_exact_reference_from_replay_seed",
                "initial_state": "stage_result.psi_final",
                "times": [float(x) for x in self.times.tolist()],
            }
            return ControllerRunArtifacts(
                trajectory=[dict(row) for row in self._trajectory],
                ledger=[dict(row) for row in self._ledger],
                summary=summary,
                reference=reference,
            )
        finally:
            self._close_oracles()


__all__ = ["RealtimeCheckpointController", "ControllerRunArtifacts", "RuntimeTermCarrier"]
