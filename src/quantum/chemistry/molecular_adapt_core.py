from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Sequence

import numpy as np

from src.quantum.ansatz_parameterization import (
    build_parameter_layout,
    project_runtime_theta_block_mean,
    runtime_indices_for_logical_indices,
    runtime_insert_position,
    serialize_layout,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.compiled_polynomial import (
    adapt_commutator_grad_from_hpsi,
    apply_compiled_polynomial,
    compile_polynomial_action,
    energy_via_one_apply,
)
from src.quantum.hartree_fock_reference_state import hartree_fock_statevector
from src.quantum.spsa_optimizer import spsa_minimize
from src.quantum.vqe_latex_python_pairs import AnsatzTerm

from src.quantum.chemistry.molecular_uccsd import build_molecular_uccsd_pool
from src.quantum.chemistry.pipeline_phase_stack import (
    FullScoreConfig,
    MeasurementCacheAudit,
    Phase1CompileCostOracle,
    Phase2CurvatureOracle,
    Phase2NoveltyOracle,
    SimpleScoreConfig,
    StageController,
    StageControllerConfig,
    allowed_positions,
    build_candidate_features,
    build_full_candidate_features,
    detect_trough,
    family_repeat_cost_from_history,
    make_reduced_objective,
    measurement_group_keys_for_term,
    phase_shortlist_records,
    predict_reopt_window_for_position,
    raw_f_metric_from_state,
    resolve_reopt_active_indices,
    shortlist_records,
    should_probe_positions,
)


@dataclass(frozen=True)
class LocalAdaptResult:
    payload: dict[str, Any]
    psi_final: np.ndarray


_MATH_GRAD = r"g_k = 2 \\mathrm{Im}\\langle H\\psi | A_k \\psi \\rangle"
_MATH_OBJECTIVE = r"E(\\theta) = \\langle \\psi(\\theta)| H |\\psi(\\theta) \\rangle"


def _normalize_state(psi: np.ndarray) -> np.ndarray:
    vec = np.asarray(psi, dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        raise ValueError("State norm must be > 0.")
    return vec / norm


def _build_executor(
    ops: Sequence[AnsatzTerm],
    *,
    pauli_action_cache: dict[str, Any],
    parameterization_mode: str = "per_pauli_term",
) -> CompiledAnsatzExecutor:
    return CompiledAnsatzExecutor(
        list(ops),
        coefficient_tolerance=1e-12,
        ignore_identity=True,
        sort_terms=True,
        pauli_action_cache=pauli_action_cache,
        parameterization_mode=str(parameterization_mode),
    )


def _logical_theta_alias(
    theta_runtime: np.ndarray,
    layout: Any,
    *,
    parameterization_mode: str,
) -> np.ndarray:
    return (
        project_runtime_theta_block_mean(np.asarray(theta_runtime, dtype=float), layout)
        if str(parameterization_mode) == "per_pauli_term"
        else np.asarray(theta_runtime, dtype=float)
    )


def _candidate_family_id(label: str) -> str:
    head = str(label).split("(", 1)[0]
    return str(head).strip() or "unknown"


def _splice_candidate_at_position(
    *,
    ops: Sequence[AnsatzTerm],
    theta_runtime: np.ndarray,
    op: AnsatzTerm,
    position_id: int,
    parameterization_mode: str,
) -> tuple[list[AnsatzTerm], np.ndarray]:
    new_ops = list(ops)
    new_ops.insert(int(position_id), op)
    if str(parameterization_mode) == "logical_shared":
        insert_at = int(position_id)
        added = 1
    else:
        old_layout = build_parameter_layout(
            list(ops),
            ignore_identity=True,
            coefficient_tolerance=1e-12,
            sort_terms=True,
        )
        new_layout = build_parameter_layout(
            new_ops,
            ignore_identity=True,
            coefficient_tolerance=1e-12,
            sort_terms=True,
        )
        insert_at = int(runtime_insert_position(old_layout, int(position_id)))
        added = int(new_layout.runtime_parameter_count) - int(old_layout.runtime_parameter_count)
    if int(added) < 0:
        raise ValueError("Candidate insertion reduced runtime parameter count unexpectedly.")
    theta_new = np.insert(
        np.asarray(theta_runtime, dtype=float),
        int(insert_at),
        np.zeros(max(0, int(added)), dtype=float),
    )
    return new_ops, np.asarray(theta_new, dtype=float)


def _runtime_indices_for_logical(
    layout: Any,
    logical_indices: Sequence[int],
    *,
    parameterization_mode: str,
) -> list[int]:
    if str(parameterization_mode) == "logical_shared":
        return [int(i) for i in logical_indices]
    return runtime_indices_for_logical_indices(layout, logical_indices)


def _optimize_theta(
    *,
    objective: Any,
    x0: np.ndarray,
    maxiter: int,
    optimizer: str,
    seed: int,
) -> tuple[np.ndarray, float, int, dict[str, Any]]:
    x0_arr = np.asarray(x0, dtype=float).reshape(-1)
    optimizer_key = str(optimizer).strip().upper()
    if optimizer_key == "SPSA":
        res = spsa_minimize(
            objective,
            x0_arr,
            maxiter=int(maxiter),
            seed=int(seed),
        )
        return (
            np.asarray(res.x, dtype=float).reshape(-1),
            float(res.fun),
            int(res.nfev),
            {
                "optimizer": "SPSA",
                "success": bool(res.success),
                "message": str(res.message),
                "nit": int(res.nit),
            },
        )

    try:
        from scipy.optimize import minimize as scipy_minimize
    except Exception as exc:  # pragma: no cover
        raise ImportError("SciPy is required for COBYLA/POWELL in chemistry-local ADAPT.") from exc

    if optimizer_key not in {"COBYLA", "POWELL"}:
        raise ValueError("optimizer must be one of {'COBYLA','POWELL','SPSA'}.")
    options = {"maxiter": int(maxiter)}
    if optimizer_key == "COBYLA":
        options["tol"] = 1e-10
    res = scipy_minimize(objective, x0_arr, method=optimizer_key, options=options)
    return (
        np.asarray(res.x, dtype=float).reshape(-1),
        float(res.fun),
        int(getattr(res, "nfev", 0)),
        {
            "optimizer": str(optimizer_key),
            "success": bool(res.success),
            "message": str(getattr(res, "message", "")),
            "nit": int(getattr(res, "nit", 0) or 0),
        },
    )


def run_pipeline_local_adapt_vqe_with_pool(
    *,
    h_poly: Any,
    psi_ref: np.ndarray,
    pool: Sequence[AnsatzTerm],
    exact_gs_energy: float | None,
    max_depth: int,
    eps_grad: float,
    eps_energy: float,
    maxiter: int,
    optimizer: str,
    seed: int,
    pool_type: str,
    metadata: dict[str, Any] | None = None,
    parameterization_mode: str = "logical_shared",
    finite_angle: float = 0.05,
    finite_angle_min_improvement: float = 0.0,
    reopt_policy: str = "windowed",
    window_size: int = 3,
    window_topk: int = 0,
    full_refit_every: int = 0,
    phase1_shortlist_size: int = 16,
) -> LocalAdaptResult:
    if int(max_depth) < 1:
        raise ValueError("max_depth must be >= 1.")
    if float(eps_grad) < 0.0 or float(eps_energy) < 0.0:
        raise ValueError("eps_grad and eps_energy must be >= 0.")

    t0 = time.perf_counter()
    psi_ref_vec = _normalize_state(np.asarray(psi_ref, dtype=complex).reshape(-1))
    pool_list = list(pool)

    pauli_action_cache: dict[str, Any] = {}
    h_compiled = compile_polynomial_action(h_poly, pauli_action_cache=pauli_action_cache)
    pool_compiled = [
        compile_polynomial_action(term.polynomial, pauli_action_cache=pauli_action_cache)
        for term in pool_list
    ]
    compiled_term_cache: dict[str, Any] = {}

    empty_layout = build_parameter_layout(
        [],
        ignore_identity=True,
        coefficient_tolerance=1e-12,
        sort_terms=True,
    )
    selected_ops: list[AnsatzTerm] = []
    selected_indices: list[int] = []
    theta_runtime = np.zeros(0, dtype=float)
    available_indices = list(range(len(pool_list)))
    executor: CompiledAnsatzExecutor | None = None
    layout = empty_layout
    history: list[dict[str, Any]] = []
    nfev_total = 0
    stop_reason = "max_depth"
    drop_plateau_hits = 0

    phase1_cfg = SimpleScoreConfig(
        lambda_F=1.0,
        lambda_compile=0.05,
        lambda_measure=0.02,
        lambda_leak=0.0,
        z_alpha=0.0,
    )
    phase2_cfg = FullScoreConfig(
        z_alpha=0.0,
        lambda_F=1.0,
        lambda_H=1e-6,
        rho=0.25,
        gamma_N=1.0,
        shortlist_fraction=0.5,
        shortlist_size=int(max(4, phase1_shortlist_size)),
        phase2_frontier_ratio=0.9,
        phase3_frontier_ratio=0.9,
    )
    stage_cfg = StageControllerConfig(
        plateau_patience=2,
        weak_drop_threshold=float(max(eps_energy, 1e-8)),
        probe_margin_ratio=1.0,
        max_probe_positions=6,
        append_admit_threshold=0.05,
        family_repeat_patience=2,
        cap_phase1_min=int(max(1, min(int(phase1_shortlist_size), len(pool_list) or 1))),
        cap_phase1_max=int(max(1, min(int(phase1_shortlist_size), len(pool_list) or 1))),
        cap_phase2_min=int(max(1, min(int(phase2_cfg.shortlist_size), len(pool_list) or 1))),
        cap_phase2_max=int(max(1, min(int(phase2_cfg.shortlist_size), len(pool_list) or 1))),
        cap_phase3_min=int(max(1, min(int(phase2_cfg.shortlist_size), len(pool_list) or 1))),
        cap_phase3_max=int(max(1, min(int(phase2_cfg.shortlist_size), len(pool_list) or 1))),
    )
    stage_controller = StageController(stage_cfg)
    stage_controller.start_with_seed()
    compile_oracle = Phase1CompileCostOracle()
    measure_cache = MeasurementCacheAudit(nominal_shots_per_group=1)
    novelty_oracle = Phase2NoveltyOracle()
    curvature_oracle = Phase2CurvatureOracle()

    psi_current = np.array(psi_ref_vec, copy=True)
    energy_current, hpsi_current = energy_via_one_apply(psi_current, h_compiled)
    if exact_gs_energy is not None and abs(float(energy_current) - float(exact_gs_energy)) <= float(eps_energy):
        payload = {
            "success": True,
            "method": "chemistry_local_phase123_adapt_v1",
            "energy": float(energy_current),
            "exact_energy_from_final_state": float(energy_current),
            "exact_gs_energy": float(exact_gs_energy),
            "delta_e": float(energy_current - float(exact_gs_energy)),
            "abs_delta_e": float(abs(energy_current - float(exact_gs_energy))),
            "ansatz_depth": 0,
            "num_parameters": 0,
            "logical_num_parameters": 0,
            "optimal_point": [],
            "logical_optimal_point": [],
            "parameterization": serialize_layout(layout),
            "operators": [],
            "selected_pool_indices": [],
            "pool_size": int(len(pool_list)),
            "pool_type": str(pool_type),
            "stop_reason": "eps_energy_initial",
            "nfev_total": 0,
            "adapt_inner_optimizer": str(optimizer).strip().upper(),
            "parameterization_mode": str(parameterization_mode),
            "history": [],
            "elapsed_sec": float(time.perf_counter() - t0),
            **dict(metadata or {}),
        }
        return LocalAdaptResult(payload=payload, psi_final=psi_current)

    prev_abs_delta = (
        None
        if exact_gs_energy is None
        else float(abs(float(energy_current) - float(exact_gs_energy)))
    )

    for depth in range(1, int(max_depth) + 1):
        if executor is not None:
            psi_current = _normalize_state(executor.prepare_state(theta_runtime, psi_ref_vec))
            energy_current, hpsi_current = energy_via_one_apply(psi_current, h_compiled)

        if not available_indices:
            stop_reason = "pool_exhausted"
            break

        layout = empty_layout if executor is None else executor.layout
        theta_logical = _logical_theta_alias(
            theta_runtime,
            layout,
            parameterization_mode=str(parameterization_mode),
        )
        append_position = int(layout.logical_parameter_count)
        pre_snapshot = stage_controller.pre_step_snapshot(depth_local=int(depth - 1), max_depth=int(max_depth))

        gradients_by_index: dict[int, float] = {}
        grad_abs_by_index: dict[int, float] = {}
        for pool_index in available_indices:
            apsi = apply_compiled_polynomial(psi_current, pool_compiled[int(pool_index)])
            grad = float(adapt_commutator_grad_from_hpsi(hpsi_current, apsi))
            gradients_by_index[int(pool_index)] = float(grad)
            grad_abs_by_index[int(pool_index)] = float(abs(grad))
        max_grad = float(max(grad_abs_by_index.values(), default=0.0))

        shortlisted_indices = sorted(
            list(available_indices),
            key=lambda idx: (-float(grad_abs_by_index[int(idx)]), int(idx)),
        )[: max(1, min(len(available_indices), 64))]

        periodic_full_refit = bool(int(full_refit_every) > 0 and int(depth) % int(full_refit_every) == 0)

        def _phase1_records_for_positions(positions_considered: Sequence[int]) -> list[dict[str, Any]]:
            records_out: list[dict[str, Any]] = []
            metric_cache: dict[int, float] = {}
            for idx in shortlisted_indices:
                label = str(pool_list[int(idx)].label)
                family = _candidate_family_id(label)
                metric_raw = metric_cache.get(int(idx))
                if metric_raw is None:
                    metric_raw = raw_f_metric_from_state(
                        psi_state=np.asarray(psi_current, dtype=complex),
                        candidate_label=str(label),
                        candidate_term=pool_list[int(idx)],
                        compiled_cache=compiled_term_cache,
                        pauli_action_cache=pauli_action_cache,
                    )
                    metric_cache[int(idx)] = float(metric_raw)
                family_repeat_cost = float(
                    family_repeat_cost_from_history(
                        history_rows=history,
                        candidate_family=str(family),
                    )
                )
                for pos in positions_considered:
                    predicted_window = predict_reopt_window_for_position(
                        theta=np.asarray(theta_logical, dtype=float),
                        position_id=int(pos),
                        policy=str(reopt_policy),
                        window_size=int(window_size),
                        window_topk=int(window_topk),
                        periodic_full_refit_triggered=bool(periodic_full_refit),
                    )
                    inherited_window = [int(i) for i in predicted_window if int(i) != int(pos)]
                    compile_est = compile_oracle.estimate(
                        candidate_term_count=int(len(pool_compiled[int(idx)].terms)),
                        position_id=int(pos),
                        append_position=int(append_position),
                        refit_active_count=int(len(inherited_window)),
                        candidate_term=pool_list[int(idx)],
                    )
                    measurement_stats = measure_cache.estimate(measurement_group_keys_for_term(pool_list[int(idx)]))
                    feat = build_candidate_features(
                        stage_name=str(stage_controller.stage_name),
                        candidate_label=str(label),
                        candidate_family=str(family),
                        candidate_pool_index=int(idx),
                        position_id=int(pos),
                        append_position=int(append_position),
                        positions_considered=[int(x) for x in positions_considered],
                        gradient_signed=float(gradients_by_index[int(idx)]),
                        metric_proxy=float(metric_raw),
                        sigma_hat=0.0,
                        refit_window_indices=[int(i) for i in inherited_window],
                        compile_cost=compile_est,
                        measurement_stats=measurement_stats,
                        leakage_penalty=0.0,
                        stage_gate_open=True,
                        leakage_gate_open=True,
                        trough_probe_triggered=(len(positions_considered) > 1),
                        trough_detected=False,
                        family_repeat_cost=float(family_repeat_cost),
                        cfg=phase1_cfg,
                        cheap_score_cfg=phase2_cfg,
                        current_depth=int(depth),
                        max_depth=int(max_depth),
                        lifetime_cost_mode="off",
                        remaining_evaluations_proxy_mode="none",
                    )
                    records_out.append(
                        {
                            "feature": feat,
                            "simple_score": float(feat.simple_score or float("-inf")),
                            "cheap_score": float(feat.cheap_score or float("-inf")),
                            "candidate_pool_index": int(idx),
                            "position_id": int(pos),
                            "candidate_term": pool_list[int(idx)],
                        }
                    )
            return records_out

        append_records = _phase1_records_for_positions([int(append_position)])
        phase1_raw_scores = [float(rec.get("cheap_score", rec.get("simple_score", float("-inf")))) for rec in append_records]
        controller_snapshot = stage_controller.finalize_step_snapshot(
            pre_snapshot=pre_snapshot,
            phase1_raw_scores=phase1_raw_scores,
        )

        append_best = max(
            append_records,
            key=lambda rec: (
                float(rec.get("cheap_score", float("-inf"))),
                float(rec.get("simple_score", float("-inf"))),
                -int(rec.get("candidate_pool_index", -1)),
            ),
        )
        append_best_feat = append_best.get("feature")
        append_best_score = float(append_best.get("cheap_score", append_best.get("simple_score", float("-inf"))))
        append_best_family = (
            str(append_best_feat.candidate_family)
            if hasattr(append_best_feat, "candidate_family")
            else "unknown"
        )
        repeated_family_flat = bool(
            family_repeat_cost_from_history(history_rows=history, candidate_family=str(append_best_family))
            >= int(stage_cfg.family_repeat_patience)
        )
        probe_positions, probe_reason = should_probe_positions(
            stage_name=str(stage_controller.stage_name),
            drop_plateau_hits=int(drop_plateau_hits),
            max_grad=float(max_grad),
            eps_grad=float(eps_grad),
            append_score=float(append_best_score),
            finite_angle_flat=bool(float(max_grad) < float(eps_grad)),
            repeated_family_flat=bool(repeated_family_flat),
            cfg=stage_cfg,
        )
        positions_considered = (
            allowed_positions(
                n_params=int(layout.logical_parameter_count),
                append_position=int(append_position),
                active_window_indices=predict_reopt_window_for_position(
                    theta=np.asarray(theta_logical, dtype=float),
                    position_id=int(append_position),
                    policy=str(reopt_policy),
                    window_size=int(window_size),
                    window_topk=int(window_topk),
                    periodic_full_refit_triggered=bool(periodic_full_refit),
                ),
                max_positions=int(stage_cfg.max_probe_positions),
            )
            if bool(probe_positions)
            else [int(append_position)]
        )
        phase1_records = (
            append_records
            if len(positions_considered) == 1
            else _phase1_records_for_positions(positions_considered)
        )
        best_non_append_score = float("-inf")
        best_non_append_g_lcb = 0.0
        for rec in phase1_records:
            if int(rec.get("position_id", append_position)) == int(append_position):
                continue
            feat = rec.get("feature")
            score_val = float(rec.get("cheap_score", rec.get("simple_score", float("-inf"))))
            if score_val > best_non_append_score:
                best_non_append_score = float(score_val)
                best_non_append_g_lcb = float(getattr(feat, "g_lcb", 0.0))
        trough_detected = bool(
            detect_trough(
                append_score=float(append_best_score),
                best_non_append_score=float(best_non_append_score),
                best_non_append_g_lcb=float(best_non_append_g_lcb),
                margin_ratio=float(stage_cfg.probe_margin_ratio),
                append_admit_threshold=float(stage_cfg.append_admit_threshold),
            )
        ) if len(positions_considered) > 1 else False

        cheap_records = shortlist_records(
            phase1_records,
            cfg=phase2_cfg,
            score_key="simple_score",
        )
        phase1_shortlisted = phase_shortlist_records(
            cheap_records,
            score_key="cheap_score",
            threshold=float(controller_snapshot.phase_thresholds.get("phase1", 0.0)),
            cap=int(controller_snapshot.phase_caps.get("phase1", phase1_shortlist_size)),
            frontier_ratio=float(controller_snapshot.frontier_ratio),
            tie_break_score_key="simple_score",
            shortlist_flag="phase1_shortlisted",
        )

        full_records: list[dict[str, Any]] = []
        scaffold_cache: dict[tuple[int, ...], Any] = {}
        for rec in phase1_shortlisted:
            feat_base = rec.get("feature")
            if feat_base is None:
                continue
            runtime_window = _runtime_indices_for_logical(
                layout,
                list(getattr(feat_base, "refit_window_indices", [])),
                parameterization_mode=str(parameterization_mode),
            )
            scaffold_key = tuple(int(i) for i in runtime_window)
            scaffold_context = scaffold_cache.get(scaffold_key)
            if scaffold_context is None:
                scaffold_context = novelty_oracle.prepare_scaffold_context(
                    selected_ops=list(selected_ops),
                    theta=np.asarray(theta_runtime, dtype=float),
                    psi_ref=np.asarray(psi_ref_vec, dtype=complex),
                    psi_state=np.asarray(psi_current, dtype=complex),
                    h_compiled=h_compiled,
                    hpsi_state=np.asarray(hpsi_current, dtype=complex),
                    refit_window_indices=list(runtime_window),
                    pauli_action_cache=pauli_action_cache,
                    parameterization_mode=str(parameterization_mode),
                )
                scaffold_cache[scaffold_key] = scaffold_context
            feat_full = build_full_candidate_features(
                base_feature=feat_base,
                candidate_term=rec["candidate_term"],
                cfg=phase2_cfg,
                novelty_oracle=novelty_oracle,
                curvature_oracle=curvature_oracle,
                scaffold_context=scaffold_context,
                h_compiled=h_compiled,
                compiled_cache=compiled_term_cache,
                pauli_action_cache=pauli_action_cache,
                optimizer_memory=None,
                parameterization_mode=str(parameterization_mode),
            )
            full_records.append(
                {
                    **dict(rec),
                    "feature": feat_full,
                    "simple_score": float(feat_full.simple_score or float("-inf")),
                    "cheap_score": float(feat_full.cheap_score or float("-inf")),
                    "phase2_raw_score": float(feat_full.phase2_raw_score or float("-inf")),
                    "full_v2_score": float(feat_full.full_v2_score or float("-inf")),
                }
            )

        phase2_shortlisted = phase_shortlist_records(
            full_records,
            score_key="phase2_raw_score",
            threshold=float(controller_snapshot.phase_thresholds.get("phase2", 0.0)),
            cap=int(controller_snapshot.phase_caps.get("phase2", phase2_cfg.shortlist_size)),
            frontier_ratio=float(phase2_cfg.phase2_frontier_ratio),
            tie_break_score_key="cheap_score",
            shortlist_flag="phase2_shortlisted",
        )
        phase3_shortlisted = phase_shortlist_records(
            phase2_shortlisted,
            score_key="full_v2_score",
            threshold=float(controller_snapshot.phase_thresholds.get("phase3", 0.0)),
            cap=int(controller_snapshot.phase_caps.get("phase3", phase2_cfg.shortlist_size)),
            frontier_ratio=float(phase2_cfg.phase3_frontier_ratio),
            tie_break_score_key="phase2_raw_score",
            shortlist_flag="phase3_shortlisted",
        )
        positive_phase3 = [
            dict(rec)
            for rec in phase3_shortlisted
            if float(rec.get("full_v2_score", float("-inf"))) > 0.0
        ]
        if positive_phase3:
            best_rec = dict(positive_phase3[0])
            selection_mode = "phase3_rerank"
        elif phase2_shortlisted:
            best_rec = dict(phase2_shortlisted[0])
            selection_mode = "phase2_raw_fallback"
        elif phase1_shortlisted:
            best_rec = dict(phase1_shortlisted[0])
            selection_mode = "phase1_shortlist_fallback"
        elif append_records:
            best_rec = dict(append_best)
            selection_mode = "append_only_fallback"
        else:
            stop_reason = "pool_exhausted"
            break

        stage_name_now, stage_reason = stage_controller.resolve_stage_transition(
            drop_plateau_hits=int(drop_plateau_hits),
            trough_detected=bool(trough_detected),
            residual_opened=False,
        )

        if float(max_grad) <= float(eps_grad):
            best_probe_improvement = 0.0
            best_probe_idx: int | None = None
            best_probe_pos = int(append_position)
            best_probe_theta = np.array(theta_runtime, copy=True)
            for pool_index in available_indices:
                trial_ops, theta_probe = _splice_candidate_at_position(
                    ops=list(selected_ops),
                    theta_runtime=np.asarray(theta_runtime, dtype=float),
                    op=pool_list[int(pool_index)],
                    position_id=int(append_position),
                    parameterization_mode=str(parameterization_mode),
                )
                trial_executor = _build_executor(
                    trial_ops,
                    pauli_action_cache=pauli_action_cache,
                    parameterization_mode=str(parameterization_mode),
                )
                added = int(trial_executor.num_parameters) - int(theta_runtime.size)
                if added <= 0:
                    continue
                insert_at = int(theta_probe.size - added)
                for sign in (+1.0, -1.0):
                    theta_trial = np.asarray(theta_probe, dtype=float)
                    theta_trial[insert_at:] = float(sign) * float(finite_angle)
                    psi_probe = _normalize_state(trial_executor.prepare_state(theta_trial, psi_ref_vec))
                    energy_probe, _ = energy_via_one_apply(psi_probe, h_compiled)
                    improvement = float(energy_current - energy_probe)
                    if improvement > float(best_probe_improvement):
                        best_probe_improvement = improvement
                        best_probe_idx = int(pool_index)
                        best_probe_theta = np.asarray(theta_trial, dtype=float)
            if best_probe_idx is None or best_probe_improvement <= float(finite_angle_min_improvement):
                stop_reason = "eps_grad"
                history.append(
                    {
                        "depth": int(depth),
                        "energy_before": float(energy_current),
                        "max_grad": float(max_grad),
                        "selected_operator": None,
                        "stop_reason": "eps_grad",
                        "probe_reason": str(probe_reason),
                    }
                )
                break
            selected_pool_index = int(best_probe_idx)
            selected_position = int(best_probe_pos)
            selection_mode = "finite_angle_fallback"
            selected_ops, theta_runtime = _splice_candidate_at_position(
                ops=list(selected_ops),
                theta_runtime=np.asarray(theta_runtime, dtype=float),
                op=pool_list[int(selected_pool_index)],
                position_id=int(selected_position),
                parameterization_mode=str(parameterization_mode),
            )
            theta_runtime = np.asarray(best_probe_theta, dtype=float)
        else:
            selected_pool_index = int(best_rec.get("candidate_pool_index"))
            selected_position = int(best_rec.get("position_id", append_position))
            selected_ops, theta_runtime = _splice_candidate_at_position(
                ops=list(selected_ops),
                theta_runtime=np.asarray(theta_runtime, dtype=float),
                op=pool_list[int(selected_pool_index)],
                position_id=int(selected_position),
                parameterization_mode=str(parameterization_mode),
            )

        selected_indices.append(int(selected_pool_index))
        available_indices = [idx for idx in available_indices if int(idx) != int(selected_pool_index)]
        executor = _build_executor(
            selected_ops,
            pauli_action_cache=pauli_action_cache,
            parameterization_mode=str(parameterization_mode),
        )
        layout = executor.layout
        theta_logical_now = _logical_theta_alias(
            theta_runtime,
            layout,
            parameterization_mode=str(parameterization_mode),
        )
        reopt_active_logical, reopt_policy_effective = resolve_reopt_active_indices(
            policy=str(reopt_policy),
            n=int(layout.logical_parameter_count),
            theta=np.asarray(theta_logical_now, dtype=float),
            window_size=int(window_size),
            window_topk=int(window_topk),
            periodic_full_refit_triggered=bool(periodic_full_refit),
        )
        reopt_active_runtime = _runtime_indices_for_logical(
            layout,
            reopt_active_logical,
            parameterization_mode=str(parameterization_mode),
        )

        def _objective(theta_vec: np.ndarray) -> float:
            psi = _normalize_state(executor.prepare_state(theta_vec, psi_ref_vec))
            energy_val, _ = energy_via_one_apply(psi, h_compiled)
            return float(energy_val)

        reduced_objective, x0 = make_reduced_objective(
            np.asarray(theta_runtime, dtype=float),
            [int(i) for i in reopt_active_runtime],
            _objective,
        )
        theta_opt, _opt_fun, nfev_opt, opt_meta = _optimize_theta(
            objective=reduced_objective,
            x0=np.asarray(x0, dtype=float),
            maxiter=int(maxiter),
            optimizer=str(optimizer),
            seed=int(seed + depth - 1),
        )
        nfev_total += int(nfev_opt)
        if len(reopt_active_runtime) == int(theta_runtime.size):
            theta_runtime = np.asarray(theta_opt, dtype=float)
        else:
            theta_runtime = np.asarray(theta_runtime, dtype=float)
            theta_opt = np.asarray(theta_opt, dtype=float).ravel()
            for k, idx in enumerate(reopt_active_runtime):
                theta_runtime[int(idx)] = float(theta_opt[int(k)])

        psi_current = _normalize_state(executor.prepare_state(theta_runtime, psi_ref_vec))
        energy_before = float(energy_current)
        energy_current, hpsi_current = energy_via_one_apply(psi_current, h_compiled)
        theta_logical_now = _logical_theta_alias(
            theta_runtime,
            layout,
            parameterization_mode=str(parameterization_mode),
        )
        measure_cache.commit(measurement_group_keys_for_term(pool_list[int(selected_pool_index)]))
        stage_controller.record_admission(
            selector_step=int(depth),
            energy_before=float(energy_before),
            energy_after_refit=float(energy_current),
        )

        abs_delta_e = (
            None
            if exact_gs_energy is None
            else float(abs(float(energy_current) - float(exact_gs_energy)))
        )
        if prev_abs_delta is not None and abs_delta_e is not None:
            delta_drop = float(prev_abs_delta - abs_delta_e)
            drop_plateau_hits = int(drop_plateau_hits + 1) if float(delta_drop) < float(stage_cfg.weak_drop_threshold) else 0
            prev_abs_delta = float(abs_delta_e)

        selected_feat = best_rec.get("feature")
        selected_family = (
            str(selected_feat.candidate_family)
            if selected_feat is not None and hasattr(selected_feat, "candidate_family")
            else _candidate_family_id(str(pool_list[int(selected_pool_index)].label))
        )
        history.append(
            {
                "depth": int(depth),
                "selected_pool_index": int(selected_pool_index),
                "selected_operator": str(pool_list[int(selected_pool_index)].label),
                "selected_position": int(selected_position),
                "selection_mode": str(selection_mode),
                "stage_name": str(stage_name_now),
                "stage_reason": str(stage_reason),
                "probe_reason": str(probe_reason),
                "positions_considered": [int(x) for x in positions_considered],
                "trough_detected": bool(trough_detected),
                "max_grad": float(max_grad),
                "energy": float(energy_current),
                "abs_delta_e": abs_delta_e,
                "nfev_opt": int(nfev_opt),
                "optimizer": dict(opt_meta),
                "parameter_count": int(theta_runtime.size),
                "logical_parameter_count": int(layout.logical_parameter_count),
                "logical_optimal_point": [float(x) for x in theta_logical_now.tolist()],
                "candidate_family": str(selected_family),
                "reopt_policy_effective": str(reopt_policy_effective),
                "reopt_active_logical": [int(x) for x in reopt_active_logical],
                "phase1_shortlist_count": int(len(phase1_shortlisted)),
                "phase2_shortlist_count": int(len(phase2_shortlisted)),
                "phase3_shortlist_count": int(len(phase3_shortlisted)),
            }
        )

        if exact_gs_energy is not None and abs(float(energy_current) - float(exact_gs_energy)) <= float(eps_energy):
            stop_reason = "eps_energy"
            break

    if executor is None:
        psi_final = np.array(psi_ref_vec, copy=True)
        layout = empty_layout
        logical_theta = np.zeros(0, dtype=float)
    else:
        psi_final = _normalize_state(executor.prepare_state(theta_runtime, psi_ref_vec))
        logical_theta = _logical_theta_alias(
            theta_runtime,
            layout,
            parameterization_mode=str(parameterization_mode),
        )

    final_energy, _ = energy_via_one_apply(psi_final, h_compiled)
    if exact_gs_energy is None:
        delta_e = None
        abs_delta_e = None
        exact_gs_value = None
    else:
        exact_gs_value = float(exact_gs_energy)
        delta_e = float(final_energy - exact_gs_value)
        abs_delta_e = float(abs(delta_e))

    payload = {
        "success": True,
        "method": "chemistry_local_phase123_adapt_v1",
        "energy": float(final_energy),
        "exact_energy_from_final_state": float(final_energy),
        "exact_gs_energy": exact_gs_value,
        "delta_e": delta_e,
        "abs_delta_e": abs_delta_e,
        "ansatz_depth": int(len(selected_ops)),
        "num_parameters": int(theta_runtime.size),
        "logical_num_parameters": int(layout.logical_parameter_count),
        "optimal_point": [float(x) for x in np.asarray(theta_runtime, dtype=float).tolist()],
        "logical_optimal_point": [float(x) for x in np.asarray(logical_theta, dtype=float).tolist()],
        "parameterization": serialize_layout(layout),
        "operators": [str(op.label) for op in selected_ops],
        "selected_pool_indices": [int(x) for x in selected_indices],
        "pool_size": int(len(pool_list)),
        "pool_type": str(pool_type),
        "stop_reason": str(stop_reason),
        "nfev_total": int(nfev_total),
        "adapt_inner_optimizer": str(optimizer).strip().upper(),
        "parameterization_mode": str(parameterization_mode),
        "history": history,
        "elapsed_sec": float(time.perf_counter() - t0),
        "phase_stack": {
            "enabled": True,
            "reopt_policy": str(reopt_policy),
            "window_size": int(window_size),
            "window_topk": int(window_topk),
            "full_refit_every": int(full_refit_every),
        },
        **dict(metadata or {}),
    }
    return LocalAdaptResult(payload=payload, psi_final=psi_final)


def run_local_adapt_vqe_with_pool(
    *,
    h_poly: Any,
    psi_ref: np.ndarray,
    pool: Sequence[AnsatzTerm],
    exact_gs_energy: float | None,
    max_depth: int,
    eps_grad: float,
    eps_energy: float,
    maxiter: int,
    optimizer: str,
    seed: int,
    pool_type: str,
    metadata: dict[str, Any] | None = None,
    parameterization_mode: str = "per_pauli_term",
    finite_angle: float = 0.05,
    finite_angle_min_improvement: float = 0.0,
) -> LocalAdaptResult:
    if int(max_depth) < 1:
        raise ValueError("max_depth must be >= 1.")
    if float(eps_grad) < 0.0 or float(eps_energy) < 0.0:
        raise ValueError("eps_grad and eps_energy must be >= 0.")

    t0 = time.perf_counter()
    psi_ref_vec = _normalize_state(np.asarray(psi_ref, dtype=complex).reshape(-1))
    pool_list = list(pool)

    pauli_action_cache: dict[str, Any] = {}
    h_compiled = compile_polynomial_action(h_poly, pauli_action_cache=pauli_action_cache)
    pool_compiled = [
        compile_polynomial_action(term.polynomial, pauli_action_cache=pauli_action_cache)
        for term in pool_list
    ]

    empty_layout = build_parameter_layout(
        [],
        ignore_identity=True,
        coefficient_tolerance=1e-12,
        sort_terms=True,
    )
    selected_ops: list[AnsatzTerm] = []
    selected_indices: list[int] = []
    theta_runtime = np.zeros(0, dtype=float)
    available_indices = list(range(len(pool_list)))
    executor: CompiledAnsatzExecutor | None = None
    layout = empty_layout
    history: list[dict[str, Any]] = []
    nfev_total = 0
    stop_reason = "max_depth"

    psi_current = np.array(psi_ref_vec, copy=True)
    energy_current, hpsi_current = energy_via_one_apply(psi_current, h_compiled)
    if exact_gs_energy is not None and abs(float(energy_current) - float(exact_gs_energy)) <= float(eps_energy):
        stop_reason = "eps_energy_initial"
        payload = {
            "success": True,
            "method": "chemistry_local_adapt_v1",
            "energy": float(energy_current),
            "exact_energy_from_final_state": float(energy_current),
            "exact_gs_energy": float(exact_gs_energy),
            "delta_e": float(energy_current - float(exact_gs_energy)),
            "abs_delta_e": float(abs(energy_current - float(exact_gs_energy))),
            "ansatz_depth": 0,
            "num_parameters": 0,
            "logical_num_parameters": 0,
            "optimal_point": [],
            "logical_optimal_point": [],
            "parameterization": serialize_layout(layout),
            "operators": [],
            "selected_pool_indices": [],
            "pool_size": int(len(pool_list)),
            "pool_type": str(pool_type),
            "stop_reason": str(stop_reason),
            "nfev_total": 0,
            "adapt_inner_optimizer": str(optimizer).strip().upper(),
            "parameterization_mode": str(parameterization_mode),
            "history": [],
            "elapsed_sec": float(time.perf_counter() - t0),
            **dict(metadata or {}),
        }
        return LocalAdaptResult(payload=payload, psi_final=psi_current)

    for depth in range(1, int(max_depth) + 1):
        if executor is not None:
            psi_current = _normalize_state(executor.prepare_state(theta_runtime, psi_ref_vec))
            energy_current, hpsi_current = energy_via_one_apply(psi_current, h_compiled)

        gradients: list[float] = []
        for pool_index in available_indices:
            apsi = apply_compiled_polynomial(psi_current, pool_compiled[int(pool_index)])
            gradients.append(float(adapt_commutator_grad_from_hpsi(hpsi_current, apsi)))

        if not gradients:
            stop_reason = "pool_exhausted"
            break

        grad_abs = np.asarray([abs(x) for x in gradients], dtype=float)
        local_best_pos = int(np.argmax(grad_abs))
        max_grad = float(grad_abs[local_best_pos])

        if max_grad <= float(eps_grad):
            best_probe_improvement = 0.0
            best_probe_idx: int | None = None
            best_probe_theta = np.array(theta_runtime, copy=True)
            for pool_index in available_indices:
                trial_ops = list(selected_ops) + [pool_list[int(pool_index)]]
                trial_executor = _build_executor(
                    trial_ops,
                    pauli_action_cache=pauli_action_cache,
                    parameterization_mode=str(parameterization_mode),
                )
                added = int(trial_executor.num_parameters) - int(theta_runtime.size)
                if added <= 0:
                    continue
                for sign in (+1.0, -1.0):
                    theta_probe = np.concatenate(
                        [
                            np.asarray(theta_runtime, dtype=float),
                            np.full(added, float(sign) * float(finite_angle), dtype=float),
                        ]
                    )
                    psi_probe = _normalize_state(trial_executor.prepare_state(theta_probe, psi_ref_vec))
                    energy_probe, _ = energy_via_one_apply(psi_probe, h_compiled)
                    improvement = float(energy_current - energy_probe)
                    if improvement > float(best_probe_improvement):
                        best_probe_improvement = improvement
                        best_probe_idx = int(pool_index)
                        best_probe_theta = np.asarray(theta_probe, dtype=float)
            if best_probe_idx is None or best_probe_improvement <= float(finite_angle_min_improvement):
                stop_reason = "eps_grad"
                history.append(
                    {
                        "depth": int(depth),
                        "energy_before": float(energy_current),
                        "max_grad": float(max_grad),
                        "selected_operator": None,
                        "stop_reason": "eps_grad",
                        "best_probe_improvement": float(best_probe_improvement),
                    }
                )
                break
            selected_pool_index = int(best_probe_idx)
            init_theta = np.asarray(best_probe_theta, dtype=float)
            selection_mode = "finite_angle_fallback"
        else:
            selected_pool_index = int(available_indices[local_best_pos])
            selection_mode = "gradient"
            selected_ops_next = list(selected_ops) + [pool_list[selected_pool_index]]
            executor_next = _build_executor(
                selected_ops_next,
                pauli_action_cache=pauli_action_cache,
                parameterization_mode=str(parameterization_mode),
            )
            added = int(executor_next.num_parameters) - int(theta_runtime.size)
            init_theta = np.concatenate(
                [np.asarray(theta_runtime, dtype=float), np.zeros(max(0, added), dtype=float)]
            )

        selected_ops = list(selected_ops) + [pool_list[selected_pool_index]]
        selected_indices.append(int(selected_pool_index))
        available_indices = [idx for idx in available_indices if int(idx) != int(selected_pool_index)]
        executor = _build_executor(
            selected_ops,
            pauli_action_cache=pauli_action_cache,
            parameterization_mode=str(parameterization_mode),
        )
        layout = executor.layout

        def _objective(theta_vec: np.ndarray) -> float:
            psi = _normalize_state(executor.prepare_state(theta_vec, psi_ref_vec))
            energy_val, _ = energy_via_one_apply(psi, h_compiled)
            return float(energy_val)

        theta_runtime, _opt_fun, nfev_opt, opt_meta = _optimize_theta(
            objective=_objective,
            x0=np.asarray(init_theta, dtype=float),
            maxiter=int(maxiter),
            optimizer=str(optimizer),
            seed=int(seed + depth - 1),
        )
        nfev_total += int(nfev_opt)

        psi_current = _normalize_state(executor.prepare_state(theta_runtime, psi_ref_vec))
        energy_current, hpsi_current = energy_via_one_apply(psi_current, h_compiled)
        logical_theta = (
            project_runtime_theta_block_mean(theta_runtime, layout)
            if str(parameterization_mode) == "per_pauli_term"
            else np.asarray(theta_runtime, dtype=float)
        )
        abs_delta_e = (
            None
            if exact_gs_energy is None
            else float(abs(float(energy_current) - float(exact_gs_energy)))
        )
        history.append(
            {
                "depth": int(depth),
                "selected_pool_index": int(selected_pool_index),
                "selected_operator": str(pool_list[selected_pool_index].label),
                "selection_mode": str(selection_mode),
                "max_grad": float(max_grad),
                "energy": float(energy_current),
                "abs_delta_e": abs_delta_e,
                "nfev_opt": int(nfev_opt),
                "optimizer": dict(opt_meta),
                "parameter_count": int(theta_runtime.size),
                "logical_parameter_count": int(layout.logical_parameter_count),
                "logical_optimal_point": [float(x) for x in logical_theta.tolist()],
            }
        )

        if exact_gs_energy is not None and abs(float(energy_current) - float(exact_gs_energy)) <= float(eps_energy):
            stop_reason = "eps_energy"
            break

    if executor is None:
        psi_final = np.array(psi_ref_vec, copy=True)
        layout = empty_layout
        logical_theta = np.zeros(0, dtype=float)
    else:
        psi_final = _normalize_state(executor.prepare_state(theta_runtime, psi_ref_vec))
        logical_theta = (
            project_runtime_theta_block_mean(theta_runtime, layout)
            if str(parameterization_mode) == "per_pauli_term"
            else np.asarray(theta_runtime, dtype=float)
        )

    final_energy, _ = energy_via_one_apply(psi_final, h_compiled)
    if exact_gs_energy is None:
        delta_e = None
        abs_delta_e = None
        exact_gs_value = None
    else:
        exact_gs_value = float(exact_gs_energy)
        delta_e = float(final_energy - exact_gs_value)
        abs_delta_e = float(abs(delta_e))

    payload = {
        "success": True,
        "method": "chemistry_local_adapt_v1",
        "energy": float(final_energy),
        "exact_energy_from_final_state": float(final_energy),
        "exact_gs_energy": exact_gs_value,
        "delta_e": delta_e,
        "abs_delta_e": abs_delta_e,
        "ansatz_depth": int(len(selected_ops)),
        "num_parameters": int(theta_runtime.size),
        "logical_num_parameters": int(layout.logical_parameter_count),
        "optimal_point": [float(x) for x in np.asarray(theta_runtime, dtype=float).tolist()],
        "logical_optimal_point": [float(x) for x in np.asarray(logical_theta, dtype=float).tolist()],
        "parameterization": serialize_layout(layout),
        "operators": [str(op.label) for op in selected_ops],
        "selected_pool_indices": [int(x) for x in selected_indices],
        "pool_size": int(len(pool_list)),
        "pool_type": str(pool_type),
        "stop_reason": str(stop_reason),
        "nfev_total": int(nfev_total),
        "adapt_inner_optimizer": str(optimizer).strip().upper(),
        "parameterization_mode": str(parameterization_mode),
        "history": history,
        "elapsed_sec": float(time.perf_counter() - t0),
        **dict(metadata or {}),
    }
    return LocalAdaptResult(payload=payload, psi_final=psi_final)


def run_local_molecular_adapt_vqe(
    *,
    h_poly: Any,
    n_spatial_orbitals: int,
    num_particles: tuple[int, int],
    exact_gs_energy: float | None,
    max_depth: int,
    eps_grad: float,
    eps_energy: float,
    maxiter: int,
    optimizer: str,
    seed: int,
    ordering: str = "blocked",
    finite_angle: float = 0.05,
    finite_angle_min_improvement: float = 0.0,
) -> LocalAdaptResult:
    if str(ordering).strip().lower() != "blocked":
        raise ValueError("Chemistry-local ADAPT currently supports ordering='blocked' only.")
    psi_ref = _normalize_state(
        np.asarray(
            hartree_fock_statevector(
                int(n_spatial_orbitals),
                tuple(int(x) for x in num_particles),
                indexing="blocked",
            ),
            dtype=complex,
        ).reshape(-1)
    )
    pool = build_molecular_uccsd_pool(
        n_spatial_orbitals=int(n_spatial_orbitals),
        num_particles=tuple(int(x) for x in num_particles),
        ordering="blocked",
    )
    return run_local_adapt_vqe_with_pool(
        h_poly=h_poly,
        psi_ref=psi_ref,
        pool=pool,
        exact_gs_energy=exact_gs_energy,
        max_depth=int(max_depth),
        eps_grad=float(eps_grad),
        eps_energy=float(eps_energy),
        maxiter=int(maxiter),
        optimizer=str(optimizer),
        seed=int(seed),
        pool_type="molecular_uccsd",
        metadata={"num_particles": {"n_up": int(num_particles[0]), "n_dn": int(num_particles[1])}},
        parameterization_mode="per_pauli_term",
        finite_angle=float(finite_angle),
        finite_angle_min_improvement=float(finite_angle_min_improvement),
    )
