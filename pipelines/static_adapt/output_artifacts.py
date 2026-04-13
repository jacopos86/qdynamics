from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from docs.reports.pdf_utils import (
    HAS_MATPLOTLIB,
    get_PdfPages,
    get_plt,
    render_command_page,
    render_text_page,
    require_matplotlib,
)
from docs.reports.report_pages import (
    render_executive_summary_page,
    render_manifest_overview_page,
    render_section_divider_page,
)
from pipelines.hardcoded.handoff_state_bundle import build_statevector_manifest

plt = get_plt() if HAS_MATPLOTLIB else None  # type: ignore[assignment]
PdfPages = get_PdfPages() if HAS_MATPLOTLIB else type("PdfPages", (), {})  # type: ignore[misc]


def _write_pipeline_pdf(pdf_path: Path, payload: dict[str, Any], run_command: str) -> None:
    require_matplotlib()
    settings = payload.get("settings", {})
    adapt = payload.get("adapt_vqe", {})
    problem = settings.get("problem", "hubbard")
    model_name = "Hubbard-Holstein" if problem == "hh" else "Hubbard"

    manifest_sections: list[tuple[str, list[tuple[str, Any]]]] = [
        (
            "Model and regime",
            [
                ("Model family", model_name),
                ("Ansatz type", f"ADAPT-VQE (pool: {settings.get('adapt_pool', '?')})"),
                ("Drive enabled", False),
                ("L", settings.get("L")),
                ("Boundary", settings.get("boundary")),
                ("Ordering", settings.get("ordering")),
            ],
        ),
        (
            "Core physical parameters",
            [
                ("t", settings.get("t")),
                ("U", settings.get("u")),
                ("dv", settings.get("dv")),
            ],
        ),
        (
            "ADAPT controls",
            [
                ("ADAPT max depth", settings.get("adapt_max_depth", "?")),
                ("ADAPT eps_grad", settings.get("adapt_eps_grad", "?")),
                ("ADAPT eps_energy", settings.get("adapt_eps_energy", "?")),
                ("Inner optimizer", settings.get("adapt_inner_optimizer", "?")),
                ("Finite-angle fallback", settings.get("adapt_finite_angle_fallback", "?")),
                ("Finite-angle probe", settings.get("adapt_finite_angle", "?")),
            ],
        ),
        (
            "Trajectory settings",
            [
                ("trotter_steps", settings.get("trotter_steps")),
                ("t_final", settings.get("t_final")),
                ("Suzuki order", settings.get("suzuki_order")),
                ("Initial state source", settings.get("initial_state_source")),
            ],
        ),
    ]
    if problem == "hh":
        manifest_sections.append(
            (
                "Hubbard-Holstein parameters",
                [
                    ("omega0", settings.get("omega0")),
                    ("g_ep", settings.get("g_ep")),
                    ("n_ph_max", settings.get("n_ph_max")),
                    ("Boson encoding", settings.get("boson_encoding")),
                ],
            )
        )
    if str(settings.get("adapt_inner_optimizer", "")).strip().upper() == "SPSA":
        adapt_spsa = settings.get("adapt_spsa", {})
        if isinstance(adapt_spsa, dict):
            manifest_sections.append(
                (
                    "SPSA settings",
                    [
                        ("a", adapt_spsa.get("a")),
                        ("c", adapt_spsa.get("c")),
                        ("A", adapt_spsa.get("A")),
                        ("alpha", adapt_spsa.get("alpha")),
                        ("gamma", adapt_spsa.get("gamma")),
                        ("eval_repeats", adapt_spsa.get("eval_repeats")),
                        ("eval_agg", adapt_spsa.get("eval_agg")),
                        ("avg_last", adapt_spsa.get("avg_last")),
                    ],
                )
            )

    summary_sections: list[tuple[str, list[tuple[str, Any]]]] = [
        (
            "ADAPT outcome",
            [
                ("ADAPT-VQE energy", adapt.get("energy")),
                ("Exact GS energy", adapt.get("exact_gs_energy")),
                ("|ΔE|", adapt.get("abs_delta_e")),
                ("Ansatz depth", adapt.get("ansatz_depth")),
                ("Pool size", adapt.get("pool_size")),
            ],
        ),
        (
            "Optimization summary",
            [
                ("Stop reason", adapt.get("stop_reason")),
                ("Total nfev", adapt.get("nfev_total")),
                ("Elapsed (s)", adapt.get("elapsed_s")),
                ("Inner optimizer", settings.get("adapt_inner_optimizer")),
            ],
        ),
        (
            "Trajectory grid",
            [
                ("trotter_steps", settings.get("trotter_steps")),
                ("t_final", settings.get("t_final")),
                ("Initial state source", settings.get("initial_state_source")),
            ],
        ),
    ]

    operator_lines = [
        "Selected operators",
        "",
        f"Ansatz depth: {adapt.get('ansatz_depth')}",
        f"Pool size: {adapt.get('pool_size')}",
        f"Stop reason: {adapt.get('stop_reason')}",
        "",
    ]
    for op_label in (adapt.get("operators") or []):
        operator_lines.append(f"  {op_label}")

    with PdfPages(str(pdf_path)) as pdf:
        render_manifest_overview_page(
            pdf,
            title=f"{model_name} ADAPT-VQE report — L={settings.get('L')}",
            experiment_statement="ADAPT-VQE state preparation followed by exact-versus-Trotter trajectory diagnostics.",
            sections=manifest_sections,
            notes=[
                "The full operator list and executed command are moved to the appendix.",
            ],
        )
        render_executive_summary_page(
            pdf,
            title="Executive summary",
            experiment_statement="Prepared-state quality and convergence summary before trajectory pages.",
            sections=summary_sections,
            notes=[
                "Trajectory pages show fidelity, energy, occupations, and doublon from the ADAPT state.",
            ],
        )
        render_section_divider_page(
            pdf,
            title="Trajectory diagnostics",
            summary="Main result pages compare exact and Trotter trajectories starting from the ADAPT-prepared state.",
            bullets=[
                "Fidelity and energy.",
                "Site-0 occupations and doublon.",
            ],
        )

        rows = payload.get("trajectory", [])
        if rows:
            times = np.array([r["time"] for r in rows])
            fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), sharex=True)
            ax_f, ax_e = axes[0]
            ax_n, ax_d = axes[1]

            ax_f.plot(times, [r["fidelity"] for r in rows], color="#0b3d91")
            ax_f.set_title("Fidelity (Trotter vs Exact)")
            ax_f.set_ylabel("F(t)")
            ax_f.grid(alpha=0.25)

            ax_e.plot(times, [r["energy_trotter"] for r in rows], label="Trotter", color="#d62728")
            ax_e.plot(times, [r["energy_exact"] for r in rows], label="Exact", color="#111111", linestyle="--")
            ax_e.set_title("Energy")
            ax_e.set_ylabel("E(t)")
            ax_e.legend(fontsize=8)
            ax_e.grid(alpha=0.25)

            ax_n.plot(times, [r["n_up_site0_trotter"] for r in rows], label="n_up trot", color="#17becf")
            ax_n.plot(times, [r["n_dn_site0_trotter"] for r in rows], label="n_dn trot", color="#9467bd")
            ax_n.set_title("Site-0 Occupations (Trotter)")
            ax_n.set_xlabel("Time")
            ax_n.legend(fontsize=8)
            ax_n.grid(alpha=0.25)

            ax_d.plot(times, [r["doublon_trotter"] for r in rows], label="Trotter", color="#e377c2")
            ax_d.plot(times, [r["doublon_exact"] for r in rows], label="Exact", color="#111111", linestyle="--")
            ax_d.set_title("Doublon")
            ax_d.set_xlabel("Time")
            ax_d.legend(fontsize=8)
            ax_d.grid(alpha=0.25)

            fig.suptitle(f"Hardcoded ADAPT-VQE Pipeline L={settings.get('L')}", fontsize=13)
            fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
            pdf.savefig(fig)
            plt.close(fig)

        render_section_divider_page(
            pdf,
            title="Technical appendix",
            summary="Detailed operator provenance and full reproducibility material.",
            bullets=[
                "Selected operator list.",
                "Executed command.",
            ],
        )
        render_text_page(pdf, operator_lines)
        render_command_page(
            pdf,
            run_command,
            script_name="pipelines/hardcoded/adapt_pipeline.py",
        )


def build_output_payload(
    *,
    args: Any,
    cli_adapt_continuation_mode: str,
    adapt_payload: dict[str, Any],
    ordered_labels_exyz: Sequence[str],
    coeff_map_exyz: Mapping[str, complex],
    hmat: np.ndarray | None,
    gs_energy_exact: float,
    gs_energy_source: str,
    psi0: np.ndarray,
    ansatz_input_state_for_adapt: np.ndarray,
    ansatz_input_state_source: str,
    ansatz_input_state_kind: str,
    trajectory: Sequence[Mapping[str, Any]],
    adapt_ref_import: dict[str, Any] | None,
    dense_eigh_enabled: bool,
    hilbert_dim: int,
    adapt_ref_base_depth: int,
    initial_state_source_resolved: str,
    initial_state_kind_resolved: str,
) -> dict[str, Any]:
    if ordered_labels_exyz:
        num_qubits = int(len(ordered_labels_exyz[0]))
    elif hmat is not None:
        num_qubits = int(round(math.log2(hmat.shape[0])))
    else:
        num_qubits = 0

    payload: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "hardcoded_adapt",
        "settings": {
            "L": int(args.L),
            "t": float(args.t),
            "u": float(args.u),
            "problem": str(args.problem),
            "omega0": float(args.omega0),
            "g_ep": float(args.g_ep),
            "n_ph_max": int(args.n_ph_max),
            "boson_encoding": str(args.boson_encoding),
            "dv": float(args.dv),
            "boundary": str(args.boundary),
            "ordering": str(args.ordering),
            "t_final": float(args.t_final),
            "num_times": int(args.num_times),
            "suzuki_order": int(args.suzuki_order),
            "trotter_steps": int(args.trotter_steps),
            "term_order": str(args.term_order),
            "dense_eigh_max_dim": int(args.dense_eigh_max_dim),
            "dense_eigh_enabled": bool(dense_eigh_enabled),
            "hilbert_dim": int(hilbert_dim),
            "adapt_pool": (str(args.adapt_pool) if args.adapt_pool is not None else None),
            "adapt_pool_class_filter_json": (
                str(args.adapt_pool_class_filter_json)
                if args.adapt_pool_class_filter_json is not None
                else None
            ),
            "adapt_pool_label_filter_json": (
                str(args.adapt_pool_label_filter_json)
                if args.adapt_pool_label_filter_json is not None
                else None
            ),
            "adapt_pool_class_filter_classifier_version": (
                adapt_payload.get("adapt_pool_class_filter_classifier_version")
            ),
            "adapt_pool_class_filter_keep_classes": (
                adapt_payload.get("adapt_pool_class_filter_keep_classes")
            ),
            "adapt_pool_label_filter_classifier_version": (
                adapt_payload.get("adapt_pool_label_filter_classifier_version")
            ),
            "adapt_pool_label_filter_drop_labels": (
                adapt_payload.get("adapt_pool_label_filter_drop_labels")
            ),
            "adapt_pool_label_filter_drop_prefixes": (
                adapt_payload.get("adapt_pool_label_filter_drop_prefixes")
            ),
            "adapt_continuation_mode": str(cli_adapt_continuation_mode),
            "adapt_max_depth": int(args.adapt_max_depth),
            "adapt_eps_grad": float(args.adapt_eps_grad),
            "adapt_eps_energy": float(args.adapt_eps_energy),
            "adapt_inner_optimizer": str(args.adapt_inner_optimizer),
            "adapt_state_backend": str(args.adapt_state_backend),
            "adapt_finite_angle_fallback": bool(args.adapt_finite_angle_fallback),
            "adapt_finite_angle": float(args.adapt_finite_angle),
            "adapt_finite_angle_min_improvement": float(args.adapt_finite_angle_min_improvement),
            "adapt_drop_floor": (float(args.adapt_drop_floor) if args.adapt_drop_floor is not None else None),
            "adapt_drop_patience": (int(args.adapt_drop_patience) if args.adapt_drop_patience is not None else None),
            "adapt_drop_min_depth": (int(args.adapt_drop_min_depth) if args.adapt_drop_min_depth is not None else None),
            "adapt_grad_floor": (float(args.adapt_grad_floor) if args.adapt_grad_floor is not None else None),
            "adapt_drop_floor_resolved": adapt_payload.get("adapt_drop_floor_resolved"),
            "adapt_drop_patience_resolved": adapt_payload.get("adapt_drop_patience_resolved"),
            "adapt_drop_min_depth_resolved": adapt_payload.get("adapt_drop_min_depth_resolved"),
            "adapt_grad_floor_resolved": adapt_payload.get("adapt_grad_floor_resolved"),
            "adapt_drop_floor_source": adapt_payload.get("adapt_drop_floor_source"),
            "adapt_drop_patience_source": adapt_payload.get("adapt_drop_patience_source"),
            "adapt_drop_min_depth_source": adapt_payload.get("adapt_drop_min_depth_source"),
            "adapt_grad_floor_source": adapt_payload.get("adapt_grad_floor_source"),
            "adapt_drop_policy_source": adapt_payload.get("adapt_drop_policy_source"),
            "adapt_eps_energy_min_extra_depth": int(args.adapt_eps_energy_min_extra_depth),
            "adapt_eps_energy_patience": int(args.adapt_eps_energy_patience),
            "adapt_ref_base_depth": int(adapt_ref_base_depth),
            "adapt_gradient_parity_check": bool(args.adapt_gradient_parity_check),
            "adapt_analytic_noise_std": float(args.adapt_analytic_noise_std),
            "adapt_analytic_noise_seed": (
                None
                if args.adapt_analytic_noise_seed is None
                else int(args.adapt_analytic_noise_seed)
            ),
            "adapt_seed": int(args.adapt_seed),
            "adapt_reopt_policy": str(args.adapt_reopt_policy),
            "adapt_window_size": int(args.adapt_window_size),
            "adapt_window_topk": int(args.adapt_window_topk),
            "adapt_full_refit_every": int(args.adapt_full_refit_every),
            "adapt_final_full_refit": str(args.adapt_final_full_refit),
            "phase1_lambda_F": float(args.phase1_lambda_F),
            "phase1_lambda_compile": float(args.phase1_lambda_compile),
            "phase1_lambda_measure": float(args.phase1_lambda_measure),
            "phase1_lambda_leak": float(args.phase1_lambda_leak),
            "phase1_score_z_alpha": float(args.phase1_score_z_alpha),
            "phase1_depth_ref": float(args.phase1_depth_ref),
            "phase1_group_ref": float(args.phase1_group_ref),
            "phase1_shot_ref": float(args.phase1_shot_ref),
            "phase1_family_ref": float(args.phase1_family_ref),
            "phase1_compile_cx_proxy_weight": float(args.phase1_compile_cx_proxy_weight),
            "phase1_compile_sq_proxy_weight": float(args.phase1_compile_sq_proxy_weight),
            "phase1_compile_rotation_step_weight": float(args.phase1_compile_rotation_step_weight),
            "phase1_compile_position_shift_weight": float(args.phase1_compile_position_shift_weight),
            "phase1_compile_refit_active_weight": float(args.phase1_compile_refit_active_weight),
            "phase1_measure_groups_weight": float(args.phase1_measure_groups_weight),
            "phase1_measure_shots_weight": float(args.phase1_measure_shots_weight),
            "phase1_measure_reuse_weight": float(args.phase1_measure_reuse_weight),
            "phase1_opt_dim_cost_scale": float(args.phase1_opt_dim_cost_scale),
            "phase1_family_repeat_cost_scale": float(args.phase1_family_repeat_cost_scale),
            "phase1_shortlist_size": int(args.phase1_shortlist_size),
            "phase1_probe_max_positions": int(args.phase1_probe_max_positions),
            "phase1_plateau_patience": int(args.phase1_plateau_patience),
            "phase1_trough_margin_ratio": float(args.phase1_trough_margin_ratio),
            "phase1_prune_enabled": bool(args.phase1_prune_enabled),
            "phase1_prune_mode": str(args.phase1_prune_mode),
            "phase1_prune_fraction": float(args.phase1_prune_fraction),
            "phase1_prune_min_candidates": int(args.phase1_prune_min_candidates),
            "phase1_prune_max_candidates": int(args.phase1_prune_max_candidates),
            "phase1_prune_max_regression": float(args.phase1_prune_max_regression),
            "phase1_prune_retained_gain_ratio": float(args.phase1_prune_retained_gain_ratio),
            "phase1_prune_protect_steps": int(args.phase1_prune_protect_steps),
            "phase1_prune_stale_age": int(args.phase1_prune_stale_age),
            "phase1_prune_stagnation_threshold": float(args.phase1_prune_stagnation_threshold),
            "phase1_prune_small_theta_abs": float(args.phase1_prune_small_theta_abs),
            "phase1_prune_small_theta_relative": float(args.phase1_prune_small_theta_relative),
            "phase1_prune_cooldown_steps": int(args.phase1_prune_cooldown_steps),
            "phase1_prune_local_window_size": int(args.phase1_prune_local_window_size),
            "phase1_prune_old_fraction": float(args.phase1_prune_old_fraction),
            "phase1_prune_checkpoint_period": int(args.phase1_prune_checkpoint_period),
            "phase1_prune_maturity_threshold": float(args.phase1_prune_maturity_threshold),
            "phase1_prune_snr_threshold": float(args.phase1_prune_snr_threshold),
            "phase2_shortlist_fraction": float(args.phase2_shortlist_fraction),
            "phase2_shortlist_size": int(args.phase2_shortlist_size),
            "phase2_lambda_H": float(args.phase2_lambda_H),
            "phase2_rho": float(args.phase2_rho),
            "phase2_gamma_N": float(args.phase2_gamma_N),
            "phase2_score_z_alpha": (
                float(args.phase2_score_z_alpha)
                if args.phase2_score_z_alpha is not None
                else None
            ),
            "phase2_lambda_F": (
                float(args.phase2_lambda_F)
                if args.phase2_lambda_F is not None
                else None
            ),
            "phase2_depth_ref": float(args.phase2_depth_ref),
            "phase2_group_ref": float(args.phase2_group_ref),
            "phase2_shot_ref": float(args.phase2_shot_ref),
            "phase2_optdim_ref": float(args.phase2_optdim_ref),
            "phase2_reuse_ref": float(args.phase2_reuse_ref),
            "phase2_family_ref": float(args.phase2_family_ref),
            "phase2_novelty_eps": float(args.phase2_novelty_eps),
            "phase2_cheap_score_eps": float(args.phase2_cheap_score_eps),
            "phase2_metric_floor": float(args.phase2_metric_floor),
            "phase2_reduced_metric_collapse_rel_tol": float(
                args.phase2_reduced_metric_collapse_rel_tol
            ),
            "phase2_ridge_growth_factor": float(args.phase2_ridge_growth_factor),
            "phase2_ridge_max_steps": int(args.phase2_ridge_max_steps),
            "phase2_leakage_cap": float(args.phase2_leakage_cap),
            "phase2_compile_cx_proxy_weight": float(args.phase2_compile_cx_proxy_weight),
            "phase2_compile_sq_proxy_weight": float(args.phase2_compile_sq_proxy_weight),
            "phase2_compile_rotation_step_weight": float(args.phase2_compile_rotation_step_weight),
            "phase2_compile_position_shift_weight": float(args.phase2_compile_position_shift_weight),
            "phase2_compile_refit_active_weight": float(args.phase2_compile_refit_active_weight),
            "phase2_measure_groups_weight": float(args.phase2_measure_groups_weight),
            "phase2_measure_shots_weight": float(args.phase2_measure_shots_weight),
            "phase2_measure_reuse_weight": float(args.phase2_measure_reuse_weight),
            "phase2_opt_dim_cost_scale": float(args.phase2_opt_dim_cost_scale),
            "phase2_family_repeat_cost_scale": float(args.phase2_family_repeat_cost_scale),
            "phase2_w_depth": float(args.phase2_w_depth),
            "phase2_w_group": float(args.phase2_w_group),
            "phase2_w_shot": float(args.phase2_w_shot),
            "phase2_w_optdim": float(args.phase2_w_optdim),
            "phase2_w_reuse": float(args.phase2_w_reuse),
            "phase2_w_lifetime": float(args.phase2_w_lifetime),
            "phase2_eta_L": float(args.phase2_eta_L),
            "phase2_motif_bonus_weight": float(args.phase2_motif_bonus_weight),
            "phase2_duplicate_penalty_weight": float(args.phase2_duplicate_penalty_weight),
            "phase2_frontier_ratio": float(args.phase2_frontier_ratio),
            "phase3_frontier_ratio": float(args.phase3_frontier_ratio),
            "phase3_tie_beam_score_ratio": float(args.phase3_tie_beam_score_ratio),
            "phase3_tie_beam_abs_tol": float(args.phase3_tie_beam_abs_tol),
            "phase3_tie_beam_max_branches": int(args.phase3_tie_beam_max_branches),
            "phase3_tie_beam_max_late_coordinate": float(args.phase3_tie_beam_max_late_coordinate),
            "phase3_tie_beam_min_depth_left": int(args.phase3_tie_beam_min_depth_left),
            "phase2_enable_batching": bool(args.phase2_enable_batching),
            "phase2_batch_target_size": int(args.phase2_batch_target_size),
            "phase2_batch_size_cap": int(args.phase2_batch_size_cap),
            "phase2_batch_near_degenerate_ratio": float(args.phase2_batch_near_degenerate_ratio),
            "phase2_batch_rank_rel_tol": float(args.phase2_batch_rank_rel_tol),
            "phase2_batch_additivity_tol": float(args.phase2_batch_additivity_tol),
            "phase2_compat_overlap_weight": float(args.phase2_compat_overlap_weight),
            "phase2_compat_comm_weight": float(args.phase2_compat_comm_weight),
            "phase2_compat_curv_weight": float(args.phase2_compat_curv_weight),
            "phase2_compat_sched_weight": float(args.phase2_compat_sched_weight),
            "phase2_compat_measure_weight": float(args.phase2_compat_measure_weight),
            "phase2_remaining_evaluations_proxy_mode": str(
                args.phase2_remaining_evaluations_proxy_mode
            ),
            "phase3_motif_source_json": (
                str(args.phase3_motif_source_json)
                if args.phase3_motif_source_json is not None
                else None
            ),
            "phase3_symmetry_mitigation_mode": str(args.phase3_symmetry_mitigation_mode),
            "phase3_enable_rescue": bool(args.phase3_enable_rescue),
            "phase3_lifetime_cost_mode": str(args.phase3_lifetime_cost_mode),
            "phase3_runtime_split_mode": str(args.phase3_runtime_split_mode),
            "phase3_selector_geometry_mode": str(args.phase3_selector_geometry_mode),
            "phase3_backend_cost_mode": str(args.phase3_backend_cost_mode),
            "phase3_backend_name": (
                None if args.phase3_backend_name in {None, ""} else str(args.phase3_backend_name)
            ),
            "phase3_backend_shortlist": (
                []
                if args.phase3_backend_shortlist in {None, ""}
                else [str(tok).strip() for tok in str(args.phase3_backend_shortlist).split(",") if str(tok).strip() != ""]
            ),
            "phase3_backend_transpile_seed": int(args.phase3_backend_transpile_seed),
            "phase3_backend_optimization_level": int(args.phase3_backend_optimization_level),
            "phase3_oracle_inner_objective_mode": str(
                adapt_payload.get(
                    "phase3_oracle_inner_objective_mode",
                    args.phase3_oracle_inner_objective_mode,
                )
            ),
            "phase3_oracle_inner_objective_mode_requested": str(
                adapt_payload.get(
                    "phase3_oracle_inner_objective_mode_requested",
                    args.phase3_oracle_inner_objective_mode,
                )
            ),
            "phase3_oracle_inner_objective_runtime_guard_reason": (
                adapt_payload.get("phase3_oracle_inner_objective_runtime_guard_reason")
            ),
            "adapt_ref_json": (str(args.adapt_ref_json) if args.adapt_ref_json is not None else None),
            "paop_r": int(args.paop_r),
            "paop_split_paulis": bool(args.paop_split_paulis),
            "paop_prune_eps": float(args.paop_prune_eps),
            "paop_normalization": str(args.paop_normalization),
            "initial_state_source": str(args.initial_state_source),
        },
        "hamiltonian": {
            "num_qubits": int(num_qubits),
            "num_terms": int(len(coeff_map_exyz)),
            "coefficients_exyz": [
                {
                    "label_exyz": lbl,
                    "coeff": {"re": float(np.real(coeff_map_exyz[lbl])), "im": float(np.imag(coeff_map_exyz[lbl]))},
                }
                for lbl in ordered_labels_exyz
            ],
        },
        "ground_state": {
            "exact_energy": float(gs_energy_exact),
            "exact_energy_source": str(gs_energy_source),
            "method": ("dense_eigh" if hmat is not None else "sector_exact_only_no_dense_eigh"),
        },
        "adapt_vqe": adapt_payload,
        "initial_state": build_statevector_manifest(
            psi_state=np.asarray(psi0, dtype=complex).reshape(-1),
            source=initial_state_source_resolved,
            handoff_state_kind=initial_state_kind_resolved,
            amplitude_cutoff=1e-12,
        ),
        "ansatz_input_state": build_statevector_manifest(
            psi_state=np.asarray(ansatz_input_state_for_adapt, dtype=complex).reshape(-1),
            source=str(ansatz_input_state_source),
            handoff_state_kind=ansatz_input_state_kind,
            amplitude_cutoff=1e-12,
        ),
        "trajectory": list(trajectory),
    }
    if str(args.adapt_inner_optimizer).strip().upper() == "SPSA":
        payload["settings"]["adapt_spsa"] = {
            "a": float(args.adapt_spsa_a),
            "c": float(args.adapt_spsa_c),
            "alpha": float(args.adapt_spsa_alpha),
            "gamma": float(args.adapt_spsa_gamma),
            "A": float(args.adapt_spsa_A),
            "avg_last": int(args.adapt_spsa_avg_last),
            "eval_repeats": int(args.adapt_spsa_eval_repeats),
            "eval_agg": str(args.adapt_spsa_eval_agg),
            "callback_every": int(args.adapt_spsa_callback_every),
            "progress_every_s": float(args.adapt_spsa_progress_every_s),
        }
    if adapt_ref_import is not None:
        adapt_ref_import["ansatz_input_state_persisted"] = True
        payload["adapt_ref_import"] = adapt_ref_import
    return payload


def persist_output_artifacts(
    *,
    output_json: Path,
    output_pdf: Path,
    payload: Mapping[str, Any],
    run_command: str,
    skip_pdf: bool,
    ai_log: Callable[..., None] | None = None,
    safe_stdout_print: Callable[..., bool] | None = None,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not bool(skip_pdf):
        _write_pipeline_pdf(output_pdf, dict(payload), run_command)

    if ai_log is not None:
        settings = payload.get("settings", {}) if isinstance(payload, Mapping) else {}
        adapt_payload = payload.get("adapt_vqe", {}) if isinstance(payload, Mapping) else {}
        ai_log(
            "hardcoded_adapt_main_done",
            L=int(settings.get("L", 0) or 0),
            output_json=str(output_json),
            output_pdf=(str(output_pdf) if not bool(skip_pdf) else None),
            adapt_energy=(
                adapt_payload.get("energy")
                if isinstance(adapt_payload, Mapping)
                else None
            ),
        )

    if safe_stdout_print is not None:
        safe_stdout_print(f"Wrote JSON: {output_json}")
        if not bool(skip_pdf):
            safe_stdout_print(f"Wrote PDF:  {output_pdf}")
