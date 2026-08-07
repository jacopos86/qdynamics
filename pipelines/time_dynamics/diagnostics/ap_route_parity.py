#!/usr/bin/env python3
"""Compare AP fixed McLachlan against the legacy fixed route at one seed.

This module is diagnostic-only.  It does not run an adaptive controller and it
does not use exact/reference data for any decision.  Its purpose is to expose
whether the AP route and legacy fixed route are evaluating the same support,
Hamiltonian, geometry, and solve convention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.scaffold.runtime_loader import load_scaffold_runtime_input
from pipelines.time_dynamics.ap_mclachlan.geometry_eval import evaluate_mclachlan_geometry
from pipelines.time_dynamics.ap_mclachlan.hamiltonian import (
    time_dependent_hamiltonian_from_runtime_input,
)
from pipelines.time_dynamics.ap_mclachlan.inverse import (
    McLachlanInversePolicy,
    solve_theta_dot,
)
from pipelines.time_dynamics.ap_mclachlan.state import (
    APMcLachlanState,
    runtime_coordinate_labels,
    state_from_scaffold_runtime_input,
    state_with_appended_terms,
)
from pipelines.time_dynamics.adapters.hh import HH_REALTIME_ADAPTER
from pipelines.time_dynamics.runners import generic_from_adapt_artifact
from pipelines.time_dynamics.tables.dynamics_benchmark_contract import json_safe


SCHEMA_V1 = "ap_mclachlan_route_parity_audit_v1"


def build_route_parity_audit(args: argparse.Namespace) -> dict[str, Any]:
    artifact_json = Path(args.artifact_json).expanduser().resolve()
    runtime_input = load_scaffold_runtime_input(
        artifact_json,
        loader_mode=str(args.loader_mode),
        tag=str(args.run_tag),
        generator_family=str(args.generator_family),
        fallback_family=str(args.fallback_family),
    )
    ap_state = state_from_scaffold_runtime_input(runtime_input)
    request = runtime_input.resolved_problem.request
    drive_config = HH_REALTIME_ADAPTER.build_drive_config(
        args,
        n_sites=int(request.num_sites),
        ordering=str(request.ordering),
    )
    hamiltonian = time_dependent_hamiltonian_from_runtime_input(
        runtime_input,
        drive_config=drive_config,
    )

    legacy_args = _legacy_fixed_args(args, output_json=Path(args.output_json).with_suffix(".legacy_probe.json"))
    legacy_bundle = generic_from_adapt_artifact.build_controller_bundle_from_args(legacy_args)
    legacy_controller = legacy_bundle["controller"]
    legacy_replay_context = legacy_controller.replay_context
    legacy_terms = tuple(getattr(legacy_replay_context, "replay_terms", ()) or ())
    ap_labels = tuple(str(getattr(term, "label", "")) for term in ap_state.terms)
    legacy_labels = tuple(str(getattr(term, "label", "")) for term in legacy_terms)
    extra_legacy_terms = tuple(
        term for term in legacy_terms if str(getattr(term, "label", "")) not in set(ap_labels)
    )
    augmented_state = (
        state_with_appended_terms(ap_state, extra_legacy_terms)
        if extra_legacy_terms
        else ap_state
    )

    probe_times = _probe_times(args)
    ap_prepared = ap_state.prepare_state()
    seed_block = {
        "artifact_json": str(artifact_json),
        "problem_family": str(runtime_input.resolved_problem.family_key),
        "structure_locked": bool(getattr(runtime_input, "structure_locked", False)),
        "candidate_pool_complete": bool(
            getattr(getattr(runtime_input, "candidate_pool_source", None), "candidate_pool_complete", False)
        ),
        "ap_can_structural_edit": bool(ap_state.can_structural_edit),
        "ap_logical_parameter_count": int(ap_state.logical_parameter_count),
        "ap_runtime_parameter_count": int(ap_state.runtime_parameter_count),
        "ap_selected_term_labels": [str(label) for label in ap_labels],
        "ap_runtime_coordinate_labels": [str(label) for label in ap_state.runtime_coordinate_labels],
        "theta_runtime_l2": float(np.linalg.norm(np.asarray(ap_state.theta_runtime, dtype=float))),
        "psi_ref_sha256": _array_sha256(ap_state.psi_ref),
        "psi_initial_sha256": _array_sha256(ap_state.psi_initial),
        "ap_prepared_state_sha256": _array_sha256(ap_prepared),
        "prepared_vs_psi_initial_l2": float(
            np.linalg.norm(np.asarray(ap_prepared, dtype=complex) - np.asarray(ap_state.psi_initial, dtype=complex))
        ),
    }
    legacy_block = {
        "mode": str(getattr(legacy_controller.cfg, "mode", "")),
        "integrator_policy": str(getattr(legacy_controller.cfg, "integrator_policy", "")),
        "regularization_lambda": float(getattr(legacy_controller.cfg, "regularization_lambda", 0.0)),
        "pinv_rcond": float(getattr(legacy_controller.cfg, "pinv_rcond", 0.0)),
        "drive_aligned_density_active": bool(
            getattr(legacy_controller, "_drive_aligned_density_active", False)
        ),
        "drive_aligned_density_label": getattr(legacy_controller, "_drive_aligned_density_label", None),
        "logical_parameter_count": int(getattr(legacy_controller.current_layout, "logical_parameter_count", 0)),
        "runtime_parameter_count": int(getattr(legacy_controller.current_layout, "runtime_parameter_count", 0)),
        "replay_term_labels": [str(label) for label in legacy_labels],
        "runtime_coordinate_labels": [
            str(label) for label in runtime_coordinate_labels(legacy_controller.current_layout)
        ],
        "extra_legacy_labels_not_in_ap": [str(getattr(term, "label", "")) for term in extra_legacy_terms],
        "support_matches_ap": bool(not extra_legacy_terms and legacy_labels == ap_labels),
        "current_theta_l2": float(np.linalg.norm(np.asarray(legacy_controller.current_theta, dtype=float))),
    }

    geometry_rows: list[dict[str, Any]] = []
    for time in probe_times:
        geometry_rows.append(
            _geometry_probe(
                support_kind="ap_seed",
                state=ap_state,
                hamiltonian=hamiltonian,
                time=float(time),
            )
        )
        if augmented_state is not ap_state:
            geometry_rows.append(
                _geometry_probe(
                    support_kind="ap_plus_legacy_extra",
                    state=augmented_state,
                    hamiltonian=hamiltonian,
                    time=float(time),
                )
            )

    payload = {
        "schema": SCHEMA_V1,
        "diagnostic_only": True,
        "decision_data_flow": "prepared_state_geometry_probe_only_no_exact_reference_inputs",
        "case": {
            "artifact_json": str(artifact_json),
            "run_tag": str(args.run_tag),
            "t_final": float(args.t_final),
            "num_times": int(args.num_times),
            "drive": _drive_payload(drive_config),
        },
        "seed_state": seed_block,
        "legacy_post_init": legacy_block,
        "hamiltonian": hamiltonian.to_json_dict(),
        "geometry": {
            "probe_times": [float(t) for t in probe_times],
            "rows": geometry_rows,
        },
        "classification": _classify(seed_block, legacy_block, geometry_rows),
    }
    return json_safe(payload)


def _legacy_fixed_args(args: argparse.Namespace, *, output_json: Path) -> argparse.Namespace:
    parser = generic_from_adapt_artifact.build_parser()
    argv = [
        "--artifact-json",
        str(args.artifact_json),
        "--output-json",
        str(output_json),
        "--run-tag",
        f"{args.run_tag}_legacy_fixed_probe",
        "--loader-mode",
        str(args.loader_mode),
        "--generator-family",
        str(args.generator_family),
        "--fallback-family",
        str(args.fallback_family),
        "--append-pool-family",
        str(args.append_pool_family),
        "--lock-fixed-manifold",
        "--t-final",
        str(float(args.t_final)),
        "--num-times",
        str(int(args.num_times)),
        "--checkpoint-controller-mode",
        "observable_v1",
        "--checkpoint-controller-exact-input-mode",
        "off",
        "--diagnostic-exact-reference-mode",
        "off",
        "--checkpoint-controller-noise-mode",
        "ideal",
        "--checkpoint-controller-strict-qpu-faithful",
        "--checkpoint-controller-integrator-policy",
        "rk4",
        "--no-checkpoint-controller-append-enabled",
        "--compile-audit-mode",
        "off",
    ]
    if bool(args.enable_drive):
        argv.extend(
            [
                "--enable-drive",
                "--drive-A",
                str(float(args.drive_A)),
                "--drive-omega",
                str(float(args.drive_omega)),
                "--drive-tbar",
                str(float(args.drive_tbar)),
                "--drive-phi",
                str(float(args.drive_phi)),
                "--drive-pattern",
                str(args.drive_pattern),
                "--drive-time-sampling",
                str(args.drive_time_sampling),
                "--drive-t0",
                str(float(args.drive_t0)),
            ]
        )
        if str(args.drive_custom_weights):
            argv.extend(["--drive-custom-weights", str(args.drive_custom_weights)])
        if bool(args.drive_include_identity):
            argv.append("--drive-include-identity")
    return parser.parse_args(argv)


def _geometry_probe(
    *,
    support_kind: str,
    state: APMcLachlanState,
    hamiltonian: Any,
    time: float,
) -> dict[str, Any]:
    evaluation = evaluate_mclachlan_geometry(
        state=state,
        hamiltonian=hamiltonian,
        time=float(time),
        metadata={"support_kind": str(support_kind)},
    )
    geometry = evaluation.geometry
    K = np.asarray(geometry.K, dtype=float)
    f = np.asarray(geometry.f, dtype=float).reshape(-1)
    default_solve = solve_theta_dot(K, f, policy=McLachlanInversePolicy())
    ridge_solve = solve_theta_dot(
        K,
        f,
        policy=McLachlanInversePolicy(ridge_lambda=1.0e-8),
    )
    return {
        "support_kind": str(support_kind),
        "time": float(time),
        "drive_coefficient": float(hamiltonian.drive_coefficient_at(float(time))),
        "runtime_parameter_count": int(state.runtime_parameter_count),
        "logical_parameter_count": int(state.logical_parameter_count),
        "selected_term_labels": [str(getattr(term, "label", "")) for term in state.terms],
        "runtime_coordinate_labels": [str(label) for label in state.runtime_coordinate_labels],
        "energy_expectation": float(evaluation.energy_expectation),
        "norm_b_sq": float(geometry.norm_b_sq),
        "K_shape": [int(x) for x in K.shape],
        "K_fro_norm": float(np.linalg.norm(K)),
        "K_rank_matrix": int(np.linalg.matrix_rank(K, tol=1.0e-10)) if K.size else 0,
        "f_l2": float(np.linalg.norm(f)),
        "f_linf": float(np.max(np.abs(f))) if f.size else 0.0,
        "f_is_zero": bool((np.linalg.norm(f) <= 1.0e-12) and ((np.max(np.abs(f)) if f.size else 0.0) <= 1.0e-12)),
        "solve_default": _solve_summary(default_solve),
        "solve_ridge_1e_minus_8": _solve_summary(ridge_solve),
    }


def _solve_summary(solve: Any) -> dict[str, Any]:
    theta_dot = np.asarray(solve.theta_dot, dtype=float).reshape(-1)
    return {
        "theta_dot_l2": float(np.linalg.norm(theta_dot)),
        "theta_dot_linf": float(np.max(np.abs(theta_dot))) if theta_dot.size else 0.0,
        "gamma": float(solve.gamma),
        "rank_retained": int(solve.inverse.rank),
        "condition_number": solve.inverse.condition_number,
        "ridge_lambda": float(solve.inverse.ridge_lambda),
        "pinv_rcond": float(solve.inverse.pinv_rcond),
    }


def _probe_times(args: argparse.Namespace) -> tuple[float, ...]:
    if args.probe_time:
        return tuple(float(value) for value in args.probe_time)
    dt = float(args.t_final) / max(int(args.num_times) - 1, 1)
    return (0.0, 0.5 * dt, dt)


def _drive_payload(drive_config: Any) -> dict[str, Any]:
    if drive_config is None:
        return {"enabled": False}
    return {
        "enabled": bool(getattr(drive_config, "enabled", False)),
        "drive_A": float(getattr(drive_config, "drive_A", 0.0)),
        "drive_omega": float(getattr(drive_config, "drive_omega", 0.0)),
        "drive_tbar": float(getattr(drive_config, "drive_tbar", 0.0)),
        "drive_phi": float(getattr(drive_config, "drive_phi", 0.0)),
        "drive_pattern": str(getattr(drive_config, "drive_pattern", "")),
        "drive_time_sampling": str(getattr(drive_config, "drive_time_sampling", "")),
        "drive_t0": float(getattr(drive_config, "drive_t0", 0.0)),
    }


def _classify(
    seed_block: Mapping[str, Any],
    legacy_block: Mapping[str, Any],
    geometry_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    seed_ok = float(seed_block.get("prepared_vs_psi_initial_l2", float("inf"))) <= 1.0e-10
    support_mismatch = not bool(legacy_block.get("support_matches_ap", False))
    ap_rows = [row for row in geometry_rows if row.get("support_kind") == "ap_seed"]
    aug_rows = [row for row in geometry_rows if row.get("support_kind") == "ap_plus_legacy_extra"]
    ap_zero = bool(ap_rows) and all(bool(row.get("f_is_zero", False)) for row in ap_rows)
    aug_nonzero = bool(aug_rows) and any(float(row.get("f_l2", 0.0)) > 1.0e-12 for row in aug_rows)
    primary = "undetermined"
    if not seed_ok:
        primary = "seed_state_reconstruction_mismatch"
    elif support_mismatch and ap_zero and aug_nonzero:
        primary = "support_mismatch_legacy_drive_aligned_density_augmentation"
    elif support_mismatch:
        primary = "support_mismatch_without_confirmed_force_recovery"
    elif ap_zero:
        primary = "ap_seed_support_force_zero"
    return {
        "primary": primary,
        "seed_state_passed": bool(seed_ok),
        "legacy_support_matches_ap": bool(not support_mismatch),
        "ap_seed_force_zero_all_probes": bool(ap_zero),
        "augmented_support_force_nonzero_any_probe": bool(aug_nonzero),
        "production_change_recommended": False,
    }


def _array_sha256(value: Any) -> str:
    arr = np.asarray(value)
    contiguous = np.ascontiguousarray(arr)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnostic AP-vs-legacy fixed McLachlan route parity probe.")
    parser.add_argument("--artifact-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--run-tag", default="ap_route_parity")
    parser.add_argument("--loader-mode", default="replay_family")
    parser.add_argument("--generator-family", default="match_adapt")
    parser.add_argument("--fallback-family", default="full_meta")
    parser.add_argument("--append-pool-family", default="match_replay")
    parser.add_argument("--t-final", type=float, default=5.0)
    parser.add_argument("--num-times", type=int, default=501)
    parser.add_argument("--probe-time", action="append", type=float, default=[])
    parser.add_argument("--enable-drive", action="store_true")
    parser.add_argument("--drive-A", type=float, default=0.0)
    parser.add_argument("--drive-omega", type=float, default=1.0)
    parser.add_argument("--drive-tbar", type=float, default=1.0)
    parser.add_argument("--drive-phi", type=float, default=0.0)
    parser.add_argument("--drive-pattern", default="staggered")
    parser.add_argument("--drive-custom-weights", default="")
    parser.add_argument("--drive-include-identity", action="store_true")
    parser.add_argument("--drive-time-sampling", default="midpoint")
    parser.add_argument("--drive-t0", type=float, default=0.0)
    parser.add_argument("--exact-steps-multiplier", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = build_route_parity_audit(args)
    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload.get("classification", {}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCHEMA_V1",
    "build_parser",
    "build_route_parity_audit",
    "main",
]
