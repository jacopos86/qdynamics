from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.hardcoded.hh_fixed_manifold_mclachlan import (
    FixedManifoldRunSpec,
    load_run_context,
)
from pipelines.hardcoded.hh_realtime_checkpoint_controller import (
    ControllerDriveConfig,
    RealtimeCheckpointController,
)
from pipelines.hardcoded.hh_realtime_checkpoint_types import RealtimeCheckpointConfig
from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix


def _parse_float_tuple(raw: str | None) -> tuple[float, ...]:
    if raw is None:
        return ()
    text = str(raw).strip()
    if not text:
        return ()
    return tuple(float(chunk.strip()) for chunk in text.split(",") if chunk.strip())


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_jsonable(item) for item in value]
    return value


def build_controller_config(args: argparse.Namespace) -> RealtimeCheckpointConfig:
    return RealtimeCheckpointConfig(
        mode=str(args.checkpoint_controller_mode),
        miss_threshold=float(args.checkpoint_controller_miss_threshold),
        gain_ratio_threshold=float(args.checkpoint_controller_gain_ratio_threshold),
        append_margin_abs=float(args.checkpoint_controller_append_margin_abs),
        candidate_step_scales=_parse_float_tuple(args.checkpoint_controller_candidate_step_scales),
        exact_forecast_baseline_step_refine_rounds=int(
            args.checkpoint_controller_exact_forecast_baseline_step_refine_rounds
        ),
        exact_forecast_baseline_blend_weights=_parse_float_tuple(
            args.checkpoint_controller_exact_forecast_baseline_blend_weights
        ),
        exact_forecast_baseline_gain_scales=_parse_float_tuple(
            args.checkpoint_controller_exact_forecast_baseline_gain_scales
        ),
        exact_forecast_include_tangent_secant_proposal=bool(
            args.checkpoint_controller_exact_forecast_include_tangent_secant_proposal
        ),
        exact_forecast_tangent_secant_trust_radius=float(
            args.checkpoint_controller_exact_forecast_tangent_secant_trust_radius
        ),
        exact_forecast_tangent_secant_signed_energy_lead_limit=float(
            args.checkpoint_controller_exact_forecast_tangent_secant_signed_energy_lead_limit
        ),
        exact_forecast_tracking_horizon_steps=int(
            args.checkpoint_controller_exact_forecast_horizon_steps
        ),
        exact_forecast_tracking_horizon_weights=_parse_float_tuple(
            args.checkpoint_controller_exact_forecast_horizon_weights
        ),
        exact_forecast_tracking_fidelity_defect_weight=float(
            args.checkpoint_controller_exact_forecast_tracking_fidelity_defect_weight
        ),
        exact_forecast_tracking_staggered_error_weight=float(
            args.checkpoint_controller_exact_forecast_tracking_staggered_error_weight
        ),
        exact_forecast_tracking_doublon_error_weight=float(
            args.checkpoint_controller_exact_forecast_tracking_doublon_error_weight
        ),
        exact_forecast_tracking_site_occupations_error_weight=float(
            args.checkpoint_controller_exact_forecast_tracking_site_occupations_error_weight
        ),
        exact_forecast_tracking_energy_total_error_weight=float(
            args.checkpoint_controller_exact_forecast_tracking_energy_total_error_weight
        ),
        exact_forecast_energy_slope_weight=float(
            args.checkpoint_controller_exact_forecast_energy_slope_weight
        ),
        exact_forecast_energy_curvature_weight=float(
            args.checkpoint_controller_exact_forecast_energy_curvature_weight
        ),
        exact_forecast_energy_excursion_under_weight=float(
            args.checkpoint_controller_exact_forecast_energy_excursion_under_weight
        ),
        exact_forecast_energy_excursion_over_weight=float(
            args.checkpoint_controller_exact_forecast_energy_excursion_over_weight
        ),
        exact_forecast_energy_excursion_rel_tolerance=float(
            args.checkpoint_controller_exact_forecast_energy_excursion_rel_tolerance
        ),
    )


def build_drive_config(
    args: argparse.Namespace,
    *,
    n_sites: int,
    ordering: str,
) -> ControllerDriveConfig | None:
    if bool(args.disable_drive):
        return None
    return ControllerDriveConfig(
        enabled=True,
        n_sites=int(n_sites),
        ordering=str(ordering),
        drive_A=float(args.drive_A),
        drive_omega=float(args.drive_omega),
        drive_tbar=float(args.drive_tbar),
        drive_phi=float(args.drive_phi),
        drive_pattern=str(args.drive_pattern),
        drive_custom_weights=_parse_float_tuple(args.drive_custom_weights) or None,
        drive_include_identity=bool(args.drive_include_identity),
        drive_time_sampling=str(args.drive_time_sampling),
        drive_t0=float(args.drive_t0),
        exact_steps_multiplier=int(args.exact_steps_multiplier),
    )


def build_output_payload(
    *,
    args: argparse.Namespace,
    loaded: Any,
    cfg: RealtimeCheckpointConfig,
    drive_config: ControllerDriveConfig | None,
    result: Any,
) -> dict[str, Any]:
    replay_context = loaded.replay_context
    return {
        "run_tag": str(args.run_tag),
        "artifact_json": str(Path(args.artifact_json).resolve()),
        "loader_mode": str(args.loader_mode),
        "loader_summary": {
            "generator_family": str(args.generator_family),
            "fallback_family": str(args.fallback_family),
            "resolved_family": str(getattr(replay_context, "family_info", {}).get("resolved", "unknown")),
            "handoff_state_kind": str(getattr(replay_context, "handoff_state_kind", "unknown")),
            "family_terms_count": int(getattr(replay_context, "family_terms_count", 0)),
            "adapt_depth": int(getattr(replay_context, "adapt_depth", 0)),
        },
        "controller_config": _to_jsonable(cfg),
        "drive_config": _to_jsonable(drive_config),
        "summary": _to_jsonable(dict(result.summary)),
        "trajectory": _to_jsonable([dict(row) for row in result.trajectory]),
        "ledger": _to_jsonable([dict(row) for row in result.ledger]),
    }


"Built Math: H_mat = matrix(H_poly), theta*(t) = Controller(H_mat, psi_0, theta_0), payload = {summary, trajectory, ledger}."
def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    artifact_json = Path(args.artifact_json).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    spec = FixedManifoldRunSpec(
        name=str(args.run_tag),
        artifact_json=artifact_json,
        loader_mode=str(args.loader_mode),
        generator_family=str(args.generator_family),
        fallback_family=str(args.fallback_family),
    )
    loaded = load_run_context(
        spec,
        tag=str(args.run_tag),
        lock_fixed_manifold=bool(args.lock_fixed_manifold),
    )
    cfg = build_controller_config(args)
    n_sites = int(
        getattr(
            getattr(loaded, "cfg", None),
            "L",
            getattr(getattr(loaded.replay_context, "cfg", None), "L", 1),
        )
    )
    ordering = str(
        getattr(
            getattr(loaded, "cfg", None),
            "ordering",
            getattr(getattr(loaded.replay_context, "cfg", None), "ordering", "blocked"),
        )
    )
    drive_config = build_drive_config(args, n_sites=n_sites, ordering=ordering)
    replay_context = loaded.replay_context
    h_poly = replay_context.h_poly
    hmat = np.asarray(hamiltonian_matrix(h_poly), dtype=complex)
    controller = RealtimeCheckpointController(
        cfg=cfg,
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=np.asarray(loaded.psi_initial, dtype=complex),
        best_theta=np.asarray(replay_context.adapt_theta_runtime, dtype=float),
        allow_repeats=bool(args.allow_repeats),
        t_final=float(args.t_final),
        num_times=int(args.num_times),
        drive_config=drive_config,
    )
    result = controller.run()
    payload = build_output_payload(
        args=args,
        loaded=loaded,
        cfg=cfg,
        drive_config=drive_config,
        result=result,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run HH realtime checkpoint controller from an ADAPT artifact."
    )
    parser.add_argument("--artifact-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--run-tag", default="hh_realtime_from_adapt_artifact")
    parser.add_argument("--loader-mode", default="replay_family")
    parser.add_argument("--generator-family", default="match_adapt")
    parser.add_argument("--fallback-family", default="full_meta")
    parser.add_argument("--lock-fixed-manifold", action="store_true")
    parser.add_argument("--allow-repeats", action="store_true")
    parser.add_argument("--t-final", type=float, default=8.0)
    parser.add_argument("--num-times", type=int, default=161)
    parser.add_argument("--disable-drive", action="store_true")
    parser.add_argument("--drive-A", type=float, default=1.5)
    parser.add_argument("--drive-omega", type=float, default=1.2)
    parser.add_argument("--drive-tbar", type=float, default=4.0)
    parser.add_argument("--drive-phi", type=float, default=0.0)
    parser.add_argument("--drive-pattern", default="staggered")
    parser.add_argument("--drive-custom-weights", default="")
    parser.add_argument("--drive-include-identity", action="store_true")
    parser.add_argument("--drive-time-sampling", default="midpoint")
    parser.add_argument("--drive-t0", type=float, default=0.0)
    parser.add_argument("--exact-steps-multiplier", type=int, default=4)
    parser.add_argument("--checkpoint-controller-mode", default="exact_v1")
    parser.add_argument("--checkpoint-controller-miss-threshold", type=float, default=0.05)
    parser.add_argument("--checkpoint-controller-gain-ratio-threshold", type=float, default=0.02)
    parser.add_argument("--checkpoint-controller-append-margin-abs", type=float, default=1e-6)
    parser.add_argument(
        "--checkpoint-controller-candidate-step-scales",
        default="0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.8,1.0",
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-baseline-step-refine-rounds",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-baseline-blend-weights",
        default="-0.25,-0.125,0.0,0.125,0.25,0.375,0.5,0.75,1.0",
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-baseline-gain-scales",
        default="",
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-include-tangent-secant-proposal",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-tangent-secant-trust-radius",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-tangent-secant-signed-energy-lead-limit",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-horizon-steps",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-horizon-weights",
        default="2.0,1.0",
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-tracking-fidelity-defect-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-tracking-staggered-error-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-tracking-doublon-error-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-tracking-site-occupations-error-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-tracking-energy-total-error-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-energy-slope-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-energy-curvature-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-energy-excursion-under-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-energy-excursion-over-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-energy-excursion-rel-tolerance",
        type=float,
        default=0.0,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_from_args(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
