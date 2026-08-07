#!/usr/bin/env python3
"""Frozen-scaffold exact-fit diagnostic for HH checkpoint-local expressivity."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.legacy.checkpoint_controller import (
    _site_resolved_number_observables,
)
from pipelines.time_dynamics.legacy.checkpoint_exact_audit import (
    exact_v1_pre_action_snapshot,
)
from pipelines.time_dynamics.runners.hh_from_adapt_artifact import (
    _to_jsonable,
    build_controller_bundle_from_args,
    build_parser as build_realtime_parser,
)

try:  # pragma: no cover - import guard only
    from scipy.optimize import minimize as scipy_minimize
except ImportError:  # pragma: no cover
    scipy_minimize = None


@dataclass(frozen=True)
class FrozenScaffoldExactFitConfig:
    objectives: tuple[str, ...] = ("fidelity_first",)
    method: str = "Powell"
    maxiter: int = 400
    restarts: int = 4
    seed: int = 7
    initial_sigma: float = 0.15
    balanced_energy_weight: float = 1.0
    balanced_site_weight: float = 1.0


def _parse_int_tuple(raw: str | None) -> tuple[int, ...]:
    if raw is None:
        return ()
    text = str(raw).strip()
    if not text:
        return ()
    return tuple(int(chunk.strip()) for chunk in text.split(",") if chunk.strip())


def _parse_str_tuple(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    text = str(raw).strip()
    if not text:
        return ()
    return tuple(str(chunk.strip()) for chunk in text.split(",") if chunk.strip())


"""
loss(theta; objective) = objective(psi(theta), psi_exact, H_k, n_sites_exact)
"""
def _objective_loss(
    metrics: Mapping[str, Any],
    *,
    objective: str,
    cfg: FrozenScaffoldExactFitConfig,
) -> float:
    key = str(objective).strip().lower()
    fidelity = float(metrics.get("fidelity_exact", float("nan")))
    abs_energy_error = float(metrics.get("abs_energy_total_error", float("nan")))
    site_error = float(metrics.get("site_occupations_abs_error_max", float("nan")))
    if key == "fidelity_first":
        return float(1.0 - fidelity)
    if key == "energy_only":
        return float(abs_energy_error)
    if key == "site_only":
        return float(site_error)
    if key == "balanced":
        return float(
            (1.0 - fidelity)
            + float(cfg.balanced_energy_weight) * float(abs_energy_error)
            + float(cfg.balanced_site_weight) * float(site_error)
        )
    raise ValueError(f"Unknown fit objective {objective!r}.")


"""
primary_density(n) = n_0 - n_1 for L=2, else staggered mean over sites
"""
def _primary_density_from_site_occupations(site_occupations: Sequence[float]) -> float:
    occ = np.asarray(site_occupations, dtype=float).reshape(-1)
    if occ.size == 0:
        return float("nan")
    if int(occ.size) == 1:
        return float(occ[0])
    if int(occ.size) == 2:
        return float(occ[0] - occ[1])
    signs = np.asarray([1.0 if (idx % 2 == 0) else -1.0 for idx in range(int(occ.size))], dtype=float)
    return float(np.sum(signs * occ) / float(occ.size))


"""
metrics(theta) = (F_exact(theta), |ΔE|(theta), Δsites(theta)) on the frozen checkpoint scaffold
"""
def evaluate_frozen_scaffold_metrics(
    snapshot: Mapping[str, Any],
    *,
    theta_runtime: np.ndarray | Sequence[float],
) -> dict[str, Any]:
    theta_arr = np.asarray(theta_runtime, dtype=float).reshape(-1)
    executor = snapshot["executor"]
    psi_ref = np.asarray(snapshot["psi_ref"], dtype=complex).reshape(-1)
    psi_exact = np.asarray(snapshot["psi_exact"], dtype=complex).reshape(-1)
    hmat_step = np.asarray(snapshot["hmat_step"], dtype=complex)
    psi_trial = np.asarray(executor.prepare_state(theta_arr, psi_ref), dtype=complex).reshape(-1)
    trial_raw = _site_resolved_number_observables(
        psi_trial,
        num_sites=int(snapshot["num_sites"]),
        ordering=str(snapshot["ordering"]),
    )
    exact_sites = np.asarray(snapshot["exact_observables"]["site_occupations"], dtype=float).reshape(-1)
    trial_sites = np.asarray(trial_raw.n_site, dtype=float).reshape(-1)
    site_error = (
        float(np.max(np.abs(trial_sites - exact_sites)))
        if int(trial_sites.size) == int(exact_sites.size) and int(trial_sites.size) > 0
        else float("nan")
    )
    energy_trial = float(np.real(np.vdot(psi_trial, hmat_step @ psi_trial)))
    energy_exact = float(snapshot["energy_exact"])
    return {
        "theta_runtime": [float(x) for x in theta_arr.tolist()],
        "fidelity_exact": float(abs(np.vdot(psi_exact, psi_trial)) ** 2),
        "energy_trial": float(energy_trial),
        "energy_exact": float(energy_exact),
        "abs_energy_total_error": float(abs(energy_trial - energy_exact)),
        "site_occupations": [float(x) for x in trial_sites.tolist()],
        "site_occupations_abs_error_max": float(site_error),
        "abs_primary_density_error": float(
            abs(
                _primary_density_from_site_occupations(trial_sites)
                - _primary_density_from_site_occupations(exact_sites)
            )
        ),
        "doublon": float(trial_raw.doublon),
        "staggered": float(trial_raw.staggered),
    }


"""
theta_seeds = {theta_current, 0, theta_current + σ ξ_r}
"""
def build_restart_thetas(
    theta_current: np.ndarray | Sequence[float],
    *,
    cfg: FrozenScaffoldExactFitConfig,
) -> list[np.ndarray]:
    theta0 = np.asarray(theta_current, dtype=float).reshape(-1)
    rng = np.random.default_rng(int(cfg.seed))
    seeds: list[np.ndarray] = [np.asarray(theta0, dtype=float).copy()]
    zero_theta = np.zeros_like(theta0)
    if not np.allclose(zero_theta, theta0):
        seeds.append(zero_theta)
    while len(seeds) < int(max(1, cfg.restarts)):
        proposal = np.asarray(
            theta0 + float(cfg.initial_sigma) * rng.normal(size=theta0.shape),
            dtype=float,
        ).reshape(-1)
        if any(np.allclose(proposal, existing) for existing in seeds):
            continue
        seeds.append(proposal)
    return [np.asarray(seed, dtype=float).reshape(-1) for seed in seeds]


"""
fit(theta_0) = argmin_theta loss(theta; objective) with scaffold frozen at the checkpoint snapshot
"""
def fit_single_restart(
    snapshot: Mapping[str, Any],
    *,
    theta_start: np.ndarray | Sequence[float],
    objective: str,
    cfg: FrozenScaffoldExactFitConfig,
) -> dict[str, Any]:
    if scipy_minimize is None:
        raise ImportError("SciPy is required for frozen-scaffold exact fit diagnostics.")

    theta_init = np.asarray(theta_start, dtype=float).reshape(-1)

    def objective_fn(theta_vec: np.ndarray) -> float:
        metrics = evaluate_frozen_scaffold_metrics(snapshot, theta_runtime=np.asarray(theta_vec, dtype=float))
        loss = _objective_loss(metrics, objective=str(objective), cfg=cfg)
        if not np.isfinite(float(loss)):
            return float(1.0e12)
        return float(loss)

    result = scipy_minimize(
        objective_fn,
        theta_init,
        method=str(cfg.method),
        options={"maxiter": int(cfg.maxiter)},
    )
    theta_best = np.asarray(
        result.x if getattr(result, "x", None) is not None else theta_init,
        dtype=float,
    ).reshape(-1)
    best_metrics = evaluate_frozen_scaffold_metrics(snapshot, theta_runtime=theta_best)
    return {
        "objective": str(objective),
        "success": bool(getattr(result, "success", False)),
        "message": str(getattr(result, "message", "")),
        "nit": (None if getattr(result, "nit", None) is None else int(result.nit)),
        "nfev": (None if getattr(result, "nfev", None) is None else int(result.nfev)),
        "loss": float(_objective_loss(best_metrics, objective=str(objective), cfg=cfg)),
        "theta_start": [float(x) for x in theta_init.tolist()],
        "best_metrics": best_metrics,
    }


"""
checkpoint_fit = best_restart_r fit(theta_start_r) over the selected checkpoint-local frozen scaffold
"""
def fit_checkpoint_snapshot(
    snapshot: Mapping[str, Any],
    *,
    cfg: FrozenScaffoldExactFitConfig,
) -> dict[str, Any]:
    theta_current = np.asarray(snapshot["theta_runtime"], dtype=float).reshape(-1)
    current_metrics = evaluate_frozen_scaffold_metrics(snapshot, theta_runtime=theta_current)
    objectives_payload: list[dict[str, Any]] = []
    restart_thetas = build_restart_thetas(theta_current, cfg=cfg)
    for objective in cfg.objectives:
        restart_results: list[dict[str, Any]] = []
        for restart_index, theta_start in enumerate(restart_thetas):
            restart_payload = fit_single_restart(
                snapshot,
                theta_start=np.asarray(theta_start, dtype=float),
                objective=str(objective),
                cfg=cfg,
            )
            restart_payload["restart_index"] = int(restart_index)
            restart_results.append(restart_payload)
        best_restart = min(restart_results, key=lambda row: float(row["loss"]))
        best_metrics = dict(best_restart["best_metrics"])
        objectives_payload.append(
            {
                "objective": str(objective),
                "restart_count": int(len(restart_results)),
                "best_restart_index": int(best_restart["restart_index"]),
                "best_loss": float(best_restart["loss"]),
                "best_metrics": best_metrics,
                "delta_vs_current": {
                    "fidelity_exact": float(best_metrics["fidelity_exact"] - current_metrics["fidelity_exact"]),
                    "abs_energy_total_error": float(
                        current_metrics["abs_energy_total_error"] - best_metrics["abs_energy_total_error"]
                    ),
                    "site_occupations_abs_error_max": float(
                        current_metrics["site_occupations_abs_error_max"]
                        - best_metrics["site_occupations_abs_error_max"]
                    ),
                },
                "restarts": restart_results,
            }
        )
    return {
        "checkpoint_index": int(snapshot["checkpoint_index"]),
        "time": float(snapshot["time"]),
        "time_stop": snapshot["time_stop"],
        "physical_time": float(snapshot["physical_time"]),
        "drive_term_count": int(snapshot["drive_term_count"]),
        "scaffold_labels": [str(x) for x in snapshot["scaffold_labels"]],
        "logical_block_count": int(snapshot["logical_block_count"]),
        "runtime_parameter_count": int(snapshot["runtime_parameter_count"]),
        "current_metrics": current_metrics,
        "exact_observables": dict(snapshot["exact_observables"]),
        "current_observables": dict(snapshot["current_observables"]),
        "objectives": objectives_payload,
    }


def _reference_energy_total_span_full_run(controller: Any, exact_helper: Any) -> float:
    times = np.asarray(getattr(controller, "times", ()), dtype=float).reshape(-1)
    if times.size == 0:
        return 0.0
    energies: list[float] = []
    for idx, time_value in enumerate(times.tolist()):
        time_stop = None if int(idx) + 1 >= int(times.size) else float(times[int(idx) + 1])
        physical_time = float(controller._projection_sample_time(float(time_value), time_stop))
        step_hamiltonian = controller._step_hamiltonian_artifacts(float(physical_time))
        psi_exact = np.asarray(exact_helper.state_at(float(time_value)), dtype=complex).reshape(-1)
        energy_exact = float(
            np.real(np.vdot(psi_exact, np.asarray(step_hamiltonian.hmat, dtype=complex) @ psi_exact))
        )
        energies.append(float(energy_exact))
    energy_arr = np.asarray(energies, dtype=float).reshape(-1)
    if energy_arr.size == 0:
        return 0.0
    return float(np.max(energy_arr) - np.min(energy_arr))


"""
snapshot(k) = replay_prefix_to(k-1) followed by frozen pre-action capture at checkpoint k
"""
def capture_checkpoint_snapshot_from_args(
    args: argparse.Namespace,
    *,
    checkpoint_index: int,
    force_stay_checkpoints: Sequence[int],
    exact_reference_cache: dict[str, object] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle = build_controller_bundle_from_args(args, exact_reference_cache=exact_reference_cache)
    controller = bundle["controller"]
    exact_helper = bundle.get("exact_helper")
    if exact_helper is None:
        if hasattr(controller, "_exact_state_at"):
            exact_helper = SimpleNamespace(
                state_at=lambda time_value: controller._exact_state_at(float(time_value))
            )
        else:
            raise ValueError("Frozen scaffold exact fit requires benchmark exact helper routing.")
    prefix_force_stay = tuple(int(x) for x in force_stay_checkpoints if int(x) < int(checkpoint_index))
    if int(checkpoint_index) > 0:
        controller.debug_probe_exact_v1(
            probe_checkpoints=(int(checkpoint_index) - 1,),
            force_stay_checkpoints=prefix_force_stay,
            candidate_rank_limit=1,
            baseline_variant_limit=1,
            reference_payload=None,
        )
    if hasattr(controller, "exact_v1_pre_action_snapshot") and not hasattr(controller, "cfg"):
        snapshot = controller.exact_v1_pre_action_snapshot(
            checkpoint_index=int(checkpoint_index)
        )
    else:
        snapshot = exact_v1_pre_action_snapshot(
            controller,
            exact_helper,
            checkpoint_index=int(checkpoint_index),
        )
    snapshot["prefix_force_stay_checkpoints"] = [int(x) for x in prefix_force_stay]
    snapshot["reference_energy_total_span_full_run"] = float(
        _reference_energy_total_span_full_run(controller, exact_helper)
    )
    return snapshot, bundle


"""
payload = {checkpoint_k : fit_checkpoint_snapshot(snapshot_k)} for requested checkpoints
"""
def run_exact_fit_from_args(args: argparse.Namespace) -> dict[str, Any]:
    fit_checkpoints = sorted({int(x) for x in _parse_int_tuple(getattr(args, "fit_checkpoints", None))})
    if not fit_checkpoints:
        raise ValueError("--fit-checkpoints must be non-empty")
    force_stay_checkpoints = tuple(int(x) for x in _parse_int_tuple(getattr(args, "force_stay_checkpoints", None)))
    fit_cfg = FrozenScaffoldExactFitConfig(
        objectives=tuple(str(x) for x in _parse_str_tuple(getattr(args, "fit_objectives", None)) or ("fidelity_first",)),
        method=str(getattr(args, "fit_method", "Powell")),
        maxiter=int(getattr(args, "fit_maxiter", 400)),
        restarts=int(getattr(args, "fit_restarts", 4)),
        seed=int(getattr(args, "fit_seed", 7)),
        initial_sigma=float(getattr(args, "fit_initial_sigma", 0.15)),
        balanced_energy_weight=float(getattr(args, "fit_balanced_energy_weight", 1.0)),
        balanced_site_weight=float(getattr(args, "fit_balanced_site_weight", 1.0)),
    )
    exact_reference_cache: dict[str, object] = {}
    bootstrap_bundle = build_controller_bundle_from_args(
        args,
        exact_reference_cache=exact_reference_cache,
    )
    results: list[dict[str, Any]] = []
    for checkpoint_index in fit_checkpoints:
        snapshot, _bundle = capture_checkpoint_snapshot_from_args(
            args,
            checkpoint_index=int(checkpoint_index),
            force_stay_checkpoints=force_stay_checkpoints,
            exact_reference_cache=exact_reference_cache,
        )
        fit_payload = fit_checkpoint_snapshot(snapshot, cfg=fit_cfg)
        fit_payload["prefix_force_stay_checkpoints"] = [
            int(x) for x in snapshot.get("prefix_force_stay_checkpoints", [])
        ]
        results.append(fit_payload)
    output_json = Path(args.output_json).expanduser().resolve()
    payload = {
        "pipeline": "hh_fixed_scaffold_exact_fit_v1",
        "run_tag": str(args.run_tag),
        "artifact_json": str(Path(args.artifact_json).expanduser().resolve()),
        "output_json": str(output_json),
        "loader_mode": str(args.loader_mode),
        "fit_checkpoints": [int(x) for x in fit_checkpoints],
        "requested_force_stay_checkpoints": [int(x) for x in force_stay_checkpoints],
        "fit_config": _to_jsonable(asdict(fit_cfg)),
        "controller_config": _to_jsonable(bootstrap_bundle["cfg"]),
        "drive_config": _to_jsonable(bootstrap_bundle["drive_config"]),
        "oracle_config": _to_jsonable(bootstrap_bundle["oracle_config"]),
        "results": _to_jsonable(results),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = build_realtime_parser()
    parser.description = "Freeze the live HH checkpoint scaffold, then fit theta-only against the exact target state."
    parser.set_defaults(
        checkpoint_controller_mode="exact_v1",
        checkpoint_controller_reference_mode="benchmark_exact",
    )
    parser.add_argument("--fit-checkpoints", required=True)
    parser.add_argument("--force-stay-checkpoints", default="")
    parser.add_argument(
        "--fit-objectives",
        default="fidelity_first",
        help="Comma-separated list: fidelity_first, energy_only, site_only, balanced.",
    )
    parser.add_argument("--fit-method", type=str, default="Powell")
    parser.add_argument("--fit-maxiter", type=int, default=400)
    parser.add_argument("--fit-restarts", type=int, default=4)
    parser.add_argument("--fit-seed", type=int, default=7)
    parser.add_argument("--fit-initial-sigma", type=float, default=0.15)
    parser.add_argument("--fit-balanced-energy-weight", type=float, default=1.0)
    parser.add_argument("--fit-balanced-site-weight", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_exact_fit_from_args(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
