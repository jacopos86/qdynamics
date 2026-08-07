#!/usr/bin/env python3
"""Global constrained multi-fidelity SNAKE policy tuner for Paper-I HH.

This module implements the GPT-Pro recommended protocol shape without changing
the Paper-I evidence rows by itself: train SNAKE policy hyperparameters on
off-canonical Hubbard-Holstein points, aggregate each trial with a scalar energy
loss plus explicit resource constraints, and reserve the six canonical regimes
for later evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench import hh_cost_energy_optuna  # noqa: E402

SCHEMA = "paper_i_hh_snake_global_policy_optuna_config_v1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "raw_outputs/local_smokes"
DEFAULT_EPSILON_TARGET = 2.0e-4
DEFAULT_CVAR_ALPHA = 0.8
DEFAULT_KAPPA = 0.5
DEFAULT_FAIL_PENALTY = 12.0
DEFAULT_LANE = "canonical"
DEFAULT_EPSILON_BAND = 1.0e9
DEFAULT_SPEED_SURFACE = hh_cost_energy_optuna._HH_ROUTEA_FULL_POLICY_PROFILE
DEFAULT_SEARCH_INNER_OPTIMIZER = "POWELL"
DEFAULT_CLASS_FILTER = REPO_ROOT / "agent_guidance/static-adapt/hh_full_meta_minus_hva_class_filter.json"


@dataclass(frozen=True)
class HHPoint:
    label: str
    u: float
    lambda_value: float
    n_ph_work: int
    n_ph_ref: int
    t: float = 1.0
    omega0: float = 1.0
    role: str = "train"


@dataclass(frozen=True)
class FidelityStage:
    name: str
    max_depth: int
    maxiter: int
    promote_top: int = 0


@dataclass(frozen=True)
class BudgetSpec:
    name: str
    depth_2q: float | None = None
    count_2q: float | None = None
    shot_proxy: float | None = None
    p_fail_max: float = 0.10


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _float_slug(value: float) -> str:
    return f"{float(value):.12g}".replace("-", "m").replace(".", "p")


def _cutoffs_for_lambda(lambda_value: float) -> tuple[int, int]:
    return (2, 2) if float(lambda_value) <= 0.75 else (4, 4)


def default_training_points() -> list[HHPoint]:
    points: list[HHPoint] = []
    for u in (0.5, 2.0, 5.0, 10.0):
        for lambda_value in (0.5, 1.0):
            n_ph_work, n_ph_ref = _cutoffs_for_lambda(lambda_value)
            points.append(
                HHPoint(
                    label=f"train_u{_float_slug(u)}_lam{_float_slug(lambda_value)}",
                    u=float(u),
                    lambda_value=float(lambda_value),
                    n_ph_work=int(n_ph_work),
                    n_ph_ref=int(n_ph_ref),
                    role="train",
                )
            )
    return points


def default_evaluation_points() -> list[HHPoint]:
    return [
        HHPoint("weak-weak", 0.25, 0.25, 2, 2, role="eval"),
        HHPoint("intermediate-weak", 1.25, 0.25, 2, 2, role="eval"),
        HHPoint("weak-strong", 0.25, 1.25, 4, 4, role="eval"),
        HHPoint("intermediate-strong", 1.25, 1.25, 4, 4, role="eval"),
        HHPoint("strong-weak-u8", 8.0, 0.25, 2, 2, role="eval"),
        HHPoint("strong-strong-u8", 8.0, 1.25, 4, 4, role="eval"),
    ]


def default_fidelity_stages() -> list[FidelityStage]:
    return [
        FidelityStage("depth8_iter50", max_depth=8, maxiter=50, promote_top=64),
        FidelityStage("depth16_iter100", max_depth=16, maxiter=100, promote_top=32),
        FidelityStage("depth30_iter200", max_depth=30, maxiter=200, promote_top=0),
    ]


def canonical_eval_pairs() -> set[tuple[float, float]]:
    return {(round(p.u, 12), round(p.lambda_value, 12)) for p in default_evaluation_points()}


def validate_training_points(points: Sequence[HHPoint]) -> None:
    canonical_labels = {p.label for p in default_evaluation_points()}
    canonical_pairs = canonical_eval_pairs()
    seen: set[str] = set()
    for point in points:
        if point.label in seen:
            raise ValueError(f"Duplicate HH training point label: {point.label}")
        seen.add(point.label)
        if point.label in canonical_labels:
            raise ValueError(f"Training point label collides with held-out canonical regime: {point.label}")
        pair = (round(float(point.u), 12), round(float(point.lambda_value), 12))
        if pair in canonical_pairs:
            raise ValueError(
                "Training point collides with held-out canonical regime coordinates: "
                f"{point.label} has U={point.u}, lambda={point.lambda_value}"
            )


def _finite_float(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _event_abs_delta_e(event: Mapping[str, Any]) -> float | None:
    for key in ("abs_delta_e", "delta_e_abs"):
        value = _finite_float(event.get(key))
        if value is not None:
            return value
    observation = event.get("observation")
    if isinstance(observation, Mapping):
        return _event_abs_delta_e(observation)
    return None


def _event_success(event: Mapping[str, Any]) -> bool:
    if event.get("failed") is True:
        return False
    if event.get("feasible") is False and _event_abs_delta_e(event) is None:
        return False
    invalid = event.get("invalid_reasons")
    if isinstance(invalid, Sequence) and not isinstance(invalid, (str, bytes, bytearray)) and len(invalid) > 0:
        return False
    observation = event.get("observation")
    if isinstance(observation, Mapping):
        return _event_success(observation)
    return _event_abs_delta_e(event) is not None


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        raise ValueError("Cannot compute median of an empty sequence.")
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return 0.5 * (float(ordered[mid - 1]) + float(ordered[mid]))


def _quantile(values: Sequence[float], q: float) -> float | None:
    finite = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    q_clamped = min(1.0, max(0.0, float(q)))
    pos = q_clamped * (len(finite) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return finite[lo]
    weight = pos - lo
    return float((1.0 - weight) * finite[lo] + weight * finite[hi])


def energy_loss(
    events: Sequence[Mapping[str, Any]],
    *,
    epsilon_target: float = DEFAULT_EPSILON_TARGET,
    kappa: float = DEFAULT_KAPPA,
    cvar_alpha: float = DEFAULT_CVAR_ALPHA,
    fail_penalty: float = DEFAULT_FAIL_PENALTY,
) -> float:
    if not events:
        return float(fail_penalty)
    z_values: list[float] = []
    for event in events:
        abs_delta_e = _event_abs_delta_e(event)
        if abs_delta_e is None or not _event_success(event):
            z_values.append(float(fail_penalty))
            continue
        z_values.append(math.log10(1.0 + float(abs_delta_e) / float(epsilon_target)))
    tail_start = max(0, int(math.floor(float(cvar_alpha) * len(z_values))))
    tail = sorted(z_values)[tail_start:] or sorted(z_values)[-1:]
    return float(_median(z_values) + float(kappa) * (sum(tail) / len(tail)))


def _event_resource_value(event: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        value = _finite_float(event.get(key))
        if value is not None:
            return value
    observation = event.get("observation")
    if isinstance(observation, Mapping):
        return _event_resource_value(observation, keys)
    return None


def _event_shot_proxy(event: Mapping[str, Any]) -> tuple[float | None, str]:
    status = str(event.get("paper_i_table_shots_status") or event.get("S_status") or "")
    value = _event_resource_value(event, ("paper_i_table_s_alg", "S_alg", "shot_proxy"))
    observation = event.get("observation")
    if value is None and isinstance(observation, Mapping):
        return _event_shot_proxy(observation)
    if value is None:
        return None, "missing"
    if status and status not in {"ok", "OK"}:
        return value, status
    return value, "ok_internal_snake_s_alg"


def constraint_summary(
    events: Sequence[Mapping[str, Any]],
    budget: BudgetSpec,
    *,
    q: float = 0.9,
) -> dict[str, Any]:
    total = max(1, len(events))
    failed = sum(1 for event in events if not _event_success(event))
    depth_values = [
        value
        for event in events
        if (value := _event_resource_value(event, ("graph_depth", "compiled_depth", "compiled_depth_2q"))) is not None
    ]
    count_values = [
        value
        for event in events
        if (value := _event_resource_value(event, ("graph_count_2q", "compiled_count_2q", "N2q"))) is not None
    ]
    shot_values: list[float] = []
    shot_statuses: list[str] = []
    for event in events:
        shot, status = _event_shot_proxy(event)
        shot_statuses.append(status)
        if shot is not None:
            shot_values.append(float(shot))

    def normalized_constraint(values: Sequence[float], bound: float | None) -> float:
        if bound is None:
            return -1.0
        quantile = _quantile(values, q)
        if quantile is None:
            return 1.0
        return float(quantile / float(bound) - 1.0)

    constraints = {
        "depth_2q": normalized_constraint(depth_values, budget.depth_2q),
        "count_2q": normalized_constraint(count_values, budget.count_2q),
        "shot_proxy": normalized_constraint(shot_values, budget.shot_proxy),
        "failure_rate": float(failed / total - float(budget.p_fail_max)),
    }
    return {
        "budget": asdict(budget),
        "q": float(q),
        "constraints": constraints,
        "constraint_vector": [
            float(constraints["depth_2q"]),
            float(constraints["count_2q"]),
            float(constraints["shot_proxy"]),
            float(constraints["failure_rate"]),
        ],
        "failed_case_count": int(failed),
        "case_count": int(len(events)),
        "depth_2q_q": _quantile(depth_values, q),
        "count_2q_q": _quantile(count_values, q),
        "shot_proxy_q": _quantile(shot_values, q),
        "shot_proxy_statuses": sorted(set(shot_statuses)),
        "shot_proxy_contract": "internal_snake_s_alg_only_not_cross_method_fair_s",
    }


def build_enqueue_params_manifest(params: Mapping[str, Any], *, point_label: str | None = None) -> dict[str, Any]:
    if point_label in {None, ""}:
        return {"schema": "paper_i_hh_snake_global_policy_enqueue_params_v1", "enqueue_params": [dict(params)]}
    return {
        "schema": "paper_i_hh_snake_global_policy_enqueue_params_v1",
        "regimes": {str(point_label): {"enqueue_params": [dict(params)]}},
    }


def build_case_argv(
    point: HHPoint,
    stage: FidelityStage,
    *,
    params_json: Path | None = None,
    output_dir: Path | None = None,
    python_bin: str = sys.executable,
    search_inner_optimizer: str = DEFAULT_SEARCH_INNER_OPTIMIZER,
    n_trials: int = 1,
    n_startup_trials: int = 1,
    no_exact_manifest: bool = True,
    preserve_cost_surface_with_skip_compile: bool = True,
) -> list[str]:
    tag = f"paper_i_hh_snake_global_policy__{point.label}__{stage.name}"
    argv = [
        str(python_bin),
        str(REPO_ROOT / "pipelines/exact_bench/paper_i_hh_speed_optuna.py"),
        "--point-label",
        str(point.label),
        "--hh-t",
        str(float(point.t)),
        "--hh-u",
        str(float(point.u)),
        "--hh-omega0",
        str(float(point.omega0)),
        "--hh-lambda",
        str(float(point.lambda_value)),
        "--n-ph-work",
        str(int(point.n_ph_work)),
        "--n-ph-ref",
        str(int(point.n_ph_ref)),
        "--tag",
        tag,
        "--objective-mode",
        "energy",
        "--speed-surface-profile",
        DEFAULT_SPEED_SURFACE,
        "--runtime-split-mode",
        "shortlist_pauli_children_v1",
        "--symmetry-mode",
        "off",
        "--search-inner-optimizer",
        str(search_inner_optimizer),
        "--n-trials",
        str(int(n_trials)),
        "--n-startup-trials",
        str(int(n_startup_trials)),
        "--max-depth",
        str(int(stage.max_depth)),
        "--maxiter",
        str(int(stage.maxiter)),
        "--final-refit-maxiter",
        str(int(stage.maxiter)),
        "--force-run-to-depth",
        "--no-default-warm-starts",
    ]
    if output_dir is not None:
        argv.extend(["--output-dir", str(Path(output_dir))])
    if params_json is not None:
        argv.extend(["--enqueue-params-json", str(Path(params_json))])
    if bool(no_exact_manifest):
        argv.append("--no-exact-manifest")
    if bool(preserve_cost_surface_with_skip_compile):
        argv.append("--preserve-cost-surface-with-skip-compile")
    return argv


def build_campaign_manifest(
    *,
    training_points: Sequence[HHPoint] | None = None,
    evaluation_points: Sequence[HHPoint] | None = None,
    stages: Sequence[FidelityStage] | None = None,
    budgets: Sequence[BudgetSpec] | None = None,
) -> dict[str, Any]:
    train = list(training_points or default_training_points())
    validate_training_points(train)
    eval_points = list(evaluation_points or default_evaluation_points())
    fidelity_stages = list(stages or default_fidelity_stages())
    budget_specs = list(budgets or [BudgetSpec("energy_first_no_resource_gate")])
    return {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "global_snaKE_policy_scalar_energy_loss_with_explicit_constraints",
        "paper": "Paper-I",
        "method": "SNAKE",
        "held_out_policy": "six canonical HH regimes are evaluation-only by default",
        "training_points": [asdict(point) for point in train],
        "evaluation_points": [asdict(point) for point in eval_points],
        "fidelity_stages": [asdict(stage) for stage in fidelity_stages],
        "budgets": [asdict(budget) for budget in budget_specs],
        "loss": {
            "epsilon_target": DEFAULT_EPSILON_TARGET,
            "kappa": DEFAULT_KAPPA,
            "cvar_alpha": DEFAULT_CVAR_ALPHA,
            "fail_penalty": DEFAULT_FAIL_PENALTY,
            "formula": "median(log10(1+abs_delta_e/epsilon_target)) + kappa*CVaR_alpha",
        },
        "fixed_identity_locks": {
            "route_id": "route_a",
            "static_meta_feature_profile": "paper_i_production_v1",
            "pool_policy": "full_meta_minus_hva",
            "pauli_children": "shortlist_pauli_children_v1",
            "symmetry_mitigation_mode": "off",
            "inner_optimizer_default": DEFAULT_SEARCH_INNER_OPTIMIZER,
        },
        "shot_proxy_note": (
            "S constraints use SNAKE internal S_alg only unless a corrected fair S sidecar is "
            "explicitly wired in; this is suitable for SNAKE policy tuning, not cross-method claims."
        ),
    }


def _point_overrides(point: HHPoint) -> hh_cost_energy_optuna.HhHamiltonianOverrides:
    return hh_cost_energy_optuna.HhHamiltonianOverrides(
        L=2,
        t=float(point.t),
        u=float(point.u),
        omega0=float(point.omega0),
        lambda_value=float(point.lambda_value),
        n_ph_work=int(point.n_ph_work),
        n_ph_ref=int(point.n_ph_ref),
        adapt_pool="full_meta",
    )


def _observation_event(point: HHPoint, stage: FidelityStage, observation: Any) -> dict[str, Any]:
    payload = asdict(observation)
    payload.update({"point_label": point.label, "stage": stage.name})
    return payload


def run_local_study(
    *,
    output_dir: Path,
    n_trials: int,
    budget: BudgetSpec,
    training_points: Sequence[HHPoint] | None = None,
    stages: Sequence[FidelityStage] | None = None,
    python_bin: str = sys.executable,
    search_inner_optimizer: str = DEFAULT_SEARCH_INNER_OPTIMIZER,
) -> dict[str, Any]:
    optuna = hh_cost_energy_optuna._import_optuna()
    train = list(training_points or default_training_points())
    validate_training_points(train)
    fidelity_stages = list(stages or default_fidelity_stages())
    output_dir.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        group=True,
        constant_liar=True,
        constraints_func=lambda frozen_trial: list(frozen_trial.user_attrs.get("constraints", [1.0, 1.0, 1.0, 1.0])),
        n_startup_trials=max(1, min(64, int(n_trials))),
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)
    trial_events: list[dict[str, Any]] = []
    for _ in range(int(n_trials)):
        trial = study.ask()
        params = hh_cost_energy_optuna._suggest_trial_params(
            trial,
            DEFAULT_LANE,
            ("resolved_default",),
            energy_only_surface=False,
            speed_surface_profile=DEFAULT_SPEED_SURFACE,
            force_spsa_profile="current" if str(search_inner_optimizer).upper() != "SPSA" else None,
            phase2_w_shot_profile_space="default",
            anchor_param_values={},
            enable_prune_prefilter_profile_space=False,
        )
        events: list[dict[str, Any]] = []
        for stage in fidelity_stages:
            for point in train:
                case_output = output_dir / "cases" / stage.name / point.label
                try:
                    observation = hh_cost_energy_optuna._evaluate_trial(
                        python_bin=str(python_bin),
                        params=params,
                        lane=DEFAULT_LANE,
                        epsilon_abs_delta_e=DEFAULT_EPSILON_BAND,
                        output_dir=case_output,
                        trial_index=int(trial.number),
                        compile_backend="FakeMarrakesh",
                        compile_opt_level=1,
                        compile_seed=7,
                        hamiltonian_overrides=_point_overrides(point),
                        compile_enabled=False,
                        runtime_split_mode_override="off",
                        child_pool_expansion_mode_override="global_pauli_child_sets_v1",
                        child_pool_expansion_symmetry_policy_override="hard_guard",
                        child_pool_expansion_max_subset_size_override=3,
                        force_adapt_pool_class_filter_json=DEFAULT_CLASS_FILTER,
                        force_static_route_id="route_a",
                        force_static_meta_feature_profile="paper_i_production_v1",
                        force_phase3_symmetry_mitigation_mode="off",
                        force_route_a_paper_i_production=True,
                        force_adapt_max_depth=int(stage.max_depth),
                        force_adapt_maxiter=int(stage.maxiter),
                        force_adapt_final_refit_maxiter=int(stage.maxiter),
                        force_adapt_drop_min_depth=int(stage.max_depth) + 1,
                        force_adapt_drop_patience=1_000_000,
                        search_inner_optimizer=str(search_inner_optimizer),
                        require_graph_cost=False,
                    )
                    events.append(_observation_event(point, stage, observation))
                except Exception as exc:
                    events.append(
                        {
                            "point_label": point.label,
                            "stage": stage.name,
                            "failed": True,
                            "error": str(exc),
                        }
                    )
        loss = energy_loss(events)
        constraints = constraint_summary(events, budget)
        trial.set_user_attr("constraints", constraints["constraint_vector"])
        trial.set_user_attr("events", events)
        trial.set_user_attr("constraint_summary", constraints)
        study.tell(trial, float(loss))
        trial_events.append(
            {
                "trial_number": int(trial.number),
                "objective_value": float(loss),
                "params": asdict(params),
                "constraints": constraints,
                "events": events,
            }
        )
        with (output_dir / "trial_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trial_events[-1], sort_keys=True) + "\n")
    best_trial = None
    best_value = None
    best_params = None
    if trial_events:
        best_trial = study.best_trial
        best_value = study.best_value
        best_params = dict(best_trial.params)
    summary = {
        "schema": "paper_i_hh_snake_global_policy_optuna_summary_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_manifest": build_campaign_manifest(training_points=train, stages=fidelity_stages, budgets=[budget]),
        "trial_count": int(len(trial_events)),
        "best_trial_number": None if best_trial is None else int(best_trial.number),
        "best_value": None if best_value is None else float(best_value),
        "best_params": best_params,
        "trials": trial_events,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or run the Paper-I HH global SNAKE policy Optuna campaign.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / f"paper_i_hh_snake_global_policy_{_timestamp_slug()}")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--print-plan-only", action="store_true")
    parser.add_argument("--run-local", action="store_true", help="Run the local global Optuna loop. This can be expensive.")
    parser.add_argument("--n-trials", type=int, default=1)
    parser.add_argument("--budget-name", type=str, default="energy_first_no_resource_gate")
    parser.add_argument("--budget-depth-2q", type=float, default=None)
    parser.add_argument("--budget-count-2q", type=float, default=None)
    parser.add_argument("--budget-shot-proxy", type=float, default=None)
    parser.add_argument("--p-fail-max", type=float, default=0.10)
    parser.add_argument(
        "--search-inner-optimizer",
        choices=sorted(hh_cost_energy_optuna._SEARCH_INNER_OPTIMIZER_CHOICES),
        default=DEFAULT_SEARCH_INNER_OPTIMIZER,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    budget = BudgetSpec(
        name=str(args.budget_name),
        depth_2q=args.budget_depth_2q,
        count_2q=args.budget_count_2q,
        shot_proxy=args.budget_shot_proxy,
        p_fail_max=float(args.p_fail_max),
    )
    if bool(args.run_local):
        payload = run_local_study(
            output_dir=Path(args.output_dir),
            n_trials=int(args.n_trials),
            budget=budget,
            search_inner_optimizer=str(args.search_inner_optimizer),
        )
    else:
        manifest = build_campaign_manifest(budgets=[budget])
        case_commands = []
        first_stage = default_fidelity_stages()[0]
        for point in default_training_points():
            case_commands.append(
                build_case_argv(
                    point,
                    first_stage,
                    output_dir=Path(args.output_dir) / "case_commands" / point.label,
                    search_inner_optimizer=str(args.search_inner_optimizer),
                )
            )
        payload = {
            **manifest,
            "output_dir": str(Path(args.output_dir)),
            "first_stage_case_commands": case_commands,
        }
    if args.output_json is not None:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if bool(args.print_plan_only) or not bool(args.run_local):
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
