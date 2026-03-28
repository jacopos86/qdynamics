#!/usr/bin/env python3
"""Static L=2 HH realtime McLachlan sweep from saved artifacts.

Purpose:
- compare adaptive exact-v1 checkpoint control against fixed-manifold exact McLachlan
- cover broad vs lean families
- run short and long horizons
- emit a simple Pareto-style summary from saved artifacts
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.hardcoded.hh_fixed_manifold_mclachlan import (
    DEFAULT_LOCKED_7TERM_ARTIFACT,
    DEFAULT_PARETO_ARTIFACT,
    FixedManifoldRunSpec,
    load_run_context,
    run_fixed_manifold_exact,
    summarize_result_artifact,
)
from pipelines.hardcoded.hh_realtime_checkpoint_controller import RealtimeCheckpointController
from pipelines.hardcoded.hh_realtime_checkpoint_types import RealtimeCheckpointConfig
from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix

FULL_META_ARTIFACT = Path("artifacts/json/campaign_A7f_L2_shortlist_off_runtime_split_off_phase3_v1.json")


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_hh_l2_static_realtime_pareto_sweep")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_adaptive_exact_from_artifact(
    spec: FixedManifoldRunSpec,
    *,
    tag: str,
    output_path: Path,
    t_final: float,
    num_times: int,
    controller_cfg: RealtimeCheckpointConfig,
) -> dict[str, Any]:
    loaded = load_run_context(spec, tag=tag, lock_fixed_manifold=False)
    if not bool(loaded.replay_context.pool_meta.get("candidate_pool_complete", False)):
        raise ValueError("Adaptive artifact sweep requires a complete candidate family pool.")

    hmat = np.asarray(hamiltonian_matrix(loaded.replay_context.h_poly), dtype=complex)
    controller = RealtimeCheckpointController(
        cfg=controller_cfg,
        replay_context=loaded.replay_context,
        h_poly=loaded.replay_context.h_poly,
        hmat=hmat,
        psi_initial=loaded.psi_initial,
        best_theta=loaded.replay_context.adapt_theta_runtime,
        allow_repeats=False,
        t_final=float(t_final),
        num_times=int(num_times),
    )
    result = controller.run()
    extra_summary = summarize_result_artifact(
        trajectory=result.trajectory,
        summary=result.summary,
    )

    settings = loaded.payload.get("settings", {})
    if not isinstance(settings, Mapping):
        settings = {}

    payload = {
        "generated_utc": _now_utc(),
        "pipeline": "hh_adaptive_artifact_exact_mclachlan_v1",
        "run_name": str(spec.name),
        "input_artifact_json": str(spec.artifact_json),
        "loader": dict(loaded.loader_summary),
        "manifest": {
            "model_family": "Hubbard-Holstein",
            "ansatz_type": str(spec.name),
            "drive_enabled": False,
            "decision_mode": "exact_v1",
            "structure_policy": "adaptive_realtime_checkpoint_candidate_family",
            "effective_pool_kind": str(loaded.loader_summary.get("family_pool_origin", "candidate_family")),
            "t": float(settings.get("t", loaded.cfg.t)),
            "U": float(settings.get("u", loaded.cfg.u)),
            "dv": float(settings.get("dv", loaded.cfg.dv)),
            "omega0": float(settings.get("omega0", loaded.cfg.omega0)),
            "g_ep": float(settings.get("g_ep", loaded.cfg.g_ep)),
            "n_ph_max": int(settings.get("n_ph_max", loaded.cfg.n_ph_max)),
            "L": int(settings.get("L", loaded.cfg.L)),
        },
        "run_config": {
            "t_final": float(t_final),
            "num_times": int(num_times),
            "allow_repeats": False,
            "decision_mode": "exact_v1",
            "controller": {
                "mode": str(controller_cfg.mode),
                "miss_threshold": float(controller_cfg.miss_threshold),
                "gain_ratio_threshold": float(controller_cfg.gain_ratio_threshold),
                "append_margin_abs": float(controller_cfg.append_margin_abs),
                "shortlist_size": int(controller_cfg.shortlist_size),
                "shortlist_fraction": float(controller_cfg.shortlist_fraction),
            },
            "initial_logical_block_count": int(loaded.replay_context.base_layout.logical_parameter_count),
            "initial_runtime_parameter_count": int(loaded.replay_context.base_layout.runtime_parameter_count),
        },
        "summary": dict(result.summary),
        "extra_summary": dict(extra_summary),
        "reference": dict(result.reference),
        "trajectory": [dict(row) for row in result.trajectory],
        "ledger": [dict(row) for row in result.ledger],
    }
    _write_json(Path(output_path), payload)

    return {
        "name": str(spec.name),
        "status": "completed",
        "branch": "adaptive_exact",
        "input_artifact_json": str(spec.artifact_json),
        "output_json": str(output_path),
        "loader_mode": str(loaded.loader_summary.get("loader_mode", "")),
        "resolved_family": str(loaded.replay_context.family_info.get("resolved", "")),
        "summary": dict(result.summary),
        "extra_summary": dict(extra_summary),
        "loader": dict(loaded.loader_summary),
    }


def _record_metrics(record: Mapping[str, Any]) -> dict[str, Any]:
    extra = record.get("extra_summary", {}) if isinstance(record.get("extra_summary", {}), Mapping) else {}
    summary = record.get("summary", {}) if isinstance(record.get("summary", {}), Mapping) else {}
    fidelity_min = float(extra.get("fidelity_min", float("nan")))
    energy_err_max = float(extra.get("abs_energy_total_error_max", float("nan")))
    runtime_count = int(extra.get("final_runtime_parameter_count", 10**9))
    logical_count = int(extra.get("final_logical_block_count", 10**9))
    append_count = int(summary.get("append_count", 0))
    return {
        "fidelity_gap": float(max(0.0, 1.0 - fidelity_min)) if np.isfinite(fidelity_min) else float("inf"),
        "energy_err_max": energy_err_max,
        "runtime_count": runtime_count,
        "logical_count": logical_count,
        "append_count": append_count,
    }


def _dominates(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> bool:
    lhs_m = _record_metrics(lhs)
    rhs_m = _record_metrics(rhs)
    lhs_vals = [
        float(lhs_m["fidelity_gap"]),
        float(lhs_m["energy_err_max"]),
        float(lhs_m["runtime_count"]),
    ]
    rhs_vals = [
        float(rhs_m["fidelity_gap"]),
        float(rhs_m["energy_err_max"]),
        float(rhs_m["runtime_count"]),
    ]
    return all(l <= r for l, r in zip(lhs_vals, rhs_vals)) and any(l < r for l, r in zip(lhs_vals, rhs_vals))


def _pareto_frontier(records: Sequence[Mapping[str, Any]]) -> list[str]:
    completed = [dict(r) for r in records if str(r.get("status", "")) == "completed"]
    frontier: list[str] = []
    for row in completed:
        if any(_dominates(other, row) for other in completed if other is not row):
            continue
        frontier.append(str(row.get("name")))
    return sorted(frontier)


def _summary_payload(*, tag: str, output_dir: Path, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [dict(r) for r in records if str(r.get("status", "")) == "completed"]
    failed = [dict(r) for r in records if str(r.get("status", "")) != "completed"]
    best_fidelity = None if not completed else max(
        completed,
        key=lambda r: float(r.get("extra_summary", {}).get("fidelity_min", float("-inf"))),
    )
    leanest = None if not completed else min(
        completed,
        key=lambda r: int(r.get("extra_summary", {}).get("final_runtime_parameter_count", 10**9)),
    )
    return {
        "generated_utc": _now_utc(),
        "pipeline": "hh_l2_static_realtime_pareto_sweep_v1",
        "tag": str(tag),
        "output_dir": str(output_dir),
        "completed_runs": int(len(completed)),
        "failed_runs": int(len(failed)),
        "frontier_names": _pareto_frontier(completed),
        "best_min_fidelity_run": None if best_fidelity is None else str(best_fidelity.get("name")),
        "leanest_runtime_run": None if leanest is None else str(leanest.get("name")),
        "runs": [dict(r) for r in records],
    }


def _case_specs(
    *,
    short_t_final: float,
    short_num_times: int,
    long_t_final: float,
    long_num_times: int,
) -> list[dict[str, Any]]:
    fixed_controller = {
        "miss_threshold": 1.0e9,
        "gain_ratio_threshold": 1.0e-9,
        "append_margin_abs": 1.0e-12,
    }
    adaptive_controller = RealtimeCheckpointConfig(
        mode="exact_v1",
        miss_threshold=0.05,
        gain_ratio_threshold=0.02,
        append_margin_abs=1.0e-6,
    )
    return [
        {
            "name": "fixed_locked7_short",
            "branch": "fixed_exact",
            "spec": FixedManifoldRunSpec(
                name="fixed_locked7_short",
                artifact_json=Path(DEFAULT_LOCKED_7TERM_ARTIFACT),
                loader_mode="fixed_scaffold",
                generator_family="fixed_scaffold_locked",
                fallback_family="full_meta",
            ),
            "t_final": float(short_t_final),
            "num_times": int(short_num_times),
            "controller": dict(fixed_controller),
        },
        {
            "name": "fixed_pareto_short",
            "branch": "fixed_exact",
            "spec": FixedManifoldRunSpec(
                name="fixed_pareto_short",
                artifact_json=Path(DEFAULT_PARETO_ARTIFACT),
                loader_mode="replay_family",
                generator_family="match_adapt",
                fallback_family="full_meta",
            ),
            "t_final": float(short_t_final),
            "num_times": int(short_num_times),
            "controller": dict(fixed_controller),
        },
        {
            "name": "fixed_fullmeta_short",
            "branch": "fixed_exact",
            "spec": FixedManifoldRunSpec(
                name="fixed_fullmeta_short",
                artifact_json=Path(FULL_META_ARTIFACT),
                loader_mode="replay_family",
                generator_family="match_adapt",
                fallback_family="full_meta",
            ),
            "t_final": float(short_t_final),
            "num_times": int(short_num_times),
            "controller": dict(fixed_controller),
        },
        {
            "name": "adaptive_pareto_short",
            "branch": "adaptive_exact",
            "spec": FixedManifoldRunSpec(
                name="adaptive_pareto_short",
                artifact_json=Path(DEFAULT_PARETO_ARTIFACT),
                loader_mode="replay_family",
                generator_family="match_adapt",
                fallback_family="full_meta",
            ),
            "t_final": float(short_t_final),
            "num_times": int(short_num_times),
            "controller": adaptive_controller,
        },
        {
            "name": "adaptive_fullmeta_short",
            "branch": "adaptive_exact",
            "spec": FixedManifoldRunSpec(
                name="adaptive_fullmeta_short",
                artifact_json=Path(FULL_META_ARTIFACT),
                loader_mode="replay_family",
                generator_family="match_adapt",
                fallback_family="full_meta",
            ),
            "t_final": float(short_t_final),
            "num_times": int(short_num_times),
            "controller": adaptive_controller,
        },
        {
            "name": "fixed_locked7_long",
            "branch": "fixed_exact",
            "spec": FixedManifoldRunSpec(
                name="fixed_locked7_long",
                artifact_json=Path(DEFAULT_LOCKED_7TERM_ARTIFACT),
                loader_mode="fixed_scaffold",
                generator_family="fixed_scaffold_locked",
                fallback_family="full_meta",
            ),
            "t_final": float(long_t_final),
            "num_times": int(long_num_times),
            "controller": dict(fixed_controller),
        },
        {
            "name": "fixed_pareto_long",
            "branch": "fixed_exact",
            "spec": FixedManifoldRunSpec(
                name="fixed_pareto_long",
                artifact_json=Path(DEFAULT_PARETO_ARTIFACT),
                loader_mode="replay_family",
                generator_family="match_adapt",
                fallback_family="full_meta",
            ),
            "t_final": float(long_t_final),
            "num_times": int(long_num_times),
            "controller": dict(fixed_controller),
        },
        {
            "name": "fixed_fullmeta_long",
            "branch": "fixed_exact",
            "spec": FixedManifoldRunSpec(
                name="fixed_fullmeta_long",
                artifact_json=Path(FULL_META_ARTIFACT),
                loader_mode="replay_family",
                generator_family="match_adapt",
                fallback_family="full_meta",
            ),
            "t_final": float(long_t_final),
            "num_times": int(long_num_times),
            "controller": dict(fixed_controller),
        },
        {
            "name": "adaptive_pareto_long",
            "branch": "adaptive_exact",
            "spec": FixedManifoldRunSpec(
                name="adaptive_pareto_long",
                artifact_json=Path(DEFAULT_PARETO_ARTIFACT),
                loader_mode="replay_family",
                generator_family="match_adapt",
                fallback_family="full_meta",
            ),
            "t_final": float(long_t_final),
            "num_times": int(long_num_times),
            "controller": adaptive_controller,
        },
        {
            "name": "adaptive_fullmeta_long",
            "branch": "adaptive_exact",
            "spec": FixedManifoldRunSpec(
                name="adaptive_fullmeta_long",
                artifact_json=Path(FULL_META_ARTIFACT),
                loader_mode="replay_family",
                generator_family="match_adapt",
                fallback_family="full_meta",
            ),
            "t_final": float(long_t_final),
            "num_times": int(long_num_times),
            "controller": adaptive_controller,
        },
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run static L=2 HH realtime McLachlan sweep from saved artifacts.")
    parser.add_argument("--tag", type=str, default=_default_tag())
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--summary-json", type=str, default=None)
    parser.add_argument("--progress-json", type=str, default=None)
    parser.add_argument("--short-t-final", type=float, default=10.0)
    parser.add_argument("--short-num-times", type=int, default=101)
    parser.add_argument("--long-t-final", type=float, default=20.0)
    parser.add_argument("--long-num-times", type=int, default=201)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir) if args.output_dir is not None else Path("artifacts/agent_runs") / str(args.tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = Path(args.summary_json) if args.summary_json is not None else output_dir / "summary.json"
    progress_json = Path(args.progress_json) if args.progress_json is not None else output_dir / "progress.json"

    cases = _case_specs(
        short_t_final=float(args.short_t_final),
        short_num_times=int(args.short_num_times),
        long_t_final=float(args.long_t_final),
        long_num_times=int(args.long_num_times),
    )
    records: list[dict[str, Any]] = []
    completed_names: set[str] = set()
    if bool(args.resume) and progress_json.exists():
        existing = json.loads(progress_json.read_text(encoding="utf-8"))
        if isinstance(existing, Mapping):
            for row in existing.get("runs", []):
                if isinstance(row, Mapping):
                    rec = dict(row)
                    records.append(rec)
                    completed_names.add(str(rec.get("name")))

    for case in cases:
        name = str(case["name"])
        if name in completed_names:
            continue
        try:
            if str(case["branch"]) == "fixed_exact":
                record = run_fixed_manifold_exact(
                    case["spec"],
                    tag=str(args.tag),
                    output_dir=output_dir,
                    t_final=float(case["t_final"]),
                    num_times=int(case["num_times"]),
                    miss_threshold=float(case["controller"]["miss_threshold"]),
                    gain_ratio_threshold=float(case["controller"]["gain_ratio_threshold"]),
                    append_margin_abs=float(case["controller"]["append_margin_abs"]),
                )
            else:
                record = _run_adaptive_exact_from_artifact(
                    case["spec"],
                    tag=str(args.tag),
                    output_path=output_dir / f"{name}.json",
                    t_final=float(case["t_final"]),
                    num_times=int(case["num_times"]),
                    controller_cfg=case["controller"],
                )
        except Exception as exc:
            record = {
                "name": name,
                "status": "failed",
                "branch": str(case["branch"]),
                "input_artifact_json": str(case["spec"].artifact_json),
                "error": repr(exc),
            }
        records.append(dict(record))
        completed_names.add(name)
        _write_json(progress_json, _summary_payload(tag=str(args.tag), output_dir=output_dir, records=records))

    summary = _summary_payload(tag=str(args.tag), output_dir=output_dir, records=records)
    _write_json(summary_json, summary)
    return 0 if int(summary["failed_runs"]) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
