#!/usr/bin/env python3
"""Driven L=2 HH fixed-manifold McLachlan sweep.

Purpose:
- driven only
- compare lean vs rich fixed manifolds
- compare unaugmented vs drive-augmented full-meta variants
- cover short and long horizons
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.exact_bench.noise_oracle_runtime import OracleConfig
from pipelines.hardcoded.hh_fixed_manifold_mclachlan import (
    DEFAULT_LOCKED_7TERM_ARTIFACT,
    DEFAULT_PARETO_ARTIFACT,
    FixedManifoldRunSpec,
)
from pipelines.hardcoded.hh_fixed_manifold_measured import (
    FixedManifoldAugmentationConfig,
    FixedManifoldDriveConfig,
    FixedManifoldMeasuredConfig,
    run_fixed_manifold_measured,
)

FULL_META_ARTIFACT = Path("artifacts/json/campaign_A7f_L2_shortlist_off_runtime_split_off_phase3_v1.json")


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_hh_l2_driven_realtime_pareto_sweep")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _record_metrics(record: Mapping[str, Any]) -> dict[str, Any]:
    summary = record.get("summary", {}) if isinstance(record.get("summary", {}), Mapping) else {}
    fidelity_min = float(summary.get("min_fidelity_exact_audit", float("nan")))
    energy_err_max = float(summary.get("max_abs_energy_total_error_exact_audit", float("nan")))
    runtime_count = int(summary.get("runtime_parameter_count", 10**9))
    cond_max = float(summary.get("max_condition_number", float("nan")))
    return {
        "fidelity_gap": float(max(0.0, 1.0 - fidelity_min)) if np.isfinite(fidelity_min) else float("inf"),
        "energy_err_max": energy_err_max,
        "runtime_count": runtime_count,
        "condition_max": cond_max,
    }


def _dominates(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> bool:
    lhs_m = _record_metrics(lhs)
    rhs_m = _record_metrics(rhs)
    lhs_vals = [float(lhs_m["fidelity_gap"]), float(lhs_m["energy_err_max"]), float(lhs_m["runtime_count"])]
    rhs_vals = [float(rhs_m["fidelity_gap"]), float(rhs_m["energy_err_max"]), float(rhs_m["runtime_count"])]
    return all(l <= r for l, r in zip(lhs_vals, rhs_vals)) and any(l < r for l, r in zip(lhs_vals, rhs_vals))


def _pareto_frontier(records: Sequence[Mapping[str, Any]]) -> list[str]:
    completed = [dict(r) for r in records if str(r.get("status", "")) == "completed"]
    out: list[str] = []
    for row in completed:
        if any(_dominates(other, row) for other in completed if other is not row):
            continue
        out.append(str(row.get("name")))
    return sorted(out)


def _render_png(summary_payload: Mapping[str, Any], output_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception:
        return

    rows = list(summary_payload.get("runs", []))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {
        "locked7": "tab:blue",
        "pareto": "tab:green",
        "fullmeta": "tab:orange",
        "aligned": "tab:red",
        "aligned_stab": "tab:purple",
    }
    markers = {"short": "o", "long": "s"}
    for row in rows:
        if str(row.get("status", "")) != "completed":
            continue
        summary = row.get("summary", {})
        name = str(row.get("name"))
        family = str(row.get("family_key"))
        horizon = str(row.get("horizon"))
        x = int(summary.get("runtime_parameter_count", 0))
        y1 = float(summary.get("min_fidelity_exact_audit", float("nan")))
        y2 = float(summary.get("max_abs_energy_total_error_exact_audit", float("nan")))
        axes[0].scatter(x, y1, color=colors.get(family, "black"), marker=markers.get(horizon, "o"), s=70)
        axes[1].scatter(x, y2, color=colors.get(family, "black"), marker=markers.get(horizon, "o"), s=70)
        axes[0].annotate(name, (x, y1), textcoords='offset points', xytext=(4,4), fontsize=7)
        axes[1].annotate(name, (x, y2), textcoords='offset points', xytext=(4,4), fontsize=7)

    axes[0].set_title('Runtime params vs min fidelity')
    axes[0].set_xlabel('Runtime parameter count')
    axes[0].set_ylabel('Min fidelity vs exact')
    axes[0].grid(alpha=0.3)

    axes[1].set_title('Runtime params vs max |ΔE|')
    axes[1].set_xlabel('Runtime parameter count')
    axes[1].set_ylabel('Max |ΔE| vs exact')
    axes[1].set_yscale('log')
    axes[1].grid(alpha=0.3)

    legend_items = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='tab:blue', markersize=8, label='locked7 short'),
        Line2D([0],[0], marker='s', color='w', markerfacecolor='tab:blue', markersize=8, label='locked7 long'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='tab:green', markersize=8, label='pareto short'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='tab:orange', markersize=8, label='fullmeta short'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='tab:red', markersize=8, label='aligned short'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='tab:purple', markersize=8, label='aligned_stab short'),
    ]
    axes[0].legend(handles=legend_items, fontsize=8, loc='lower right')
    fig.suptitle('HH L=2 driven realtime McLachlan sweep', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def _run_case(case: Mapping[str, Any], *, tag: str, output_dir: Path) -> dict[str, Any]:
    output_json = output_dir / f"{case['name']}.json"
    payload = run_fixed_manifold_measured(
        case["spec"],
        tag=str(tag),
        output_json=output_json,
        t_final=float(case["t_final"]),
        num_times=int(case["num_times"]),
        oracle_cfg=case["oracle_cfg"],
        geom_cfg=case["geom_cfg"],
        drive_cfg=case["drive_cfg"],
        aug_cfg=case["aug_cfg"],
    )
    return {
        "name": str(case["name"]),
        "status": "completed",
        "family_key": str(case["family_key"]),
        "horizon": str(case["horizon"]),
        "input_artifact_json": str(case["spec"].artifact_json),
        "output_json": str(output_json),
        "summary": dict(payload.get("summary", {})),
        "loader": dict(payload.get("loader", {})),
        "manifest": dict(payload.get("manifest", {})),
    }


def _summary_payload(*, tag: str, output_dir: Path, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [dict(r) for r in records if str(r.get("status", "")) == "completed"]
    failed = [dict(r) for r in records if str(r.get("status", "")) != "completed"]
    best_fidelity = None if not completed else max(
        completed,
        key=lambda r: float(r.get("summary", {}).get("min_fidelity_exact_audit", float("-inf"))),
    )
    leanest = None if not completed else min(
        completed,
        key=lambda r: int(r.get("summary", {}).get("runtime_parameter_count", 10**9)),
    )
    return {
        "generated_utc": _now_utc(),
        "pipeline": "hh_l2_driven_realtime_pareto_sweep_v1",
        "tag": str(tag),
        "output_dir": str(output_dir),
        "completed_runs": int(len(completed)),
        "failed_runs": int(len(failed)),
        "frontier_names": _pareto_frontier(completed),
        "best_min_fidelity_run": None if best_fidelity is None else str(best_fidelity.get("name")),
        "leanest_runtime_run": None if leanest is None else str(leanest.get("name")),
        "runs": [dict(r) for r in records],
    }


def _case_specs(*, short_t_final: float, short_num_times: int, long_t_final: float, long_num_times: int) -> list[dict[str, Any]]:
    oracle_cfg = OracleConfig(noise_mode='ideal', shots=2048, seed=7, oracle_repeats=1, oracle_aggregate='mean')
    drive_cfg = FixedManifoldDriveConfig(
        enable_drive=True,
        drive_A=0.6,
        drive_omega=1.0,
        drive_tbar=1.0,
        drive_phi=0.0,
        drive_pattern='staggered',
        drive_custom_s=None,
        drive_include_identity=False,
        drive_time_sampling='midpoint',
        drive_t0=0.0,
        exact_steps_multiplier=2,
    )
    geom_base = FixedManifoldMeasuredConfig(
        regularization_lambda=1.0e-8,
        pinv_rcond=1.0e-10,
        observable_drop_abs_tol=1.0e-12,
        observable_hermiticity_tol=1.0e-10,
        observable_max_terms=1024,
        variance_floor=0.0,
        g_symmetrize_tol=1.0e-12,
    )
    geom_stab = FixedManifoldMeasuredConfig(
        regularization_lambda=1.0e-4,
        pinv_rcond=1.0e-10,
        observable_drop_abs_tol=1.0e-12,
        observable_hermiticity_tol=1.0e-10,
        observable_max_terms=1024,
        variance_floor=0.0,
        g_symmetrize_tol=1.0e-12,
    )
    def mk(name, family_key, horizon, artifact, loader_mode, t_final, num_times, aug_mode='none', geom_cfg=geom_base):
        return {
            'name': name,
            'family_key': family_key,
            'horizon': horizon,
            'spec': FixedManifoldRunSpec(
                name=name,
                artifact_json=Path(artifact),
                loader_mode=loader_mode,
                generator_family=('fixed_scaffold_locked' if loader_mode == 'fixed_scaffold' else 'match_adapt'),
                fallback_family='full_meta',
            ),
            't_final': float(t_final),
            'num_times': int(num_times),
            'oracle_cfg': oracle_cfg,
            'geom_cfg': geom_cfg,
            'drive_cfg': drive_cfg,
            'aug_cfg': FixedManifoldAugmentationConfig(drive_generator_mode=str(aug_mode)),
        }
    cases = []
    for horizon, t_final, num_times in [
        ('short', short_t_final, short_num_times),
        ('long', long_t_final, long_num_times),
    ]:
        cases.extend([
            mk(f'driven_locked7_{horizon}', 'locked7', horizon, DEFAULT_LOCKED_7TERM_ARTIFACT, 'fixed_scaffold', t_final, num_times),
            mk(f'driven_pareto_{horizon}', 'pareto', horizon, DEFAULT_PARETO_ARTIFACT, 'replay_family', t_final, num_times),
            mk(f'driven_fullmeta_{horizon}', 'fullmeta', horizon, FULL_META_ARTIFACT, 'replay_family', t_final, num_times),
            mk(f'driven_fullmeta_aligned_{horizon}', 'aligned', horizon, FULL_META_ARTIFACT, 'replay_family', t_final, num_times, aug_mode='aligned_density', geom_cfg=geom_base),
            mk(f'driven_fullmeta_aligned_stab_{horizon}', 'aligned_stab', horizon, FULL_META_ARTIFACT, 'replay_family', t_final, num_times, aug_mode='aligned_density', geom_cfg=geom_stab),
        ])
    return cases


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run driven L=2 HH fixed-manifold McLachlan sweep.')
    parser.add_argument('--tag', type=str, default=_default_tag())
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--summary-json', type=str, default=None)
    parser.add_argument('--progress-json', type=str, default=None)
    parser.add_argument('--plot-png', type=str, default=None)
    parser.add_argument('--short-t-final', type=float, default=10.0)
    parser.add_argument('--short-num-times', type=int, default=101)
    parser.add_argument('--long-t-final', type=float, default=20.0)
    parser.add_argument('--long-num-times', type=int, default=201)
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir) if args.output_dir is not None else Path('artifacts/agent_runs') / str(args.tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = Path(args.summary_json) if args.summary_json is not None else output_dir / 'summary.json'
    progress_json = Path(args.progress_json) if args.progress_json is not None else output_dir / 'progress.json'
    plot_png = Path(args.plot_png) if args.plot_png is not None else Path('artifacts/reports') / f"{args.tag}.png"

    cases = _case_specs(
        short_t_final=float(args.short_t_final),
        short_num_times=int(args.short_num_times),
        long_t_final=float(args.long_t_final),
        long_num_times=int(args.long_num_times),
    )
    records: list[dict[str, Any]] = []
    completed_names: set[str] = set()
    if bool(args.resume) and progress_json.exists():
        existing = json.loads(progress_json.read_text(encoding='utf-8'))
        if isinstance(existing, Mapping):
            for row in existing.get('runs', []):
                if isinstance(row, Mapping):
                    rec = dict(row)
                    records.append(rec)
                    completed_names.add(str(rec.get('name')))

    for case in cases:
        name = str(case['name'])
        if name in completed_names:
            continue
        try:
            record = _run_case(case, tag=str(args.tag), output_dir=output_dir)
        except Exception as exc:
            record = {
                'name': name,
                'status': 'failed',
                'family_key': str(case['family_key']),
                'horizon': str(case['horizon']),
                'input_artifact_json': str(case['spec'].artifact_json),
                'error': repr(exc),
            }
        records.append(dict(record))
        completed_names.add(name)
        progress = _summary_payload(tag=str(args.tag), output_dir=output_dir, records=records)
        _write_json(progress_json, progress)
        _render_png(progress, plot_png)

    summary = _summary_payload(tag=str(args.tag), output_dir=output_dir, records=records)
    _write_json(summary_json, summary)
    _render_png(summary, plot_png)
    return 0 if int(summary['failed_runs']) == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
