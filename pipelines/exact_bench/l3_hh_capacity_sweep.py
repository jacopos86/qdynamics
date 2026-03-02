#!/usr/bin/env python3
"""Capacity sweep for L=3 HH convergence diagnostics.

This runner executes:
  1) scout pass over candidate configs
  2) per-method capacity ladders with early stall-stop heuristics

It does not change Hamiltonian/model definitions; it only orchestrates runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Path setup
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.overnight_l3_hh_four_method_benchmark import (  # noqa: E402
    AttemptConfig,
    METHOD_SPECS,
    SUMMARY_FIELDS,
    _execute_with_timeout,
)


DEFAULT_METHODS = ["m1_hh_hva", "m3_adapt_paop_std", "m4_adapt_paop_lf_std"]
DEFAULT_SCOUT_CONFIGS = ["interleaved,periodic,binary", "blocked,open,binary"]


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ai_log(event: str, **fields: Any) -> None:
    payload = {"event": str(event), "ts_utc": _now_utc_iso(), **fields}
    print(f"AI_LOG {json.dumps(payload, sort_keys=True, default=str)}", flush=True)


def _parse_sector(raw: str) -> tuple[int, int]:
    s = str(raw).strip().replace("(", "").replace(")", "")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Invalid sector '{raw}'. Use n_up,n_dn.")
    return int(parts[0]), int(parts[1])


def _parse_config(raw: str) -> tuple[str, str, str]:
    parts = [p.strip().lower() for p in str(raw).split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(
            f"Invalid config '{raw}'. Use ordering,boundary,boson_encoding."
        )
    ordering, boundary, encoding = parts
    if ordering not in {"blocked", "interleaved"}:
        raise ValueError(f"Unsupported ordering '{ordering}'.")
    if boundary not in {"open", "periodic"}:
        raise ValueError(f"Unsupported boundary '{boundary}'.")
    if encoding not in {"binary", "unary"}:
        raise ValueError(f"Unsupported boson encoding '{encoding}'.")
    return ordering, boundary, encoding


def _seed_list(seed_start: int, count: int) -> list[int]:
    return [int(seed_start) + i for i in range(int(count))]


def _method_kind(method_id: str) -> str:
    return str(METHOD_SPECS[str(method_id)]["kind"])


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _safe_float(x: Any) -> float | None:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def _safe_int(x: Any) -> int | None:
    try:
        if x is None or x == "":
            return None
        return int(x)
    except Exception:
        return None


# Built-in math:
#   score(config) = mean_m min_seed DeltaE_m(config), tie-broken by runtime
def _score_scout_config(rows: list[dict[str, Any]], methods: list[str]) -> tuple[float, float, int]:
    by_method: dict[str, list[dict[str, Any]]] = {m: [] for m in methods}
    timeout_count = 0
    for row in rows:
        mid = str(row.get("method_id", ""))
        if mid not in by_method:
            continue
        if str(row.get("status", "")) == "timeout":
            timeout_count += 1
        if str(row.get("status", "")) == "ok" and _safe_float(row.get("delta_E_abs")) is not None:
            by_method[mid].append(row)

    best_deltas: list[float] = []
    best_runtimes: list[float] = []
    for m in methods:
        candidates = by_method[m]
        if len(candidates) == 0:
            return (float("inf"), float("inf"), timeout_count)
        best = min(candidates, key=lambda r: float(r["delta_E_abs"]))
        best_deltas.append(float(best["delta_E_abs"]))
        best_runtimes.append(float(best["runtime_s"]))
    return (float(statistics.mean(best_deltas)), float(statistics.mean(best_runtimes)), timeout_count)


# Built-in math:
#   stop if two consecutive rung-improvements are both smaller than eps
def _small_improvement_stall(best_by_rung: list[float], eps: float) -> bool:
    if len(best_by_rung) < 3:
        return False
    d1 = best_by_rung[-3] - best_by_rung[-2]
    d2 = best_by_rung[-2] - best_by_rung[-1]
    return (d1 < float(eps)) and (d2 < float(eps))


# Built-in math:
#   plateau if same depth for last 3 and max(E_best)-min(E_best) < eps_e
def _adapt_plateau(last_ok_rows: list[dict[str, Any]], eps_e: float = 1e-5) -> bool:
    if len(last_ok_rows) < 3:
        return False
    tail = last_ok_rows[-3:]
    depths = [_safe_int(r.get("adapt_depth_reached")) for r in tail]
    if any(d is None for d in depths):
        return False
    if len(set(depths)) != 1:
        return False
    energies = [_safe_float(r.get("E_best")) for r in tail]
    if any(e is None for e in energies):
        return False
    return (max(energies) - min(energies)) < float(eps_e)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, None) for k in fieldnames})


def _write_markdown_summary(path: Path, payload: dict[str, Any]) -> None:
    _ensure_parent(path)
    lines: list[str] = []
    lines.append("# L=3 HH Capacity Sweep Summary")
    lines.append("")
    lines.append("## Run Meta")
    lines.append("")
    lines.append(f"- created_utc: `{payload['created_utc']}`")
    lines.append(f"- out_dir: `{payload['out_dir']}`")
    lines.append(f"- sector: `{payload['sector']}`")
    lines.append(f"- methods: `{', '.join(payload['methods'])}`")
    lines.append(f"- selected_config: `{payload['selected_config']}`")
    lines.append("")

    lines.append("## Scout Results")
    lines.append("")
    lines.append("| config | mean best ΔE | mean best runtime_s | timeouts |")
    lines.append("|---|---:|---:|---:|")
    for row in payload["scout_table"]:
        lines.append(
            f"| {row['config']} | {row['mean_best_delta']:.6e} | "
            f"{row['mean_best_runtime_s']:.1f} | {row['timeouts']} |"
        )
    lines.append("")

    lines.append("## Best Per Method")
    lines.append("")
    lines.append("| method | best ΔE | E_best | E_exact_sector | seed | rung | runtime_s |")
    lines.append("|---|---:|---:|---:|---:|---|---:|")
    for row in payload["best_per_method"]:
        if row["best_delta"] is None:
            lines.append(f"| {row['method']} | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| {row['method']} | {row['best_delta']:.6e} | {row['E_best']:.12f} | "
            f"{row['E_exact_sector']:.12f} | {row['seed']} | {row['rung']} | {row['runtime_s']:.1f} |"
        )
    lines.append("")

    lines.append("## Rung Progress")
    lines.append("")
    lines.append("| method | rung | n_runs | n_ok | best ΔE | median ΔE | median runtime_s | decision |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for row in payload["rung_table"]:
        best = "-" if row["best_delta"] is None else f"{row['best_delta']:.6e}"
        med = "-" if row["median_delta"] is None else f"{row['median_delta']:.6e}"
        rt = "-" if row["median_runtime_s"] is None else f"{row['median_runtime_s']:.1f}"
        lines.append(
            f"| {row['method']} | {row['rung']} | {row['n_runs']} | {row['n_ok']} | "
            f"{best} | {med} | {rt} | {row['decision']} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="L=3 HH capacity sweep with stall-stop heuristics.")
    p.add_argument("--L", type=int, default=3)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--U", type=float, default=2.0)
    p.add_argument("--dv", type=float, default=0.0)
    p.add_argument("--omega0", type=float, default=1.0)
    p.add_argument("--g-ep", type=float, default=1.0)
    p.add_argument("--n-ph-max", type=int, default=1)
    p.add_argument("--sector", type=str, default="2,1")
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    p.add_argument("--scout-configs", nargs="+", default=DEFAULT_SCOUT_CONFIGS)

    p.add_argument("--seed-start", type=int, default=101)
    p.add_argument("--scout-seeds", type=int, default=1)
    p.add_argument("--ladder-seeds", type=int, default=2)
    p.add_argument("--ladder-max-seeds", type=int, default=5)

    p.add_argument("--vqe-restarts", type=int, default=5)
    p.add_argument("--vqe-method", choices=["COBYLA", "SLSQP", "L-BFGS-B"], default="COBYLA")
    p.add_argument("--hva-reps-ladder", nargs="+", type=int, default=[2, 3, 4, 5])
    p.add_argument("--hva-maxiter-ladder", nargs="+", type=int, default=[3000, 4500, 6000, 6000])

    p.add_argument("--adapt-max-depth-ladder", nargs="+", type=int, default=[80, 120, 160, 160])
    p.add_argument("--adapt-maxiter-ladder", nargs="+", type=int, default=[3000, 3000, 3000, 4500])
    p.add_argument("--adapt-eps-grad", type=float, default=1e-6)
    p.add_argument("--adapt-eps-energy", type=float, default=1e-8)
    p.add_argument("--adapt-allow-repeats", action="store_true")
    p.add_argument("--adapt-no-repeats", dest="adapt_allow_repeats", action="store_false")
    p.set_defaults(adapt_allow_repeats=True)

    p.add_argument("--paop-r", type=int, default=1)
    p.add_argument("--paop-prune-eps", type=float, default=0.0)
    p.add_argument("--paop-normalization", choices=["none", "fro", "maxcoeff"], default="none")

    p.add_argument("--vqe-cap-s", type=int, default=900)
    p.add_argument("--adapt-cap-s", type=int, default=1200)

    p.add_argument("--stall-improve-eps", type=float, default=1e-2)
    p.add_argument("--stall-time-gain-eps", type=float, default=5e-3)
    p.add_argument("--success-delta", type=float, default=1e-3)

    p.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "l3_hh_capacity_sweep",
    )
    p.add_argument("--tag", type=str, default="run")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if int(args.L) != 3:
        raise ValueError("This sweep is intended for L=3.")
    if str(args.vqe_method).upper() != "COBYLA":
        raise ValueError("This sweep is configured for COBYLA for L=3 HH consistency.")
    if int(args.vqe_restarts) < 4:
        raise ValueError("Use vqe_restarts >= 4 for HH L=3.")
    if any(int(x) < 2 for x in args.hva_reps_ladder):
        raise ValueError("All HVA reps values must be >= 2 for HH L=3.")
    if any(int(x) < 2400 for x in args.hva_maxiter_ladder):
        raise ValueError("All HVA maxiter values must be >= 2400 for HH L=3.")
    if any(int(x) < 2400 for x in args.adapt_maxiter_ladder):
        raise ValueError("All ADAPT maxiter values must be >= 2400 for HH L=3.")

    methods = [m for m in args.methods if m in METHOD_SPECS]
    if sorted(methods) != sorted(DEFAULT_METHODS):
        raise ValueError(f"methods must be exactly {DEFAULT_METHODS}. Got {methods}")

    n_up, n_dn = _parse_sector(str(args.sector))
    scout_configs = [_parse_config(c) for c in args.scout_configs]
    scout_seeds = _seed_list(int(args.seed_start), int(args.scout_seeds))
    ladder_seeds = _seed_list(int(args.seed_start), int(args.ladder_seeds))
    extra_ladder_seeds = _seed_list(
        int(args.seed_start) + int(args.ladder_seeds),
        max(0, int(args.ladder_max_seeds) - int(args.ladder_seeds)),
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_root) / f"L3_hh_capacity_{args.tag}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    _ai_log("capacity_sweep_start", out_dir=str(out_dir))
    meta = {
        "created_utc": _now_utc_iso(),
        "script": str(Path(__file__).resolve()),
        "out_dir": str(out_dir),
        "sector": [int(n_up), int(n_dn)],
        "methods": methods,
        "scout_configs": [",".join(c) for c in scout_configs],
        "scout_seeds": scout_seeds,
        "ladder_seeds": ladder_seeds,
        "extra_ladder_seeds": extra_ladder_seeds,
        "physics": {
            "L": int(args.L),
            "problem": "hh",
            "t": float(args.t),
            "U": float(args.U),
            "dv": float(args.dv),
            "omega0": float(args.omega0),
            "g_ep": float(args.g_ep),
            "n_ph_max": int(args.n_ph_max),
        },
        "knobs": {
            "vqe_restarts": int(args.vqe_restarts),
            "vqe_method": str(args.vqe_method),
            "hva_reps_ladder": [int(x) for x in args.hva_reps_ladder],
            "hva_maxiter_ladder": [int(x) for x in args.hva_maxiter_ladder],
            "adapt_max_depth_ladder": [int(x) for x in args.adapt_max_depth_ladder],
            "adapt_maxiter_ladder": [int(x) for x in args.adapt_maxiter_ladder],
            "adapt_eps_grad": float(args.adapt_eps_grad),
            "adapt_eps_energy": float(args.adapt_eps_energy),
            "adapt_allow_repeats": bool(args.adapt_allow_repeats),
            "paop_r": int(args.paop_r),
            "paop_prune_eps": float(args.paop_prune_eps),
            "paop_normalization": str(args.paop_normalization),
            "vqe_cap_s": int(args.vqe_cap_s),
            "adapt_cap_s": int(args.adapt_cap_s),
            "stall_improve_eps": float(args.stall_improve_eps),
            "stall_time_gain_eps": float(args.stall_time_gain_eps),
            "success_delta": float(args.success_delta),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    common_fields = [
        "phase",
        "rung",
        "ordering",
        "boundary",
        "boson_encoding",
        "method_id_requested",
        "seed_requested",
        "vqe_reps_used",
        "vqe_maxiter_used",
        "adapt_max_depth_used",
        "adapt_maxiter_used",
        "decision",
    ]
    csv_fields = common_fields + SUMMARY_FIELDS
    all_rows: list[dict[str, Any]] = []
    attempt_idx = 0

    def run_one(
        *,
        phase: str,
        rung: str,
        ordering: str,
        boundary: str,
        boson_encoding: str,
        method_id: str,
        seed: int,
        vqe_reps: int,
        vqe_maxiter: int,
        adapt_max_depth: int,
        adapt_maxiter: int,
        decision: str = "",
    ) -> dict[str, Any]:
        nonlocal attempt_idx
        attempt_idx += 1
        wallclock = int(args.adapt_cap_s if _method_kind(method_id) == "adapt" else args.vqe_cap_s)
        run_id = (
            f"{phase}|{method_id}|{rung}|S{n_up}_{n_dn}|"
            f"{ordering}_{boundary}|{boson_encoding}|seed{seed}"
        )
        cfg = AttemptConfig(
            run_id=str(run_id),
            method_id=str(method_id),
            sector_n_up=int(n_up),
            sector_n_dn=int(n_dn),
            seed=int(seed),
            attempt_idx=int(attempt_idx),
            smoke_test=False,
            L=int(args.L),
            t=float(args.t),
            U=float(args.U),
            dv=float(args.dv),
            omega0=float(args.omega0),
            g_ep=float(args.g_ep),
            n_ph_max=int(args.n_ph_max),
            boson_encoding=str(boson_encoding),
            ordering=str(ordering),
            boundary=str(boundary),
            vqe_reps=int(vqe_reps),
            vqe_restarts=int(args.vqe_restarts),
            vqe_maxiter=int(vqe_maxiter),
            vqe_method=str(args.vqe_method),
            adapt_max_depth=int(adapt_max_depth),
            adapt_eps_grad=float(args.adapt_eps_grad),
            adapt_eps_energy=float(args.adapt_eps_energy),
            adapt_maxiter=int(adapt_maxiter),
            paop_r=int(args.paop_r),
            paop_split_paulis=False,
            paop_prune_eps=float(args.paop_prune_eps),
            paop_normalization=str(args.paop_normalization),
            adapt_allow_repeats=bool(args.adapt_allow_repeats),
            wallclock_cap_s=max(60, int(wallclock)),
        )
        _ai_log(
            "capacity_attempt_start",
            phase=phase,
            rung=rung,
            method_id=method_id,
            seed=seed,
            ordering=ordering,
            boundary=boundary,
            boson_encoding=boson_encoding,
            attempt_idx=attempt_idx,
        )
        result = _execute_with_timeout(cfg)
        row = {
            "phase": str(phase),
            "rung": str(rung),
            "ordering": str(ordering),
            "boundary": str(boundary),
            "boson_encoding": str(boson_encoding),
            "method_id_requested": str(method_id),
            "seed_requested": int(seed),
            "vqe_reps_used": int(vqe_reps),
            "vqe_maxiter_used": int(vqe_maxiter),
            "adapt_max_depth_used": int(adapt_max_depth),
            "adapt_maxiter_used": int(adapt_maxiter),
            "decision": str(decision),
            **result,
        }
        all_rows.append(row)
        _ai_log(
            "capacity_attempt_done",
            phase=phase,
            rung=rung,
            method_id=method_id,
            seed=seed,
            status=row.get("status"),
            delta_E_abs=row.get("delta_E_abs"),
            runtime_s=row.get("runtime_s"),
        )
        return row

    # ---------------- Scout phase ----------------
    scout_rows: list[dict[str, Any]] = []
    scout_scores: list[dict[str, Any]] = []
    for cfg in scout_configs:
        ordering, boundary, boson_encoding = cfg
        config_rows: list[dict[str, Any]] = []
        for method_id in methods:
            for seed in scout_seeds:
                row = run_one(
                    phase="scout",
                    rung="scout_base",
                    ordering=ordering,
                    boundary=boundary,
                    boson_encoding=boson_encoding,
                    method_id=method_id,
                    seed=seed,
                    vqe_reps=2,
                    vqe_maxiter=3000,
                    adapt_max_depth=80,
                    adapt_maxiter=3000,
                )
                config_rows.append(row)
                scout_rows.append(row)
        score = _score_scout_config(config_rows, methods)
        scout_scores.append(
            {
                "config": f"{ordering},{boundary},{boson_encoding}",
                "rows": config_rows,
                "score_delta": score[0],
                "score_runtime": score[1],
                "timeouts": score[2],
            }
        )

    scout_scores.sort(key=lambda x: (x["score_delta"], x["score_runtime"], x["timeouts"]))
    selected = scout_scores[0]
    selected_ordering, selected_boundary, selected_encoding = _parse_config(selected["config"])
    _ai_log(
        "scout_selected_config",
        config=selected["config"],
        score_delta=selected["score_delta"],
        score_runtime=selected["score_runtime"],
        timeouts=selected["timeouts"],
    )

    # ---------------- Ladder phase ----------------
    rung_records: list[dict[str, Any]] = []
    method_best_rows: dict[str, dict[str, Any]] = {}
    method_stop: dict[str, str] = {}

    for method_id in methods:
        _ai_log("ladder_method_start", method_id=method_id, config=selected["config"])
        best_by_rung: list[float] = []
        runtime_by_rung: list[float] = []
        recent_ok_for_plateau: list[dict[str, Any]] = []
        previous_best: float | None = None

        if method_id == "m1_hh_hva":
            ladder = list(zip(args.hva_reps_ladder, args.hva_maxiter_ladder))
        else:
            ladder = list(zip(args.adapt_max_depth_ladder, args.adapt_maxiter_ladder))

        for rung_idx, rung_item in enumerate(ladder, start=1):
            if method_id == "m1_hh_hva":
                reps = int(rung_item[0])
                vqe_maxiter = int(rung_item[1])
                adapt_depth = 80
                adapt_maxiter = 3000
                rung_name = f"hva_r{reps}_mi{vqe_maxiter}"
            else:
                reps = 2
                vqe_maxiter = 3000
                adapt_depth = int(rung_item[0])
                adapt_maxiter = int(rung_item[1])
                rung_name = f"adapt_d{adapt_depth}_mi{adapt_maxiter}"

            rung_rows: list[dict[str, Any]] = []
            for seed in ladder_seeds:
                rung_rows.append(
                    run_one(
                        phase="ladder",
                        rung=rung_name,
                        ordering=selected_ordering,
                        boundary=selected_boundary,
                        boson_encoding=selected_encoding,
                        method_id=method_id,
                        seed=seed,
                        vqe_reps=reps,
                        vqe_maxiter=vqe_maxiter,
                        adapt_max_depth=adapt_depth,
                        adapt_maxiter=adapt_maxiter,
                    )
                )

            ok_rows = [r for r in rung_rows if str(r.get("status")) == "ok" and _safe_float(r.get("delta_E_abs")) is not None]
            if len(ok_rows) > 0:
                best_row = min(ok_rows, key=lambda r: float(r["delta_E_abs"]))
                best_delta = float(best_row["delta_E_abs"])
                median_delta = float(statistics.median(float(r["delta_E_abs"]) for r in ok_rows))
                median_runtime = float(statistics.median(float(r["runtime_s"]) for r in ok_rows))
                best_by_rung.append(best_delta)
                runtime_by_rung.append(median_runtime)
                recent_ok_for_plateau.extend(ok_rows)
                if method_id not in method_best_rows:
                    method_best_rows[method_id] = best_row
                else:
                    if float(best_row["delta_E_abs"]) < float(method_best_rows[method_id]["delta_E_abs"]):
                        method_best_rows[method_id] = best_row
            else:
                best_delta = None
                median_delta = None
                median_runtime = None

            decision = "continue"
            if best_delta is not None and best_delta <= float(args.success_delta):
                decision = "stop_success_gate"
                method_stop[method_id] = decision
            elif len(best_by_rung) >= 3 and _small_improvement_stall(best_by_rung, float(args.stall_improve_eps)):
                decision = "stop_small_improvement_twice"
                method_stop[method_id] = decision
            elif (
                len(best_by_rung) >= 2
                and len(runtime_by_rung) >= 2
                and runtime_by_rung[-1] > 2.0 * runtime_by_rung[-2]
                and (best_by_rung[-2] - best_by_rung[-1]) < float(args.stall_time_gain_eps)
            ):
                decision = "stop_runtime_inefficient"
                method_stop[method_id] = decision
            elif method_id in {"m3_adapt_paop_std", "m4_adapt_paop_lf_std"} and _adapt_plateau(recent_ok_for_plateau):
                decision = "stop_adapt_plateau"
                method_stop[method_id] = decision

            rung_records.append(
                {
                    "method": method_id,
                    "rung": rung_name,
                    "n_runs": len(rung_rows),
                    "n_ok": len(ok_rows),
                    "best_delta": best_delta,
                    "median_delta": median_delta,
                    "median_runtime_s": median_runtime,
                    "decision": decision,
                }
            )

            # Optional seed expansion only when trend is productive.
            if (
                decision == "continue"
                and len(extra_ladder_seeds) > 0
                and best_delta is not None
                and previous_best is not None
                and (previous_best - best_delta) >= 0.05
            ):
                for seed in extra_ladder_seeds:
                    extra_row = run_one(
                        phase="ladder_extra",
                        rung=rung_name,
                        ordering=selected_ordering,
                        boundary=selected_boundary,
                        boson_encoding=selected_encoding,
                        method_id=method_id,
                        seed=seed,
                        vqe_reps=reps,
                        vqe_maxiter=vqe_maxiter,
                        adapt_max_depth=adapt_depth,
                        adapt_maxiter=adapt_maxiter,
                    )
                    if (
                        str(extra_row.get("status")) == "ok"
                        and _safe_float(extra_row.get("delta_E_abs")) is not None
                        and method_id in method_best_rows
                        and float(extra_row["delta_E_abs"]) < float(method_best_rows[method_id]["delta_E_abs"])
                    ):
                        method_best_rows[method_id] = extra_row

            if best_delta is not None:
                previous_best = best_delta

            if decision.startswith("stop_"):
                _ai_log("ladder_method_stop", method_id=method_id, rung=rung_name, decision=decision)
                break

        if method_id not in method_stop:
            method_stop[method_id] = "max_ladder_reached"
        _ai_log("ladder_method_done", method_id=method_id, decision=method_stop[method_id])

    # Persist rows
    _write_csv(summary_dir / "capacity_sweep_runs.csv", all_rows, csv_fields)
    _write_csv(summary_dir / "scout_rows.csv", scout_rows, csv_fields)

    scout_table = []
    for item in scout_scores:
        scout_table.append(
            {
                "config": item["config"],
                "mean_best_delta": float(item["score_delta"]),
                "mean_best_runtime_s": float(item["score_runtime"]),
                "timeouts": int(item["timeouts"]),
            }
        )

    best_per_method = []
    for method_id in methods:
        br = method_best_rows.get(method_id)
        if br is None:
            best_per_method.append(
                {
                    "method": method_id,
                    "best_delta": None,
                    "E_best": None,
                    "E_exact_sector": None,
                    "seed": None,
                    "rung": None,
                    "runtime_s": None,
                }
            )
            continue
        best_per_method.append(
            {
                "method": method_id,
                "best_delta": float(br["delta_E_abs"]),
                "E_best": float(br["E_best"]),
                "E_exact_sector": float(br["E_exact_sector"]),
                "seed": int(br["seed"]),
                "rung": str(br["rung"]),
                "runtime_s": float(br["runtime_s"]),
            }
        )

    payload = {
        "created_utc": _now_utc_iso(),
        "out_dir": str(out_dir),
        "sector": f"({n_up},{n_dn})",
        "methods": methods,
        "selected_config": selected["config"],
        "method_stop": method_stop,
        "scout_table": scout_table,
        "best_per_method": best_per_method,
        "rung_table": rung_records,
    }
    (summary_dir / "capacity_sweep_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    _write_markdown_summary(summary_dir / "capacity_sweep_summary.md", payload)

    _ai_log(
        "capacity_sweep_done",
        out_dir=str(out_dir),
        total_rows=len(all_rows),
        selected_config=selected["config"],
    )


if __name__ == "__main__":
    main()
