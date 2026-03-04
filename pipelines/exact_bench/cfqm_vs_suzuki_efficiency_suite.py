#!/usr/bin/env python3
"""CFQM vs Suzuki cost-efficiency benchmark suite.

Benchmark objective:
- quantify error-vs-cost efficiency at fixed target accuracy for driven Hubbard/HH
- keep integrator-order and hardware-proxy comparisons separated

Math summary:
- slope fit: log(error) = a + p*log(dt)
- proxy 2q term model: cx_proxy_term(p) = 2 * max(weight(p)-1, 0)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reports.pdf_utils import (  # noqa: E402
    current_command_string,
    get_PdfPages,
    get_plt,
    render_command_page,
    render_compact_table,
    render_text_page,
    require_matplotlib,
)
from pipelines.exact_bench.cfqm_vs_suzuki_qproc_proxy_benchmark import (  # noqa: E402
    _build_drive_provider_from_settings,
    _compute_cfqm_proxy_cost,
    _compute_suzuki_proxy_cost,
    _extract_ordered_static_maps,
)


@dataclass(frozen=True)
class BenchmarkScenario:
    scenario_id: str
    problem: str
    L: int
    n_ph_max: int | None
    t: float
    u: float
    dv: float
    omega0: float | None
    g_ep: float | None


@dataclass(frozen=True)
class BenchmarkRunRecord:
    run_id: str
    scenario_id: str
    problem: str
    drive_case_id: str
    track: str
    stage_mode: str
    method: str
    trotter_steps: int
    dt: float
    reference_steps: int
    final_infidelity: float
    max_doublon_abs_err: float
    max_energy_abs_err: float
    wall_time_s: float
    expm_multiply_calls_method: int
    expm_multiply_calls_reference: int
    expm_multiply_calls_total: int
    pauli_rot_count_total: int
    cx_proxy_total: int
    depth_proxy_total: int
    sq_proxy_total: int
    transpiled_2q_count: int | None
    transpiled_depth: int | None

    def to_public_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EfficiencyConfig:
    problem_grid: tuple[str, ...]
    drive_grid: tuple[str, ...]
    methods: tuple[str, ...]
    stage_mode_grid: tuple[str, ...]
    steps_grid_override: tuple[int, ...] | None
    reference_steps_multiplier: int
    error_metrics: tuple[str, ...]
    cost_metrics: tuple[str, ...]
    equal_cost_axis: tuple[str, ...]
    equal_cost_policy: str
    calibrate_transpile: bool
    calibration_backend: str
    calibration_strict: bool
    output_dir: Path
    include_fallback_appendix: bool

    boundary: str
    ordering: str
    t_final: float
    num_times: int
    drive_A: float
    drive_phi: float
    drive_include_identity: bool
    drive_time_sampling: str
    sinusoid_omegas: tuple[float, ...]
    gaussian_tbars: tuple[float, ...]

    initial_state_source: str
    adapt_input_json: Path | None
    adapt_strict_match: bool
    vqe_ansatz: str
    vqe_reps: int
    vqe_restarts: int
    vqe_maxiter: int
    vqe_method: str
    adapt_pool: str
    adapt_max_depth: int
    adapt_maxiter: int


_SUPPORTED_PROBLEMS = ("hubbard_L4", "hh_L2_nb1", "hh_L2_nb2", "hh_L2_nb3", "hh_L3_nb1")
_SUPPORTED_DRIVES = ("sinusoid", "gaussian_sharp")
_SUPPORTED_METHODS = ("suzuki2", "cfqm4", "cfqm6")
_SUPPORTED_STAGE_MODES = ("exact_sparse", "exact_dense", "pauli_suzuki2")
_SUPPORTED_ERROR_METRICS = (
    "final_infidelity",
    "max_doublon_abs_err",
    "max_energy_abs_err",
)
_SUPPORTED_COST_METRICS = (
    "expm_calls",
    "pauli_rot_count",
    "cx_proxy",
    "depth_proxy",
    "wall_time",
)
_SUPPORTED_EQUAL_COST_AXIS = ("cx_proxy", "pauli_rot_count", "expm_calls", "wall_time")


def _parse_csv(raw: str) -> tuple[str, ...]:
    vals = tuple(tok.strip() for tok in str(raw).split(",") if tok.strip())
    if not vals:
        raise ValueError("Expected non-empty CSV list.")
    return vals


def _parse_csv_ints(raw: str) -> tuple[int, ...]:
    vals = tuple(int(tok.strip()) for tok in str(raw).split(",") if tok.strip())
    if not vals:
        raise ValueError("Expected non-empty CSV integer list.")
    return vals


def _parse_csv_positive_floats(raw: str) -> tuple[float, ...]:
    vals = tuple(float(tok.strip()) for tok in str(raw).split(",") if tok.strip())
    if not vals:
        raise ValueError("Expected non-empty CSV float list.")
    for v in vals:
        if not math.isfinite(float(v)) or float(v) <= 0.0:
            raise ValueError(f"Expected positive finite float values; got {v}.")
    return vals


def _expand_scenarios(problem_grid: Sequence[str]) -> list[BenchmarkScenario]:
    out: list[BenchmarkScenario] = []
    for token in problem_grid:
        key = str(token).strip()
        if key == "hubbard_L4":
            out.append(
                BenchmarkScenario(
                    scenario_id="hubbard_L4",
                    problem="hubbard",
                    L=4,
                    n_ph_max=None,
                    t=1.0,
                    u=4.0,
                    dv=0.0,
                    omega0=None,
                    g_ep=None,
                )
            )
        elif key == "hh_L2_nb2":
            out.append(
                BenchmarkScenario(
                    scenario_id="hh_L2_nb2",
                    problem="hh",
                    L=2,
                    n_ph_max=2,
                    t=1.0,
                    u=2.0,
                    dv=0.0,
                    omega0=1.0,
                    g_ep=0.5,
                )
            )
        elif key == "hh_L2_nb1":
            out.append(
                BenchmarkScenario(
                    scenario_id="hh_L2_nb1",
                    problem="hh",
                    L=2,
                    n_ph_max=1,
                    t=1.0,
                    u=2.0,
                    dv=0.0,
                    omega0=1.0,
                    g_ep=0.5,
                )
            )
        elif key == "hh_L2_nb3":
            out.append(
                BenchmarkScenario(
                    scenario_id="hh_L2_nb3",
                    problem="hh",
                    L=2,
                    n_ph_max=3,
                    t=1.0,
                    u=2.0,
                    dv=0.0,
                    omega0=1.0,
                    g_ep=0.5,
                )
            )
        elif key == "hh_L3_nb1":
            out.append(
                BenchmarkScenario(
                    scenario_id="hh_L3_nb1",
                    problem="hh",
                    L=3,
                    n_ph_max=1,
                    t=1.0,
                    u=2.0,
                    dv=0.0,
                    omega0=1.0,
                    g_ep=0.5,
                )
            )
        else:
            raise ValueError(f"Unsupported scenario key '{key}'.")
    return out


def _scenario_default_steps(scenario_id: str) -> tuple[int, ...]:
    if scenario_id == "hubbard_L4":
        return (256, 384, 512, 768)
    if scenario_id == "hh_L2_nb1":
        return (64, 96, 128, 192)
    if scenario_id == "hh_L2_nb2":
        return (128, 192, 256, 384)
    if scenario_id == "hh_L2_nb3":
        return (192, 256, 384, 512)
    if scenario_id == "hh_L3_nb1":
        return (192, 256, 320, 384)
    raise ValueError(f"Unknown scenario_id '{scenario_id}'.")


def _scenario_dimension_proxy(s: BenchmarkScenario) -> int:
    if s.problem == "hubbard":
        return int(2 ** (2 * int(s.L)))
    assert s.n_ph_max is not None
    return int((2 ** (2 * int(s.L))) * ((int(s.n_ph_max) + 1) ** int(s.L)))


def _expand_drive_cases(cfg: EfficiencyConfig) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for drive_kind in cfg.drive_grid:
        if drive_kind == "sinusoid":
            for omega in cfg.sinusoid_omegas:
                out.append(
                    {
                        "drive_kind": "sinusoid",
                        "drive_case_id": f"sin_om{omega:.1f}_tb5.0",
                        "A": float(cfg.drive_A),
                        "omega": float(omega),
                        "tbar": 5.0,
                        "phi": float(cfg.drive_phi),
                        "t0": 0.0,
                    }
                )
        elif drive_kind == "gaussian_sharp":
            for tbar in cfg.gaussian_tbars:
                out.append(
                    {
                        "drive_kind": "gaussian_sharp",
                        "drive_case_id": f"gauss_om2.0_tb{tbar:.2f}",
                        "A": float(cfg.drive_A),
                        "omega": 2.0,
                        "tbar": float(tbar),
                        "phi": float(cfg.drive_phi),
                        "t0": float(cfg.t_final) / 2.0,
                    }
                )
        else:
            raise ValueError(f"Unsupported drive case '{drive_kind}'.")
    return out


def _stage_mode_to_backend(stage_mode: str) -> str:
    sm = str(stage_mode).strip().lower()
    if sm == "exact_sparse":
        return "expm_multiply_sparse"
    if sm == "exact_dense":
        return "dense_expm"
    if sm == "pauli_suzuki2":
        return "pauli_suzuki2"
    raise ValueError(f"Unsupported stage mode '{stage_mode}'.")


def _track_for_stage_mode(stage_mode: str) -> str:
    sm = str(stage_mode).strip().lower()
    if sm == "pauli_suzuki2":
        return "hardware_proxy"
    return "integrator_exact"


def _effective_track_id(stage_mode: str) -> str:
    sm = str(stage_mode).strip().lower()
    if sm == "pauli_suzuki2":
        return "hardware_proxy"
    if sm == "exact_sparse":
        return "integrator_exact_sparse"
    if sm == "exact_dense":
        return "integrator_exact_dense"
    return f"unknown_{sm}"


def _should_include_stage_mode_for_scenario(
    stage_mode: str,
    scenario: BenchmarkScenario,
    min_dim: int,
) -> bool:
    if str(stage_mode).strip().lower() != "exact_dense":
        return True
    return _scenario_dimension_proxy(scenario) == min_dim


def _build_pipeline_cmd(
    *,
    scenario: BenchmarkScenario,
    method: str,
    trotter_steps: int,
    output_json: Path,
    cfqm_stage_exp: str | None,
    cfg: EfficiencyConfig,
    drive_case: Mapping[str, Any],
    exact_steps_multiplier: int,
) -> list[str]:
    adapt_pool = str(cfg.adapt_pool).strip()
    if adapt_pool == "auto":
        adapt_pool = "paop_std" if scenario.problem == "hh" else "uccsd"

    cmd = [
        sys.executable,
        "pipelines/hardcoded/hubbard_pipeline.py",
        "--problem",
        str(scenario.problem),
        "--L",
        str(int(scenario.L)),
        "--t",
        str(float(scenario.t)),
        "--u",
        str(float(scenario.u)),
        "--dv",
        str(float(scenario.dv)),
        "--boundary",
        str(cfg.boundary),
        "--ordering",
        str(cfg.ordering),
        "--t-final",
        str(float(cfg.t_final)),
        "--num-times",
        str(int(cfg.num_times)),
        "--suzuki-order",
        "2",
        "--trotter-steps",
        str(int(trotter_steps)),
        "--propagator",
        str(method),
        "--initial-state-source",
        str(cfg.initial_state_source),
        "--vqe-ansatz",
        str(cfg.vqe_ansatz),
        "--vqe-reps",
        str(int(cfg.vqe_reps)),
        "--vqe-restarts",
        str(int(cfg.vqe_restarts)),
        "--vqe-maxiter",
        str(int(cfg.vqe_maxiter)),
        "--vqe-method",
        str(cfg.vqe_method),
        "--adapt-pool",
        str(adapt_pool),
        "--adapt-max-depth",
        str(int(cfg.adapt_max_depth)),
        "--adapt-maxiter",
        str(int(cfg.adapt_maxiter)),
        "--enable-drive",
        "--drive-A",
        str(float(drive_case["A"])),
        "--drive-omega",
        str(float(drive_case["omega"])),
        "--drive-tbar",
        str(float(drive_case["tbar"])),
        "--drive-phi",
        str(float(drive_case["phi"])),
        "--drive-t0",
        str(float(drive_case["t0"])),
        "--drive-pattern",
        "staggered",
        "--drive-time-sampling",
        str(cfg.drive_time_sampling),
        "--exact-steps-multiplier",
        str(int(exact_steps_multiplier)),
        "--skip-qpe",
        "--skip-pdf",
        "--output-json",
        str(output_json),
    ]

    if bool(cfg.drive_include_identity):
        cmd.append("--drive-include-identity")

    if scenario.problem == "hh":
        if scenario.omega0 is None or scenario.g_ep is None or scenario.n_ph_max is None:
            raise ValueError("HH scenario is missing omega0/g_ep/n_ph_max.")
        cmd.extend(
            [
                "--omega0",
                str(float(scenario.omega0)),
                "--g-ep",
                str(float(scenario.g_ep)),
                "--n-ph-max",
                str(int(scenario.n_ph_max)),
            ]
        )

    if str(cfg.initial_state_source).strip().lower() == "adapt_json":
        if cfg.adapt_input_json is None:
            raise ValueError("initial_state_source=adapt_json requires adapt_input_json path.")
        cmd.extend(
            [
                "--adapt-input-json",
                str(cfg.adapt_input_json),
            ]
        )
        if not bool(cfg.adapt_strict_match):
            cmd.append("--no-adapt-strict-match")

    if cfqm_stage_exp is not None:
        cmd.extend(
            [
                "--cfqm-stage-exp",
                str(cfqm_stage_exp),
                "--cfqm-coeff-drop-abs-tol",
                "0.0",
            ]
        )

    return cmd


def _run_pipeline_subprocess(cmd: list[str], *, cwd: Path) -> dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = float(time.perf_counter() - started)
    if proc.returncode != 0:
        tail_out = (proc.stdout or "")[-4000:]
        tail_err = (proc.stderr or "")[-4000:]
        raise RuntimeError(
            "Pipeline run failed. "
            f"cmd={' '.join(cmd)}\n"
            f"returncode={proc.returncode}\n"
            f"stdout_tail=\n{tail_out}\n"
            f"stderr_tail=\n{tail_err}\n"
        )

    out_json: Path | None = None
    for i, tok in enumerate(cmd):
        if tok == "--output-json" and (i + 1) < len(cmd):
            out_json = Path(cmd[i + 1])
            break
    if out_json is None:
        raise RuntimeError("Internal error: --output-json missing in command.")
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    payload["_run_runtime_s"] = elapsed
    payload["_run_cmd"] = cmd
    return payload


def _extract_time_series(payload: Mapping[str, Any], key: str) -> tuple[list[float], list[float]]:
    traj = payload.get("trajectory")
    if not isinstance(traj, list) or not traj:
        return ([], [])
    times: list[float] = []
    vals: list[float] = []
    for row in traj:
        if not isinstance(row, Mapping):
            continue
        t_raw = row.get("time")
        v_raw = row.get(key)
        if t_raw is None or v_raw is None:
            continue
        times.append(float(t_raw))
        vals.append(float(v_raw))
    return times, vals


def _align_reference_values(
    *,
    times: Sequence[float],
    ref_times: Sequence[float],
    ref_vals: Sequence[float],
) -> list[float]:
    if not times or not ref_times or not ref_vals:
        return []
    if len(times) == len(ref_times) and all(abs(float(a) - float(b)) <= 1e-12 for a, b in zip(times, ref_times)):
        return [float(v) for v in ref_vals]
    out: list[float] = []
    for t in times:
        j = min(range(len(ref_times)), key=lambda idx: abs(float(ref_times[idx]) - float(t)))
        out.append(float(ref_vals[j]))
    return out


def _compute_error_metrics(
    *,
    method_payload: Mapping[str, Any],
    reference_payload: Mapping[str, Any],
) -> dict[str, float]:
    traj = method_payload.get("trajectory")
    ref_traj = reference_payload.get("trajectory")
    if not isinstance(traj, list) or not traj:
        raise ValueError("Method payload has empty trajectory.")
    if not isinstance(ref_traj, list) or not ref_traj:
        raise ValueError("Reference payload has empty trajectory.")

    final_row = traj[-1]
    fidelity_final = float(final_row.get("fidelity", float("nan")))
    if math.isnan(fidelity_final):
        final_infidelity = float("nan")
    else:
        final_infidelity = float(max(0.0, 1.0 - fidelity_final))

    t_e, method_energy = _extract_time_series(method_payload, "energy_total_trotter")
    t_re, ref_energy = _extract_time_series(reference_payload, "energy_total_trotter")
    ref_energy_aligned = _align_reference_values(times=t_e, ref_times=t_re, ref_vals=ref_energy)
    max_energy_abs_err = float(
        max((abs(a - b) for a, b in zip(method_energy, ref_energy_aligned)), default=float("nan"))
    )

    t_d, method_d = _extract_time_series(method_payload, "doublon_trotter")
    t_rd, ref_d = _extract_time_series(reference_payload, "doublon_trotter")
    ref_d_aligned = _align_reference_values(times=t_d, ref_times=t_rd, ref_vals=ref_d)
    max_doublon_abs_err = float(
        max((abs(a - b) for a, b in zip(method_d, ref_d_aligned)), default=float("nan"))
    )

    return {
        "final_infidelity": final_infidelity,
        "max_doublon_abs_err": max_doublon_abs_err,
        "max_energy_abs_err": max_energy_abs_err,
    }


def _compute_expm_multiply_calls(
    *,
    method: str,
    stage_mode: str,
    n_steps: int,
    n_ref_steps: int,
) -> tuple[int, int, int]:
    m = str(method).strip().lower()
    sm = str(stage_mode).strip().lower()

    method_calls = 0
    if m == "piecewise_exact":
        method_calls = int(n_steps)
    elif m in {"cfqm4", "cfqm6"}:
        if sm in {"exact_sparse", "exact_dense"}:
            n_stages = 2 if m == "cfqm4" else 5
            method_calls = int(n_steps) * int(n_stages)
        else:
            method_calls = 0
    elif m == "suzuki2":
        method_calls = 0
    else:
        raise ValueError(f"Unsupported method '{method}'.")

    ref_calls = int(n_ref_steps)
    return method_calls, ref_calls, int(method_calls + ref_calls)


def _compute_cost_metrics(
    *,
    method: str,
    stage_mode: str,
    n_steps: int,
    n_ref_steps: int,
    payload: Mapping[str, Any],
    active_coeff_tol: float,
) -> dict[str, Any]:
    method_calls, ref_calls, total_calls = _compute_expm_multiply_calls(
        method=method,
        stage_mode=stage_mode,
        n_steps=n_steps,
        n_ref_steps=n_ref_steps,
    )

    ordered_labels, static_map = _extract_ordered_static_maps(payload)
    settings = payload.get("settings", {})
    if not isinstance(settings, Mapping):
        settings = {}

    drive_provider = _build_drive_provider_from_settings(
        settings,
        int(settings.get("L", 0)),
        len(ordered_labels[0]),
    )

    drive_cfg = settings.get("drive") if isinstance(settings.get("drive"), Mapping) else {}
    t0 = float(drive_cfg.get("t0", 0.0)) if isinstance(drive_cfg, Mapping) else 0.0
    sampling = (
        str(drive_cfg.get("time_sampling", "midpoint"))
        if isinstance(drive_cfg, Mapping)
        else "midpoint"
    )

    t_final = float(settings.get("t_final", 0.0))

    m = str(method).strip().lower()
    if m == "suzuki2":
        proxy = _compute_suzuki_proxy_cost(
            T=t_final,
            n_steps=int(n_steps),
            t0=float(t0),
            sampling=str(sampling),
            static_coeff_map=static_map,
            drive_provider=drive_provider,
            ordered_labels=ordered_labels,
            active_coeff_tol=float(active_coeff_tol),
        )
    else:
        proxy = _compute_cfqm_proxy_cost(
            method=m,
            T=t_final,
            n_steps=int(n_steps),
            t0=float(t0),
            static_coeff_map=static_map,
            drive_provider=drive_provider,
            ordered_labels=ordered_labels,
            active_coeff_tol=float(active_coeff_tol),
            coeff_drop_abs_tol=0.0,
        )

    pauli_rot_count_total = int(proxy.term_exp_count)
    cx_proxy_total = int(proxy.cx_proxy)
    sq_proxy_total = int(proxy.sq_proxy)
    depth_proxy_total = int(pauli_rot_count_total)

    return {
        "expm_multiply_calls_method": int(method_calls),
        "expm_multiply_calls_reference": int(ref_calls),
        "expm_multiply_calls_total": int(total_calls),
        "pauli_rot_count_total": int(pauli_rot_count_total),
        "cx_proxy_total": int(cx_proxy_total),
        "sq_proxy_total": int(sq_proxy_total),
        "depth_proxy_total": int(depth_proxy_total),
        "ordered_labels": ordered_labels,
        "static_coeff_map": {lbl: static_map[lbl] for lbl in ordered_labels},
    }


def _cost_axis_field(axis: str) -> str:
    mapping = {
        "cx_proxy": "cx_proxy_total",
        "pauli_rot_count": "pauli_rot_count_total",
        "expm_calls": "expm_multiply_calls_total",
        "wall_time": "wall_time_s",
    }
    if axis not in mapping:
        raise ValueError(f"Unsupported equal-cost axis '{axis}'.")
    return mapping[axis]


def _group_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("scenario_id", "")),
        str(row.get("drive_case_id", "")),
        str(row.get("track", "")),
    )


def _build_exact_tie_tables(
    rows: list[dict[str, Any]],
    *,
    axis: str,
) -> list[dict[str, Any]]:
    field = _cost_axis_field(axis)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_group_key(row), []).append(row)

    out: list[dict[str, Any]] = []
    for gkey, grows in grouped.items():
        methods = sorted({str(r["method"]) for r in grows})
        by_method: dict[str, list[dict[str, Any]]] = {
            m: [r for r in grows if str(r["method"]) == m] for m in methods
        }
        targets = sorted({float(r[field]) for r in grows})
        for target in targets:
            matched: list[dict[str, Any]] = []
            all_present = True
            for method in methods:
                candidates = [r for r in by_method[method] if float(r[field]) == float(target)]
                if not candidates:
                    all_present = False
                    break
                chosen = min(candidates, key=lambda r: float(r.get("max_energy_abs_err", float("inf"))))
                matched.append(chosen)
            if not all_present:
                continue
            out.append(
                {
                    "scenario_id": gkey[0],
                    "drive_case_id": gkey[1],
                    "track": gkey[2],
                    "axis": axis,
                    "target_cost": float(target),
                    "rows": matched,
                }
            )
    return out


def _build_walltime_near_ties(
    rows: list[dict[str, Any]],
    *,
    rel_tol: float = 0.05,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_group_key(row), []).append(row)

    out: list[dict[str, Any]] = []
    for gkey, grows in grouped.items():
        methods = sorted({str(r["method"]) for r in grows})
        if not methods:
            continue
        anchors = sorted(grows, key=lambda r: float(r["wall_time_s"]))
        for anchor in anchors:
            target = float(anchor["wall_time_s"])
            matched: list[dict[str, Any]] = []
            ok = True
            for method in methods:
                method_rows = [r for r in grows if str(r["method"]) == method]
                if not method_rows:
                    ok = False
                    break
                chosen = min(method_rows, key=lambda r: abs(float(r["wall_time_s"]) - target))
                denom = max(abs(target), 1e-12)
                rel = abs(float(chosen["wall_time_s"]) - target) / denom
                if rel > float(rel_tol):
                    ok = False
                    break
                matched.append(chosen)
            if not ok:
                continue
            out.append(
                {
                    "scenario_id": gkey[0],
                    "drive_case_id": gkey[1],
                    "track": gkey[2],
                    "axis": "wall_time",
                    "target_cost": float(target),
                    "rows": matched,
                    "approximate": True,
                    "relative_tolerance": float(rel_tol),
                }
            )
    return out


def _build_fallback_appendix(
    rows: list[dict[str, Any]],
    *,
    axis: str,
) -> list[dict[str, Any]]:
    field = _cost_axis_field(axis)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_group_key(row), []).append(row)

    out: list[dict[str, Any]] = []
    for gkey, grows in grouped.items():
        methods = sorted({str(r["method"]) for r in grows})
        by_method = {m: [r for r in grows if str(r["method"]) == m] for m in methods}
        targets = sorted({float(r[field]) for r in grows})
        for target in targets:
            matches: list[dict[str, Any]] = []
            for method in methods:
                candidates = by_method[method]
                chosen = min(candidates, key=lambda r: abs(float(r[field]) - target))
                delta = abs(float(chosen[field]) - target)
                entry = dict(chosen)
                entry["fallback_delta"] = float(delta)
                matches.append(entry)
            out.append(
                {
                    "scenario_id": gkey[0],
                    "drive_case_id": gkey[1],
                    "track": gkey[2],
                    "axis": axis,
                    "target_cost": float(target),
                    "rows": matches,
                }
            )
    return out


def _fit_loglog_slope(
    *,
    dts: Sequence[float],
    errors: Sequence[float],
) -> float | None:
    pts = [
        (float(dt), float(err))
        for dt, err in zip(dts, errors)
        if float(dt) > 0.0 and float(err) > 0.0 and math.isfinite(float(err))
    ]
    if len(pts) < 2:
        return None
    pts = sorted(pts, key=lambda x: x[0])
    pts = pts[: min(3, len(pts))]
    xs = np.log(np.asarray([p[0] for p in pts], dtype=float))
    ys = np.log(np.asarray([p[1] for p in pts], dtype=float))
    coeff = np.polyfit(xs, ys, deg=1)
    return float(coeff[0])


def _build_slope_fits(rows: list[dict[str, Any]], error_metrics: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        gkey = (
            str(row["scenario_id"]),
            str(row["drive_case_id"]),
            str(row["track"]),
            str(row["method"]),
        )
        grouped.setdefault(gkey, []).append(row)

    out: list[dict[str, Any]] = []
    for gkey, grows in grouped.items():
        grows = sorted(grows, key=lambda r: float(r["dt"]))
        dts = [float(r["dt"]) for r in grows]
        for metric in error_metrics:
            errs = [float(r.get(metric, float("nan"))) for r in grows]
            slope = _fit_loglog_slope(dts=dts, errors=errs)
            out.append(
                {
                    "scenario_id": gkey[0],
                    "drive_case_id": gkey[1],
                    "track": gkey[2],
                    "method": gkey[3],
                    "error_metric": str(metric),
                    "slope": slope,
                }
            )
    return out


def _build_pareto_by_metric(
    rows: list[dict[str, Any]],
    *,
    error_metrics: Sequence[str],
    cost_axes: Sequence[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_group_key(row), []).append(row)

    for gkey, grows in grouped.items():
        for err_metric in error_metrics:
            for axis in cost_axes:
                cost_field = _cost_axis_field(axis)
                pts = [
                    r
                    for r in grows
                    if math.isfinite(float(r.get(err_metric, float("nan"))))
                    and math.isfinite(float(r.get(cost_field, float("nan"))))
                ]
                pts_sorted = sorted(
                    pts,
                    key=lambda r: (
                        float(r[cost_field]),
                        float(r[err_metric]),
                        str(r["method"]),
                        int(r["trotter_steps"]),
                    ),
                )
                frontier: list[dict[str, Any]] = []
                best_err = float("inf")
                for p in pts_sorted:
                    e = float(p[err_metric])
                    if e <= best_err + 1e-18:
                        frontier.append(
                            {
                                "method": str(p["method"]),
                                "trotter_steps": int(p["trotter_steps"]),
                                "stage_mode": str(p["stage_mode"]),
                                "cost": float(p[cost_field]),
                                "error": e,
                            }
                        )
                        best_err = min(best_err, e)
                out.append(
                    {
                        "scenario_id": gkey[0],
                        "drive_case_id": gkey[1],
                        "track": gkey[2],
                        "cost_axis": axis,
                        "error_metric": err_metric,
                        "frontier": frontier,
                    }
                )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _chunk_rows(rows: list[list[str]], chunk_size: int = 24) -> Iterable[list[list[str]]]:
    for i in range(0, len(rows), int(chunk_size)):
        yield rows[i : i + int(chunk_size)]


def _compact_cell_text(value: Any, *, width: int = 12, max_lines: int = 2) -> str:
    """Format long cell values to avoid table overflow in PDF pages."""
    txt = str(value)
    if len(txt) <= width:
        return txt
    chunks = [txt[i : i + width] for i in range(0, len(txt), width)]
    if len(chunks) > max_lines:
        kept = chunks[:max_lines]
        tail = kept[-1]
        kept[-1] = (tail[:-1] + "~") if len(tail) >= 1 else "~"
        chunks = kept
    return "\n".join(chunks)


def _display_drive_id(drive_case_id: Any) -> str:
    return _compact_cell_text(drive_case_id, width=12, max_lines=2)


def _display_track_id(track_id: Any) -> str:
    return _compact_cell_text(track_id, width=12, max_lines=2)


def _as_finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _relative_improvement_str(
    *,
    baseline: float | None,
    candidate: float | None,
) -> str:
    """Return relative improvement of candidate vs baseline as a percentage string.

    Positive means candidate is lower (better for error metrics).
    """
    if baseline is None or candidate is None:
        return "nan"
    b = abs(float(baseline))
    if b > 0.0:
        rel = (float(baseline) - float(candidate)) / b
        return f"{100.0 * rel:.3g}%"
    if abs(float(candidate)) == 0.0:
        return "0%"
    return "n/a"


def _closest_suzuki_cfqm4_pair_for_axis(
    *,
    rows: Sequence[Mapping[str, Any]],
    axis_field: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], float] | None:
    suz_rows = [
        r
        for r in rows
        if str(r.get("method", "")) == "suzuki2" and _as_finite_float(r.get(axis_field)) is not None
    ]
    c4_rows = [
        r
        for r in rows
        if str(r.get("method", "")) == "cfqm4" and _as_finite_float(r.get(axis_field)) is not None
    ]
    best: tuple[Mapping[str, Any], Mapping[str, Any], float] | None = None
    best_tie: tuple[float, int, int] | None = None
    for rs in suz_rows:
        sv = _as_finite_float(rs.get(axis_field))
        if sv is None:
            continue
        for rc in c4_rows:
            # Keep comparisons within same benchmark track and scenario.
            if str(rs.get("scenario_id", "")) != str(rc.get("scenario_id", "")):
                continue
            if str(rs.get("track", "")) != str(rc.get("track", "")):
                continue
            cv = _as_finite_float(rc.get(axis_field))
            if cv is None:
                continue
            delta = abs(float(sv) - float(cv))
            tie_key = (
                float(max(abs(sv), abs(cv))),
                int(rs.get("trotter_steps", 0)),
                int(rc.get("trotter_steps", 0)),
            )
            if best is None or delta < best[2] - 1e-18 or (abs(delta - best[2]) <= 1e-18 and tie_key < best_tie):
                best = (rs, rc, float(delta))
                best_tie = tie_key
    return best


def _format_closest_pair_line(
    *,
    label: str,
    axis_field: str,
    rows: Sequence[Mapping[str, Any]],
) -> str:
    pair = _closest_suzuki_cfqm4_pair_for_axis(rows=rows, axis_field=axis_field)
    if pair is None:
        return f"- Closest {label} (Suzuki2 vs CFQM4): unavailable in this payload."

    rs, rc, delta = pair
    sval = _as_finite_float(rs.get(axis_field))
    cval = _as_finite_float(rc.get(axis_field))
    e_s = _as_finite_float(rs.get("max_energy_abs_err"))
    e_c = _as_finite_float(rc.get("max_energy_abs_err"))
    d_s = _as_finite_float(rs.get("max_doublon_abs_err"))
    d_c = _as_finite_float(rc.get("max_doublon_abs_err"))
    s_drive = str(rs.get("drive_case_id", ""))
    c_drive = str(rc.get("drive_case_id", ""))
    s_steps = int(rs.get("trotter_steps", 0))
    c_steps = int(rc.get("trotter_steps", 0))
    metric_name = str(axis_field)
    rel_delta = float("nan")
    if sval is not None and cval is not None:
        denom = abs(float(sval))
        if denom > 0.0:
            rel_delta = abs(float(sval) - float(cval)) / denom
        elif abs(float(cval)) == 0.0:
            rel_delta = 0.0
        else:
            rel_delta = float("inf")
    rel_str = "inf" if math.isinf(rel_delta) else ("nan" if not math.isfinite(rel_delta) else f"{100.0*rel_delta:.3g}%")
    rel_improve_e = _relative_improvement_str(baseline=e_s, candidate=e_c)
    rel_improve_d = _relative_improvement_str(baseline=d_s, candidate=d_c)

    return (
        f"- Closest {label} (Suzuki2 vs CFQM4): "
        f"Suzuki2[drive={s_drive}, S={s_steps}, {metric_name}={sval:.6g}] vs "
        f"CFQM4[drive={c_drive}, S={c_steps}, {metric_name}={cval:.6g}], "
        f"|Δ|={delta:.6g}, rel|Δ|={rel_str} (vs Suzuki2); "
        f"max_E: {e_s:.3e} vs {e_c:.3e} (rel improve={rel_improve_e}); "
        f"max_doublon: {d_s:.3e} vs {d_c:.3e} (rel improve={rel_improve_d})."
    )


def _build_pdf_headline_lines(
    *,
    rows: Sequence[Mapping[str, Any]],
    drives: Sequence[str],
    tracks: Sequence[str],
) -> list[str]:
    """Create a concise, explicit conclusions block for the PDF front matter."""

    lines: list[str] = [
        "Headline Conclusions (Auto-generated from this report payload)",
        "",
        f"- Drive cases benchmarked: {', '.join(str(d) for d in drives)}",
        f"- Tracks benchmarked: {', '.join(str(t) for t in tracks)}",
    ]

    # Same-step comparisons: suzuki2 vs cfqm4 on identical scenario/drive/track/stage/S.
    suz_by_key: dict[tuple[str, str, str, str, int], Mapping[str, Any]] = {}
    c4_by_key: dict[tuple[str, str, str, str, int], Mapping[str, Any]] = {}
    for r in rows:
        key = (
            str(r.get("scenario_id", "")),
            str(r.get("drive_case_id", "")),
            str(r.get("track", "")),
            str(r.get("stage_mode", "")),
            int(r.get("trotter_steps", 0)),
        )
        m = str(r.get("method", ""))
        if m == "suzuki2":
            suz_by_key[key] = r
        elif m == "cfqm4":
            c4_by_key[key] = r

    shared_keys = sorted(set(suz_by_key).intersection(c4_by_key))
    same_step_pairs = 0
    same_step_c4_better_E = 0
    same_step_c4_better_D = 0
    for k in shared_keys:
        rs = suz_by_key[k]
        rc = c4_by_key[k]
        e_s = float(rs.get("max_energy_abs_err", float("nan")))
        e_c = float(rc.get("max_energy_abs_err", float("nan")))
        d_s = float(rs.get("max_doublon_abs_err", float("nan")))
        d_c = float(rc.get("max_doublon_abs_err", float("nan")))
        if math.isfinite(e_s) and math.isfinite(e_c) and math.isfinite(d_s) and math.isfinite(d_c):
            same_step_pairs += 1
            if e_c < e_s:
                same_step_c4_better_E += 1
            if d_c < d_s:
                same_step_c4_better_D += 1

    if same_step_pairs > 0:
        if same_step_c4_better_E == same_step_pairs and same_step_c4_better_D == same_step_pairs:
            lines.append(
                "- CFQM4 is better than Suzuki2 on all same-step comparisons in this payload "
                "(both max_energy_abs_err and max_doublon_abs_err)."
            )
        else:
            lines.append(
                "- Same-step CFQM4 vs Suzuki2: "
                f"max_energy_abs_err better in {same_step_c4_better_E}/{same_step_pairs}, "
                f"max_doublon_abs_err better in {same_step_c4_better_D}/{same_step_pairs}."
            )
    else:
        lines.append("- No same-step CFQM4 vs Suzuki2 overlap was found in this payload.")

    # Pairwise equal proxy-cost ties (cx + pauli-rotation) for suzuki2 vs cfqm4.
    suz_by_proxy: dict[tuple[str, str, str, str, int, int], Mapping[str, Any]] = {}
    c4_by_proxy: dict[tuple[str, str, str, str, int, int], Mapping[str, Any]] = {}
    for r in rows:
        key = (
            str(r.get("scenario_id", "")),
            str(r.get("drive_case_id", "")),
            str(r.get("track", "")),
            str(r.get("stage_mode", "")),
            int(r.get("cx_proxy_total", 0)),
            int(r.get("pauli_rot_count_total", 0)),
        )
        m = str(r.get("method", ""))
        if m == "suzuki2":
            suz_by_proxy[key] = r
        elif m == "cfqm4":
            c4_by_proxy[key] = r

    tie_keys = sorted(set(suz_by_proxy).intersection(c4_by_proxy))
    tie_pairs = 0
    tie_c4_better_E = 0
    tie_c4_better_D = 0
    for k in tie_keys:
        rs = suz_by_proxy[k]
        rc = c4_by_proxy[k]
        e_s = float(rs.get("max_energy_abs_err", float("nan")))
        e_c = float(rc.get("max_energy_abs_err", float("nan")))
        d_s = float(rs.get("max_doublon_abs_err", float("nan")))
        d_c = float(rc.get("max_doublon_abs_err", float("nan")))
        if math.isfinite(e_s) and math.isfinite(e_c) and math.isfinite(d_s) and math.isfinite(d_c):
            tie_pairs += 1
            if e_c < e_s:
                tie_c4_better_E += 1
            if d_c < d_s:
                tie_c4_better_D += 1

    if tie_pairs > 0:
        lines.append(
            "- Pairwise equal-cost ties (cx_proxy_total + pauli_rot_count_total) between "
            f"Suzuki2 and CFQM4: {tie_pairs}; CFQM4 better in max_energy_abs_err "
            f"{tie_c4_better_E}/{tie_pairs}, max_doublon_abs_err {tie_c4_better_D}/{tie_pairs}."
        )
    else:
        lines.append(
            "- No pairwise exact ties on (cx_proxy_total, pauli_rot_count_total) were found for Suzuki2 vs CFQM4."
        )

    lines.append(
        "- Closest-pair comparisons requested (Suzuki2 vs CFQM4) when exact ties are absent:"
    )
    lines.append(
        _format_closest_pair_line(
            label="CX proxy total",
            axis_field="cx_proxy_total",
            rows=rows,
        )
    )
    lines.append(
        _format_closest_pair_line(
            label="2Q transpiled total",
            axis_field="transpiled_2q_count",
            rows=rows,
        )
    )
    lines.append(
        _format_closest_pair_line(
            label="PROT total (pauli_rot_count_total)",
            axis_field="pauli_rot_count_total",
            rows=rows,
        )
    )
    lines.append(
        _format_closest_pair_line(
            label="XM/EXPM total (expm_multiply_calls_total)",
            axis_field="expm_multiply_calls_total",
            rows=rows,
        )
    )

    # Explicit wall-time caveat for recovered reports.
    wall_vals = [float(r.get("wall_time_s", float("nan"))) for r in rows]
    finite_wall = [w for w in wall_vals if math.isfinite(w)]
    if not finite_wall:
        lines.append(
            "- Wall-time comparisons are unavailable in this report (wall_time_s is NaN in payload)."
        )
    else:
        lines.append(
            f"- Wall-time values present for {len(finite_wall)}/{len(wall_vals)} runs; interpret with machine-load caveats."
        )

    lines.append(
        "- Scope note: conclusions above apply only to the listed drive cases/tracks in this report."
    )
    return lines


def _write_efficiency_pdf(
    *,
    output_pdf: Path,
    payload: Mapping[str, Any],
) -> None:
    require_matplotlib()
    plt = get_plt()
    PdfPages = get_PdfPages()

    rows = payload.get("runs", [])
    if not isinstance(rows, list):
        return
    rows = [r for r in rows if isinstance(r, Mapping)]
    if not rows:
        return
    unique_drive_ids = sorted({str(r.get("drive_case_id", "")) for r in rows})
    unique_track_ids = sorted({str(r.get("track", "")) for r in rows})

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(output_pdf)) as pdf:
        render_command_page(
            pdf,
            command=current_command_string(),
            script_name="pipelines/exact_bench/cfqm_vs_suzuki_efficiency_suite.py",
            extra_header_lines=[
                "Benchmark: CFQM vs Suzuki cost-efficiency",
                "Main fairness policy: exact-cost ties only",
            ],
        )

        settings = payload.get("settings", {})
        manifest = [
            "Parameter Manifest",
            "",
            f"- Model family/name: Hubbard / Hubbard-Holstein",
            f"- Ansatz type(s) used: initial_state_source={settings.get('initial_state_source')}",
            f"- Drive enabled: true",
            f"- Core physical parameters: t={settings.get('t_default')} U={settings.get('u_default')} dv={settings.get('dv_default')}",
            f"- t_final={settings.get('t_final')} num_times={settings.get('num_times')}",
            f"- methods={','.join(str(m) for m in settings.get('methods', []))}",
            f"- stage_mode_grid={','.join(str(m) for m in settings.get('stage_mode_grid', []))}",
            f"- equal_cost_axis={','.join(str(m) for m in settings.get('equal_cost_axis', []))}",
            f"- Drive cases in this report: {', '.join(unique_drive_ids)}",
            f"- Tracks in this report: {', '.join(unique_track_ids)}",
            "- Table formatting note: scenario/drive/track cells are wrapped for readability.",
        ]
        render_text_page(pdf, manifest, fontsize=9, line_spacing=0.035)
        render_text_page(
            pdf,
            _build_pdf_headline_lines(
                rows=rows,
                drives=unique_drive_ids,
                tracks=unique_track_ids,
            ),
            fontsize=9,
            line_spacing=0.035,
            max_line_width=110,
        )

        headers = [
            "scenario",
            "drive",
            "track",
            "method",
            "stage",
            "S",
            "dt",
            "infid",
            "max_d",
            "max_E",
            "cx",
            "prot",
            "expm",
            "wall_s",
        ]
        table_rows: list[list[str]] = []
        for r in rows:
            table_rows.append(
                [
                    _compact_cell_text(r.get("scenario_id", ""), width=11, max_lines=2),
                    _display_drive_id(r.get("drive_case_id", "")),
                    _display_track_id(r.get("track", "")),
                    str(r.get("method", "")),
                    str(r.get("stage_mode", "")),
                    f"{int(r.get('trotter_steps', 0))}",
                    f"{float(r.get('dt', float('nan'))):.3e}",
                    f"{float(r.get('final_infidelity', float('nan'))):.3e}",
                    f"{float(r.get('max_doublon_abs_err', float('nan'))):.3e}",
                    f"{float(r.get('max_energy_abs_err', float('nan'))):.3e}",
                    f"{int(r.get('cx_proxy_total', 0))}",
                    f"{int(r.get('pauli_rot_count_total', 0))}",
                    f"{int(r.get('expm_multiply_calls_total', 0))}",
                    f"{float(r.get('wall_time_s', float('nan'))):.2f}",
                ]
            )

        for chunk in _chunk_rows(table_rows, chunk_size=22):
            fig = plt.figure(figsize=(13, 8))
            ax = fig.add_subplot(111)
            render_compact_table(
                ax,
                title="Run table (error + cost metrics)",
                col_labels=headers,
                rows=chunk,
                fontsize=7,
            )
            pdf.savefig(fig)
            plt.close(fig)

        slope_rows = payload.get("slope_fits", [])
        if isinstance(slope_rows, list) and slope_rows:
            rows_fmt = [
                [
                    _compact_cell_text(r.get("scenario_id", ""), width=12, max_lines=2),
                    _display_drive_id(r.get("drive_case_id", "")),
                    _display_track_id(r.get("track", "")),
                    str(r.get("method", "")),
                    str(r.get("error_metric", "")),
                    "nan" if r.get("slope") is None else f"{float(r['slope']):.3f}",
                ]
                for r in slope_rows
                if isinstance(r, Mapping)
            ]
            for chunk in _chunk_rows(rows_fmt, chunk_size=26):
                fig = plt.figure(figsize=(12, 7))
                ax = fig.add_subplot(111)
                render_compact_table(
                    ax,
                    title="Convergence slopes: log(error) vs log(dt)",
                    col_labels=["scenario", "drive", "track", "method", "metric", "slope"],
                    rows=chunk,
                    fontsize=7,
                )
                pdf.savefig(fig)
                plt.close(fig)

        tie_summary = payload.get("equal_cost_exact_ties", {})
        if isinstance(tie_summary, Mapping):
            for axis, axis_rows in tie_summary.items():
                if axis == "wall_time" or not isinstance(axis_rows, list) or not axis_rows:
                    continue
                display_rows: list[list[str]] = []
                for block in axis_rows:
                    if not isinstance(block, Mapping):
                        continue
                    target = float(block.get("target_cost", float("nan")))
                    for rr in block.get("rows", []) if isinstance(block.get("rows"), list) else []:
                        if not isinstance(rr, Mapping):
                            continue
                        display_rows.append(
                            [
                                _compact_cell_text(block.get("scenario_id", ""), width=12, max_lines=2),
                                _display_drive_id(block.get("drive_case_id", "")),
                                _display_track_id(block.get("track", "")),
                                str(rr.get("method", "")),
                                str(rr.get("stage_mode", "")),
                                f"{int(rr.get('trotter_steps', 0))}",
                                f"{target:.3e}",
                                f"{float(rr.get('max_energy_abs_err', float('nan'))):.3e}",
                                f"{float(rr.get('max_doublon_abs_err', float('nan'))):.3e}",
                                f"{float(rr.get('final_infidelity', float('nan'))):.3e}",
                            ]
                        )
                if not display_rows:
                    continue
                for chunk in _chunk_rows(display_rows, chunk_size=24):
                    fig = plt.figure(figsize=(13, 8))
                    ax = fig.add_subplot(111)
                    render_compact_table(
                        ax,
                        title=f"Equal-cost exact ties ({axis})",
                        col_labels=[
                            "scenario",
                            "drive",
                            "track",
                            "method",
                            "stage",
                            "S",
                            "cost",
                            "max_E",
                            "max_d",
                            "infid",
                        ],
                        rows=chunk,
                        fontsize=7,
                    )
                    pdf.savefig(fig)
                    plt.close(fig)

        wall_bins = tie_summary.get("wall_time") if isinstance(tie_summary, Mapping) else None
        if isinstance(wall_bins, list) and wall_bins:
            fmt_rows: list[list[str]] = []
            for block in wall_bins:
                if not isinstance(block, Mapping):
                    continue
                target = float(block.get("target_cost", float("nan")))
                for rr in block.get("rows", []) if isinstance(block.get("rows"), list) else []:
                    if not isinstance(rr, Mapping):
                        continue
                    fmt_rows.append(
                        [
                            _compact_cell_text(block.get("scenario_id", ""), width=12, max_lines=2),
                            _display_drive_id(block.get("drive_case_id", "")),
                            _display_track_id(block.get("track", "")),
                            str(rr.get("method", "")),
                            f"{int(rr.get('trotter_steps', 0))}",
                            f"{target:.3e}",
                            f"{float(rr.get('wall_time_s', float('nan'))):.3e}",
                            f"{float(rr.get('max_energy_abs_err', float('nan'))):.3e}",
                        ]
                    )
            for chunk in _chunk_rows(fmt_rows, chunk_size=24):
                fig = plt.figure(figsize=(12, 7))
                ax = fig.add_subplot(111)
                render_compact_table(
                    ax,
                    title="Approximate wall-time bins (explicitly approximate)",
                    col_labels=["scenario", "drive", "track", "method", "S", "target_wall", "wall_s", "max_E"],
                    rows=chunk,
                    fontsize=7,
                )
                pdf.savefig(fig)
                plt.close(fig)

        # Convergence and Pareto plots.
        rows_by_group: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
        for r in rows:
            key = (str(r.get("scenario_id")), str(r.get("drive_case_id")), str(r.get("track")))
            rows_by_group.setdefault(key, []).append(r)

        for gkey, grows in rows_by_group.items():
            fig = plt.figure(figsize=(12, 7))
            ax = fig.add_subplot(111)
            by_method: dict[str, list[Mapping[str, Any]]] = {}
            for r in grows:
                by_method.setdefault(str(r.get("method", "")), []).append(r)
            for method, mrows in sorted(by_method.items()):
                mrows = sorted(mrows, key=lambda x: float(x.get("dt", float("inf"))))
                xs = [float(r.get("dt", float("nan"))) for r in mrows]
                ys = [float(r.get("max_energy_abs_err", float("nan"))) for r in mrows]
                if xs and ys:
                    ax.loglog(xs, ys, marker="o", linewidth=1.2, label=method)
            ax.set_title(f"Convergence: max_energy_abs_err vs dt ({gkey[0]} | {gkey[1]} | {gkey[2]})")
            ax.set_xlabel("dt")
            ax.set_ylabel("max_energy_abs_err")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(loc="best", fontsize=8)
            pdf.savefig(fig)
            plt.close(fig)

            for axis in ("cx_proxy_total", "pauli_rot_count_total", "expm_multiply_calls_total", "wall_time_s"):
                fig = plt.figure(figsize=(12, 7))
                ax = fig.add_subplot(111)
                for method, mrows in sorted(by_method.items()):
                    xs = [float(r.get(axis, float("nan"))) for r in mrows]
                    ys = [float(r.get("max_energy_abs_err", float("nan"))) for r in mrows]
                    if xs and ys:
                        ax.loglog(xs, ys, marker="o", linewidth=1.2, label=method)
                ax.set_title(f"Pareto: max_energy_abs_err vs {axis} ({gkey[0]} | {gkey[1]} | {gkey[2]})")
                ax.set_xlabel(axis)
                ax.set_ylabel("max_energy_abs_err")
                ax.grid(True, which="both", alpha=0.3)
                ax.legend(loc="best", fontsize=8)
                pdf.savefig(fig)
                plt.close(fig)

        # Calibration summary page if present.
        calib = payload.get("transpile_calibration", {})
        if isinstance(calib, Mapping):
            summ = calib.get("summary", {})
            if isinstance(summ, Mapping):
                coverage_lines = [
                    "Transpile Calibration Coverage",
                    "",
                    f"- enabled: {bool(calib.get('enabled', False))}",
                    f"- status: {str(calib.get('status', 'unknown'))}",
                    f"- backend_request: {str(calib.get('backend_request', 'n/a'))}",
                    f"- backend_names_used: {','.join(str(x) for x in calib.get('backend_names_used', []))}",
                    f"- targets_total: {int(summ.get('targets_total', 0))}",
                    f"- ok: {int(summ.get('ok', 0))}",
                    f"- skipped: {int(summ.get('skipped', 0))}",
                    f"- failed: {int(summ.get('failed', 0))}",
                ]
                reason = calib.get("reason")
                if reason is not None:
                    coverage_lines.append(f"- reason: {str(reason)}")
                render_text_page(pdf, coverage_lines, fontsize=9, line_spacing=0.035)

        samples = calib.get("samples") if isinstance(calib, Mapping) else None
        if isinstance(samples, list) and samples:
            fmt_rows: list[list[str]] = []
            for s in samples:
                if not isinstance(s, Mapping):
                    continue
                fmt_rows.append(
                    [
                        str(s.get("run_id", "")),
                        str(s.get("method", "")),
                        str(s.get("stage_mode", "")),
                        str(s.get("scenario_id", "")),
                        f"{int(s.get('trotter_steps', 0))}",
                        str(s.get("status", "")),
                        str(s.get("backend", "")),
                        f"{int(s.get('proxy_cx_total', 0))}",
                        (
                            "nan"
                            if s.get("transpiled_2q_count") is None
                            else f"{int(s.get('transpiled_2q_count', 0))}"
                        ),
                        (
                            "nan"
                            if s.get("transpiled_depth") is None
                            else f"{int(s.get('transpiled_depth', 0))}"
                        ),
                    ]
                )
            for chunk in _chunk_rows(fmt_rows, chunk_size=24):
                fig = plt.figure(figsize=(13, 8))
                ax = fig.add_subplot(111)
                render_compact_table(
                    ax,
                    title="Calibration (proxy vs transpiled 2q/depth)",
                    col_labels=[
                        "run_id",
                        "method",
                        "stage",
                        "scenario",
                        "S",
                        "status",
                        "backend",
                        "proxy_cx",
                        "trans_2q",
                        "trans_depth",
                    ],
                    rows=chunk,
                    fontsize=7,
                )
                pdf.savefig(fig)
                plt.close(fig)



def _backend_num_qubits(backend: Any) -> int | None:
    nq = getattr(backend, "num_qubits", None)
    if nq is None:
        cfg_fn = getattr(backend, "configuration", None)
        if callable(cfg_fn):
            cfg = cfg_fn()
            nq = getattr(cfg, "num_qubits", None)
    if nq is None:
        return None
    try:
        return int(nq)
    except Exception:
        return None


def _list_fake_backend_specs(fake_provider: Any) -> list[tuple[str, type, int]]:
    specs: list[tuple[str, type, int]] = []
    for name in dir(fake_provider):
        if not (name.startswith("Fake") and name.endswith("V2")):
            continue
        cls = getattr(fake_provider, name, None)
        if not isinstance(cls, type):
            continue
        try:
            backend = cls()
        except Exception:
            continue
        nq = _backend_num_qubits(backend)
        if nq is None:
            continue
        specs.append((str(name), cls, int(nq)))
    return sorted(specs, key=lambda x: (x[2], x[0]))


def _select_fake_backend_spec(
    *,
    specs: Sequence[tuple[str, type, int]],
    required_nq: int,
    requested: str,
) -> tuple[tuple[str, type, int] | None, str | None]:
    if not specs:
        return None, "no_fake_backends_available"

    req = str(requested).strip()
    if req != "auto":
        for spec in specs:
            if spec[0] == req:
                if int(spec[2]) < int(required_nq):
                    return None, f"requested_backend_too_small:{req}:{spec[2]}<{required_nq}"
                return spec, None
        return None, f"requested_backend_not_found:{req}"

    candidates = [spec for spec in specs if int(spec[2]) >= int(required_nq)]
    if not candidates:
        return None, f"no_backend_wide_enough:{required_nq}"
    return candidates[0], None


def _calibration_summary_from_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    targets_total: int,
) -> dict[str, int]:
    ok = 0
    skipped = 0
    failed = 0
    for s in samples:
        status = str(s.get("status", ""))
        if status == "ok":
            ok += 1
        elif status == "skipped":
            skipped += 1
        elif status == "failed":
            failed += 1
    return {
        "targets_total": int(targets_total),
        "ok": int(ok),
        "skipped": int(skipped),
        "failed": int(failed),
    }


def _maybe_calibrate_transpile(
    *,
    rows_internal: list[dict[str, Any]],
    active_coeff_tol: float,
    t_final: float,
    enabled: bool,
    backend_request: str,
    strict: bool,
) -> dict[str, Any]:
    targets_total = int(len(rows_internal))
    if not bool(enabled):
        return {
            "enabled": False,
            "status": "skipped",
            "reason": "--calibrate-transpile disabled",
            "backend_request": str(backend_request),
            "backend_names_used": [],
            "summary": _calibration_summary_from_samples([], targets_total=targets_total),
            "samples": [],
        }

    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.synthesis import SuzukiTrotter
        from qiskit_ibm_runtime import fake_provider
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"Calibration transpile skipped due to missing stack: {exc}", RuntimeWarning)
        return {
            "enabled": True,
            "status": "skipped",
            "reason": f"import_failed: {exc}",
            "backend_request": str(backend_request),
            "backend_names_used": [],
            "summary": _calibration_summary_from_samples([], targets_total=targets_total),
            "samples": [],
        }

    specs = _list_fake_backend_specs(fake_provider)
    if not specs:
        return {
            "enabled": True,
            "status": "skipped",
            "reason": "no_fake_backends_available",
            "backend_request": str(backend_request),
            "backend_names_used": [],
            "summary": _calibration_summary_from_samples([], targets_total=targets_total),
            "samples": [],
        }

    samples: list[dict[str, Any]] = []
    backend_names_used: set[str] = set()
    for row in rows_internal:
        sample: dict[str, Any] = {
            "run_id": str(row.get("run_id", "")),
            "scenario_id": str(row.get("scenario_id", "")),
            "method": str(row.get("method", "")),
            "stage_mode": str(row.get("stage_mode", "")),
            "trotter_steps": int(row.get("trotter_steps", 0)),
            "proxy_cx_total": int(row.get("cx_proxy_total", 0)),
            "status": "skipped",
            "reason": "",
            "backend": "",
            "backend_num_qubits": None,
            "circuit_num_qubits": None,
            "transpiled_depth": None,
            "transpiled_2q_count": None,
        }
        try:
            ordered = row.get("ordered_labels", [])
            static_map = row.get("static_coeff_map", {})
            if not isinstance(ordered, list) or not ordered:
                sample["status"] = "skipped"
                sample["reason"] = "missing_or_empty_ordered_labels"
                samples.append(sample)
                continue
            if not isinstance(static_map, Mapping):
                sample["status"] = "skipped"
                sample["reason"] = "invalid_static_coeff_map"
                samples.append(sample)
                continue

            terms: list[tuple[str, complex]] = []
            for lbl in ordered:
                coeff = complex(static_map.get(lbl, 0.0 + 0.0j))
                if abs(coeff) <= float(active_coeff_tol):
                    continue
                terms.append((str(lbl).replace("e", "I").upper(), coeff))
            if not terms:
                sample["status"] = "skipped"
                sample["reason"] = "no_active_terms_after_tolerance"
                samples.append(sample)
                continue

            qop = SparsePauliOp.from_list(terms)
            nq = int(qop.num_qubits)
            sample["circuit_num_qubits"] = nq

            spec, select_reason = _select_fake_backend_spec(
                specs=specs,
                required_nq=nq,
                requested=str(backend_request),
            )
            if spec is None:
                sample["status"] = "skipped"
                sample["reason"] = str(select_reason or "backend_selection_failed")
                samples.append(sample)
                continue

            backend_name, backend_cls, backend_nq = spec
            backend = backend_cls()
            sample["backend"] = str(backend_name)
            sample["backend_num_qubits"] = int(backend_nq)

            qc = QuantumCircuit(nq)
            qc.append(
                PauliEvolutionGate(
                    qop,
                    time=float(t_final),
                    synthesis=SuzukiTrotter(order=2, reps=1, preserve_order=True),
                ),
                list(range(nq)),
            )
            tqc = transpile(qc, backend=backend, optimization_level=1)
            counts = tqc.count_ops()
            sample["status"] = "ok"
            sample["reason"] = ""
            sample["transpiled_depth"] = int(tqc.depth())
            sample["transpiled_2q_count"] = int(
                sum(int(v) for k, v in counts.items() if str(k) in {"cx", "ecr", "cz"})
            )
            backend_names_used.add(str(backend_name))
            samples.append(sample)
        except Exception as exc:
            sample["status"] = "failed"
            sample["reason"] = f"{type(exc).__name__}: {exc}"
            samples.append(sample)
            if bool(strict):
                raise RuntimeError(
                    f"Calibration failed for run_id={sample['run_id']}: {sample['reason']}"
                ) from exc

    summary = _calibration_summary_from_samples(samples, targets_total=targets_total)
    failed = int(summary.get("failed", 0))
    ok = int(summary.get("ok", 0))
    skipped = int(summary.get("skipped", 0))
    if failed > 0:
        status = "partial_with_failures"
    elif ok > 0 and skipped > 0:
        status = "partial"
    elif ok > 0:
        status = "ok"
    else:
        status = "skipped"

    return {
        "enabled": True,
        "status": status,
        "backend_request": str(backend_request),
        "backend_names_used": sorted(backend_names_used),
        "summary": summary,
        "samples": samples,
    }


def run_efficiency_suite(
    config: EfficiencyConfig,
    *,
    run_pipeline: Callable[[list[str], Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    run_fn = run_pipeline
    if run_fn is None:
        run_fn = lambda cmd, cwd: _run_pipeline_subprocess(cmd, cwd=cwd)

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = _expand_scenarios(config.problem_grid)
    min_dim = min(_scenario_dimension_proxy(s) for s in scenarios)
    drive_cases = _expand_drive_cases(config)

    records: list[BenchmarkRunRecord] = []
    rows_internal_for_calib: list[dict[str, Any]] = []

    active_coeff_tol = 1e-14

    # Cache references by scenario/drive/steps to avoid duplicate piecewise runs.
    ref_cache: dict[tuple[str, str, int], Mapping[str, Any]] = {}

    run_index = 0
    for scenario in scenarios:
        steps_grid = (
            tuple(config.steps_grid_override)
            if config.steps_grid_override is not None
            else _scenario_default_steps(scenario.scenario_id)
        )
        stage_modes_for_scenario = [
            sm
            for sm in config.stage_mode_grid
            if _should_include_stage_mode_for_scenario(sm, scenario, min_dim)
        ]
        if not stage_modes_for_scenario:
            continue
        suzuki_baseline_mode = str(stage_modes_for_scenario[0])

        for drive_case in drive_cases:
            for stage_mode in stage_modes_for_scenario:
                track_id = _effective_track_id(stage_mode)
                cfqm_backend = _stage_mode_to_backend(stage_mode)

                for method in config.methods:
                    for steps in steps_grid:
                        if method == "suzuki2" and stage_mode != suzuki_baseline_mode:
                            # Keep one suzuki baseline run per scenario/drive/steps.
                            continue

                        run_index += 1
                        run_id = (
                            f"run_{run_index:06d}_"
                            f"{scenario.scenario_id}_{drive_case['drive_case_id']}_"
                            f"{track_id}_{method}_S{int(steps)}"
                        )

                        n_ref_steps = int(steps) * int(config.reference_steps_multiplier)
                        ref_key = (scenario.scenario_id, str(drive_case["drive_case_id"]), int(steps))
                        if ref_key not in ref_cache:
                            ref_json = out_dir / f"_ref_{scenario.scenario_id}_{drive_case['drive_case_id']}_S{int(steps)}.json"
                            ref_cmd = _build_pipeline_cmd(
                                scenario=scenario,
                                method="piecewise_exact",
                                trotter_steps=int(n_ref_steps),
                                output_json=ref_json,
                                cfqm_stage_exp=None,
                                cfg=config,
                                drive_case=drive_case,
                                exact_steps_multiplier=1,
                            )
                            ref_cache[ref_key] = run_fn(ref_cmd, REPO_ROOT)
                        ref_payload = ref_cache[ref_key]

                        run_json = out_dir / f"_{run_id}.json"
                        cfqm_stage_exp = cfqm_backend if method in {"cfqm4", "cfqm6"} else None
                        cmd = _build_pipeline_cmd(
                            scenario=scenario,
                            method=str(method),
                            trotter_steps=int(steps),
                            output_json=run_json,
                            cfqm_stage_exp=cfqm_stage_exp,
                            cfg=config,
                            drive_case=drive_case,
                            exact_steps_multiplier=int(config.reference_steps_multiplier),
                        )
                        payload = run_fn(cmd, REPO_ROOT)

                        err = _compute_error_metrics(method_payload=payload, reference_payload=ref_payload)
                        cost = _compute_cost_metrics(
                            method=str(method),
                            stage_mode=str(stage_mode),
                            n_steps=int(steps),
                            n_ref_steps=int(n_ref_steps),
                            payload=payload,
                            active_coeff_tol=float(active_coeff_tol),
                        )

                        rec = BenchmarkRunRecord(
                            run_id=run_id,
                            scenario_id=scenario.scenario_id,
                            problem=scenario.problem,
                            drive_case_id=str(drive_case["drive_case_id"]),
                            track=track_id,
                            stage_mode=str(stage_mode),
                            method=str(method),
                            trotter_steps=int(steps),
                            dt=float(config.t_final) / float(steps),
                            reference_steps=int(n_ref_steps),
                            final_infidelity=float(err["final_infidelity"]),
                            max_doublon_abs_err=float(err["max_doublon_abs_err"]),
                            max_energy_abs_err=float(err["max_energy_abs_err"]),
                            wall_time_s=float(payload.get("_run_runtime_s", float("nan"))),
                            expm_multiply_calls_method=int(cost["expm_multiply_calls_method"]),
                            expm_multiply_calls_reference=int(cost["expm_multiply_calls_reference"]),
                            expm_multiply_calls_total=int(cost["expm_multiply_calls_total"]),
                            pauli_rot_count_total=int(cost["pauli_rot_count_total"]),
                            cx_proxy_total=int(cost["cx_proxy_total"]),
                            depth_proxy_total=int(cost["depth_proxy_total"]),
                            sq_proxy_total=int(cost["sq_proxy_total"]),
                            transpiled_2q_count=None,
                            transpiled_depth=None,
                        )
                        records.append(rec)

                        rows_internal_for_calib.append(
                            {
                                "run_id": run_id,
                                "scenario_id": scenario.scenario_id,
                                "method": str(method),
                                "stage_mode": str(stage_mode),
                                "trotter_steps": int(steps),
                                "cx_proxy_total": int(cost["cx_proxy_total"]),
                                "ordered_labels": list(cost["ordered_labels"]),
                                "static_coeff_map": dict(cost["static_coeff_map"]),
                            }
                        )

                        # Clone suzuki baseline into remaining tracks without rerunning.
                        if method == "suzuki2" and stage_mode == suzuki_baseline_mode:
                            for clone_mode in stage_modes_for_scenario:
                                if clone_mode == suzuki_baseline_mode:
                                    continue
                                clone_track = _effective_track_id(clone_mode)
                                clone_run_id = (
                                    f"{run_id}_clone_{clone_mode}"
                                )
                                rec_clone = BenchmarkRunRecord(
                                    **{
                                        **asdict(rec),
                                        "run_id": clone_run_id,
                                        "track": clone_track,
                                        "stage_mode": str(clone_mode),
                                    }
                                )
                                records.append(rec_clone)
                                rows_internal_for_calib.append(
                                    {
                                        "run_id": clone_run_id,
                                        "scenario_id": scenario.scenario_id,
                                        "method": "suzuki2",
                                        "stage_mode": str(clone_mode),
                                        "trotter_steps": int(steps),
                                        "cx_proxy_total": int(cost["cx_proxy_total"]),
                                        "ordered_labels": list(cost["ordered_labels"]),
                                        "static_coeff_map": dict(cost["static_coeff_map"]),
                                    }
                                )

    rows_public = [r.to_public_dict() for r in records]

    calibration = _maybe_calibrate_transpile(
        rows_internal=rows_internal_for_calib,
        active_coeff_tol=float(active_coeff_tol),
        t_final=float(config.t_final),
        enabled=bool(config.calibrate_transpile),
        backend_request=str(config.calibration_backend),
        strict=bool(config.calibration_strict),
    )

    if isinstance(calibration.get("samples"), list):
        by_run = {
            str(s.get("run_id")): s
            for s in calibration.get("samples", [])
            if isinstance(s, Mapping)
        }
        for row in rows_public:
            sample = by_run.get(str(row["run_id"]))
            if sample is None:
                continue
            if str(sample.get("status", "")) != "ok":
                continue
            t2q = sample.get("transpiled_2q_count")
            tdp = sample.get("transpiled_depth")
            if t2q is not None:
                row["transpiled_2q_count"] = int(t2q)
            if tdp is not None:
                row["transpiled_depth"] = int(tdp)

    slope_fits = _build_slope_fits(rows_public, config.error_metrics)
    pareto = _build_pareto_by_metric(
        rows_public,
        error_metrics=config.error_metrics,
        cost_axes=config.equal_cost_axis,
    )

    exact_ties: dict[str, list[dict[str, Any]]] = {}
    for axis in config.equal_cost_axis:
        if axis == "wall_time":
            exact_ties[axis] = _build_walltime_near_ties(rows_public, rel_tol=0.05)
        else:
            exact_ties[axis] = _build_exact_tie_tables(rows_public, axis=axis)

    fallback_appendix: dict[str, list[dict[str, Any]]] = {}
    if bool(config.include_fallback_appendix):
        for axis in config.equal_cost_axis:
            fallback_appendix[axis] = _build_fallback_appendix(rows_public, axis=axis)

    settings_block = {
        "problem_grid": [str(x) for x in config.problem_grid],
        "drive_grid": [str(x) for x in config.drive_grid],
        "methods": [str(x) for x in config.methods],
        "stage_mode_grid": [str(x) for x in config.stage_mode_grid],
        "t_final": float(config.t_final),
        "num_times": int(config.num_times),
        "reference_steps_multiplier": int(config.reference_steps_multiplier),
        "error_metrics": [str(x) for x in config.error_metrics],
        "cost_metrics": [str(x) for x in config.cost_metrics],
        "equal_cost_axis": [str(x) for x in config.equal_cost_axis],
        "equal_cost_policy": str(config.equal_cost_policy),
        "calibrate_transpile": bool(config.calibrate_transpile),
        "calibration_backend": str(config.calibration_backend),
        "calibration_strict": bool(config.calibration_strict),
        "include_fallback_appendix": bool(config.include_fallback_appendix),
        "sinusoid_omegas": [float(x) for x in config.sinusoid_omegas],
        "gaussian_tbars": [float(x) for x in config.gaussian_tbars],
        "initial_state_source": str(config.initial_state_source),
        "adapt_input_json": None if config.adapt_input_json is None else str(config.adapt_input_json),
        "adapt_strict_match": bool(config.adapt_strict_match),
        "t_default": 1.0,
        "u_default": 4.0,
        "dv_default": 0.0,
    }

    payload = {
        "schema": "cfqm_efficiency_suite_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "settings": settings_block,
        "runs": rows_public,
        "slope_fits": slope_fits,
        "pareto_by_metric": pareto,
        "equal_cost_exact_ties": exact_ties,
        "fallback_appendix": fallback_appendix,
        "transpile_calibration": calibration,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    runs_json = out_dir / "runs_full.json"
    runs_csv = out_dir / "runs_full.csv"
    summary_json = out_dir / "summary_by_scenario.json"
    pareto_json = out_dir / "pareto_by_metric.json"
    slope_json = out_dir / "slope_fits.json"
    pdf_path = out_dir / "cfqm_efficiency_suite.pdf"

    runs_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(runs_csv, rows_public)
    summary_json.write_text(
        json.dumps(
            {
                "equal_cost_exact_ties": exact_ties,
                "fallback_appendix": fallback_appendix,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    pareto_json.write_text(json.dumps(pareto, indent=2), encoding="utf-8")
    slope_json.write_text(json.dumps(slope_fits, indent=2), encoding="utf-8")

    _write_efficiency_pdf(output_pdf=pdf_path, payload=payload)

    # Per-axis exact tie CSVs.
    for axis, tie_rows in exact_ties.items():
        flat_rows: list[dict[str, Any]] = []
        for block in tie_rows:
            if not isinstance(block, Mapping):
                continue
            target = float(block.get("target_cost", float("nan")))
            for rr in block.get("rows", []) if isinstance(block.get("rows"), list) else []:
                if not isinstance(rr, Mapping):
                    continue
                flat_rows.append(
                    {
                        "scenario_id": str(block.get("scenario_id", "")),
                        "drive_case_id": str(block.get("drive_case_id", "")),
                        "track": str(block.get("track", "")),
                        "axis": str(axis),
                        "target_cost": target,
                        "method": str(rr.get("method", "")),
                        "stage_mode": str(rr.get("stage_mode", "")),
                        "trotter_steps": int(rr.get("trotter_steps", 0)),
                        "max_energy_abs_err": float(rr.get("max_energy_abs_err", float("nan"))),
                        "max_doublon_abs_err": float(rr.get("max_doublon_abs_err", float("nan"))),
                        "final_infidelity": float(rr.get("final_infidelity", float("nan"))),
                    }
                )
        _write_csv(out_dir / f"equal_cost_exact_ties_{axis}.csv", flat_rows)

    return {
        "payload": payload,
        "runs_json": runs_json,
        "runs_csv": runs_csv,
        "summary_json": summary_json,
        "pareto_json": pareto_json,
        "slope_json": slope_json,
        "pdf": pdf_path,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CFQM vs Suzuki efficiency suite: error-vs-cost and apples-to-apples ties."
    )

    p.add_argument("--problem-grid", type=str, default="hubbard_L4,hh_L2_nb2,hh_L2_nb3")
    p.add_argument("--drive-grid", type=str, default="sinusoid,gaussian_sharp")
    p.add_argument("--methods", type=str, default="suzuki2,cfqm4,cfqm6")
    p.add_argument("--stage-mode-grid", type=str, default="exact_sparse,exact_dense,pauli_suzuki2")
    p.add_argument("--steps-grid", type=str, default=None)
    p.add_argument("--reference-steps-multiplier", type=int, default=8)

    p.add_argument(
        "--error-metrics",
        type=str,
        default="final_infidelity,max_doublon_abs_err,max_energy_abs_err",
    )
    p.add_argument(
        "--cost-metrics",
        type=str,
        default="expm_calls,pauli_rot_count,cx_proxy,depth_proxy,wall_time",
    )
    p.add_argument(
        "--equal-cost-axis",
        type=str,
        default="cx_proxy,pauli_rot_count,expm_calls,wall_time",
    )
    p.add_argument("--equal-cost-policy", type=str, choices=["exact_tie_only"], default="exact_tie_only")
    p.add_argument("--calibrate-transpile", action="store_true")
    p.add_argument("--calibration-backend", type=str, default="auto")
    p.add_argument("--calibration-strict", action="store_true")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "cfqm_efficiency_benchmark")
    p.add_argument("--include-fallback-appendix", action="store_true", default=True)
    p.add_argument("--no-fallback-appendix", dest="include_fallback_appendix", action="store_false")

    p.add_argument("--boundary", choices=["periodic", "open"], default="periodic")
    p.add_argument("--ordering", choices=["blocked", "interleaved"], default="blocked")
    p.add_argument("--t-final", type=float, default=10.0)
    p.add_argument("--num-times", type=int, default=201)
    p.add_argument("--drive-A", type=float, default=0.2)
    p.add_argument("--drive-phi", type=float, default=0.0)
    p.add_argument("--drive-include-identity", action="store_true")
    p.add_argument("--drive-time-sampling", choices=["midpoint", "left", "right"], default="midpoint")
    p.add_argument("--sinusoid-omegas", type=str, default="0.5,2.0,8.0")
    p.add_argument("--gaussian-tbars", type=str, default="0.25,0.5")

    p.add_argument("--initial-state-source", choices=["exact", "vqe", "hf", "adapt_json"], default="exact")
    p.add_argument("--adapt-input-json", type=Path, default=None)
    p.set_defaults(adapt_strict_match=True)
    p.add_argument(
        "--adapt-strict-match",
        dest="adapt_strict_match",
        action="store_true",
        help="Require strict metadata match for adapt_json initial-state imports.",
    )
    p.add_argument(
        "--no-adapt-strict-match",
        dest="adapt_strict_match",
        action="store_false",
        help="Allow adapt_json initial-state imports with metadata mismatches.",
    )
    p.add_argument("--vqe-ansatz", choices=["uccsd", "hva", "hh_hva", "hh_hva_tw", "hh_hva_ptw"], default="uccsd")
    p.add_argument("--vqe-reps", type=int, default=2)
    p.add_argument("--vqe-restarts", type=int, default=2)
    p.add_argument("--vqe-maxiter", type=int, default=600)
    p.add_argument("--vqe-method", choices=["SLSQP", "COBYLA", "L-BFGS-B", "Powell", "Nelder-Mead"], default="COBYLA")
    p.add_argument("--adapt-pool", type=str, default="auto")
    p.add_argument("--adapt-max-depth", type=int, default=2)
    p.add_argument("--adapt-maxiter", type=int, default=30)

    return p.parse_args(argv)


def _to_config(args: argparse.Namespace) -> EfficiencyConfig:
    problem_grid = _parse_csv(str(args.problem_grid))
    drive_grid = _parse_csv(str(args.drive_grid))
    methods = _parse_csv(str(args.methods))
    stage_mode_grid = _parse_csv(str(args.stage_mode_grid))
    error_metrics = _parse_csv(str(args.error_metrics))
    cost_metrics = _parse_csv(str(args.cost_metrics))
    equal_cost_axis = _parse_csv(str(args.equal_cost_axis))
    sinusoid_omegas = _parse_csv_positive_floats(str(args.sinusoid_omegas))
    gaussian_tbars = _parse_csv_positive_floats(str(args.gaussian_tbars))

    for token in problem_grid:
        if token not in _SUPPORTED_PROBLEMS:
            raise ValueError(f"Unsupported problem-grid token '{token}'.")
    for token in drive_grid:
        if token not in _SUPPORTED_DRIVES:
            raise ValueError(f"Unsupported drive-grid token '{token}'.")
    for token in methods:
        if token not in _SUPPORTED_METHODS:
            raise ValueError(f"Unsupported method '{token}'.")
    for token in stage_mode_grid:
        if token not in _SUPPORTED_STAGE_MODES:
            raise ValueError(f"Unsupported stage mode '{token}'.")
    for token in error_metrics:
        if token not in _SUPPORTED_ERROR_METRICS:
            raise ValueError(f"Unsupported error metric '{token}'.")
    for token in cost_metrics:
        if token not in _SUPPORTED_COST_METRICS:
            raise ValueError(f"Unsupported cost metric '{token}'.")
    for token in equal_cost_axis:
        if token not in _SUPPORTED_EQUAL_COST_AXIS:
            raise ValueError(f"Unsupported equal-cost axis '{token}'.")

    steps_override = None if args.steps_grid is None else _parse_csv_ints(str(args.steps_grid))

    if int(args.reference_steps_multiplier) < 1:
        raise ValueError("--reference-steps-multiplier must be >= 1")
    if float(args.t_final) <= 0.0:
        raise ValueError("--t-final must be > 0")
    if int(args.num_times) < 2:
        raise ValueError("--num-times must be >= 2")
    if str(args.initial_state_source).strip().lower() == "adapt_json":
        if args.adapt_input_json is None:
            raise ValueError("--adapt-input-json is required when --initial-state-source adapt_json")
        adapt_path = Path(args.adapt_input_json)
        if not adapt_path.exists():
            raise FileNotFoundError(f"ADAPT input JSON not found: {adapt_path}")
    else:
        adapt_path = None
    calibration_backend = str(args.calibration_backend).strip()
    if not calibration_backend:
        raise ValueError("--calibration-backend must be non-empty")

    return EfficiencyConfig(
        problem_grid=tuple(problem_grid),
        drive_grid=tuple(drive_grid),
        methods=tuple(methods),
        stage_mode_grid=tuple(stage_mode_grid),
        steps_grid_override=steps_override,
        reference_steps_multiplier=int(args.reference_steps_multiplier),
        error_metrics=tuple(error_metrics),
        cost_metrics=tuple(cost_metrics),
        equal_cost_axis=tuple(equal_cost_axis),
        equal_cost_policy=str(args.equal_cost_policy),
        calibrate_transpile=bool(args.calibrate_transpile),
        calibration_backend=calibration_backend,
        calibration_strict=bool(args.calibration_strict),
        output_dir=Path(args.output_dir),
        include_fallback_appendix=bool(args.include_fallback_appendix),
        boundary=str(args.boundary),
        ordering=str(args.ordering),
        t_final=float(args.t_final),
        num_times=int(args.num_times),
        drive_A=float(args.drive_A),
        drive_phi=float(args.drive_phi),
        drive_include_identity=bool(args.drive_include_identity),
        drive_time_sampling=str(args.drive_time_sampling),
        sinusoid_omegas=tuple(float(x) for x in sinusoid_omegas),
        gaussian_tbars=tuple(float(x) for x in gaussian_tbars),
        initial_state_source=str(args.initial_state_source),
        adapt_input_json=adapt_path,
        adapt_strict_match=bool(args.adapt_strict_match),
        vqe_ansatz=str(args.vqe_ansatz),
        vqe_reps=int(args.vqe_reps),
        vqe_restarts=int(args.vqe_restarts),
        vqe_maxiter=int(args.vqe_maxiter),
        vqe_method=str(args.vqe_method),
        adapt_pool=str(args.adapt_pool),
        adapt_max_depth=int(args.adapt_max_depth),
        adapt_maxiter=int(args.adapt_maxiter),
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = _to_config(args)
    res = run_efficiency_suite(cfg)

    print(f"WROTE {res['runs_json']}")
    print(f"WROTE {res['runs_csv']}")
    print(f"WROTE {res['summary_json']}")
    print(f"WROTE {res['pareto_json']}")
    print(f"WROTE {res['slope_json']}")
    print(f"WROTE {res['pdf']}")


if __name__ == "__main__":
    main()
