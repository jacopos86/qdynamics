#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from docs.reports.pdf_utils import (
    current_command_string,
    get_PdfPages,
    get_plt,
    render_command_page,
    render_compact_table,
    render_parameter_manifest,
    require_matplotlib,
)
from docs.reports.report_pages import render_executive_summary_page

_WINDOW_START_DEFAULT = 4.2
_WINDOW_END_DEFAULT = 5.7


"run = payload ⊕ seed_settings ⊕ sibling logs ⊕ derived trajectory arrays"
@dataclass(frozen=True)
class RealtimeRunReport:
    input_json: Path
    payload: dict[str, Any]
    run_tag: str
    seed_json: Path
    seed_settings: dict[str, Any]
    summary: dict[str, Any]
    controller_config: dict[str, Any]
    drive_config: dict[str, Any]
    loader_summary: dict[str, Any]
    reference: dict[str, Any]
    trajectory_rows: list[dict[str, Any]]
    times: np.ndarray
    fidelity_exact: np.ndarray
    abs_energy_total_error: np.ndarray
    site_occupations_abs_error_max: np.ndarray
    logical_block_count: np.ndarray
    runtime_parameter_count: np.ndarray
    rho_miss: np.ndarray
    motion_kink_score: np.ndarray
    energy_total_controller: np.ndarray
    energy_total_exact: np.ndarray
    action_kinds: tuple[str, ...]
    command_text: str | None
    exit_code_text: str | None
    signal_text: str | None
    started_at_text: str | None
    finished_at_text: str | None


"metrics(window) = extrema + action totals + first append time over t in [t0, t1]"
@dataclass(frozen=True)
class WindowMetrics:
    t_start: float
    t_end: float
    num_points: int
    min_fidelity_exact: float
    max_abs_energy_total_error: float
    max_site_occupations_abs_error: float
    append_count: int
    retarget_count: int
    stay_count: int
    first_append_time: float | None
    first_retarget_time: float | None
    block_count_start: float
    block_count_end: float
    runtime_parameter_count_start: float
    runtime_parameter_count_end: float


"payload = json(path)"
def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(payload).__name__}.")
    return payload


"text(path) = stripped file contents if path exists else None"
def _read_text_optional(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="replace").strip() or None


"arr_i = float(row_i[key]) if present else nan"
def _array_from_rows(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        raw = row.get(key, float("nan"))
        try:
            values.append(float(raw))
        except Exception:
            values.append(float("nan"))
    return np.asarray(values, dtype=float)


"actions_i = str(row_i[action_kind])"
def _actions_from_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(str(row.get("action_kind", "unknown")) for row in rows)


"seed_settings = json(seed_json).settings"
def _load_seed_settings(seed_json: Path) -> dict[str, Any]:
    payload = _read_json(seed_json)
    settings = payload.get("settings", {})
    if not isinstance(settings, Mapping):
        raise KeyError(f"Missing settings mapping in seed artifact {seed_json}.")
    return dict(settings)


"log_dir(input_json) = sibling run/logs directory"
def _infer_log_dir(input_json: Path) -> Path:
    json_dir = input_json.resolve().parent
    return json_dir.parent / "logs"


"run = parse realtime controller artifact and adjacent log files"
def load_realtime_run(input_json: str | Path) -> RealtimeRunReport:
    path = Path(input_json).expanduser().resolve()
    payload = _read_json(path)
    trajectory = payload.get("trajectory", [])
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError(f"Expected non-empty top-level trajectory in {path}.")
    rows = [dict(row) for row in trajectory if isinstance(row, Mapping)]
    if len(rows) != len(trajectory):
        raise TypeError(f"Trajectory rows must be mappings in {path}.")

    summary = payload.get("summary", {})
    controller_config = payload.get("controller_config", {})
    drive_config = payload.get("drive_config", {}) or {}
    loader_summary = payload.get("loader_summary", {})
    reference = payload.get("reference", {})
    if not isinstance(summary, Mapping):
        raise TypeError("summary must be a mapping.")
    if not isinstance(controller_config, Mapping):
        raise TypeError("controller_config must be a mapping.")
    if not isinstance(drive_config, Mapping):
        raise TypeError("drive_config must be a mapping.")
    if not isinstance(loader_summary, Mapping):
        raise TypeError("loader_summary must be a mapping.")
    if not isinstance(reference, Mapping):
        raise TypeError("reference must be a mapping.")

    seed_json = Path(str(payload.get("artifact_json", ""))).expanduser().resolve()
    if not seed_json.exists():
        raise FileNotFoundError(f"Seed artifact referenced by report input does not exist: {seed_json}")
    seed_settings = _load_seed_settings(seed_json)

    log_dir = _infer_log_dir(path)
    command_text = _read_text_optional(log_dir / "command.sh")
    exit_code_text = _read_text_optional(log_dir / "exit_code.txt")
    signal_text = _read_text_optional(log_dir / "signal.txt")
    started_at_text = _read_text_optional(log_dir / "started_at.txt")
    finished_at_text = _read_text_optional(log_dir / "finished_at.txt")

    return RealtimeRunReport(
        input_json=path,
        payload=payload,
        run_tag=str(payload.get("run_tag") or path.stem),
        seed_json=seed_json,
        seed_settings=dict(seed_settings),
        summary=dict(summary),
        controller_config=dict(controller_config),
        drive_config=dict(drive_config),
        loader_summary=dict(loader_summary),
        reference=dict(reference),
        trajectory_rows=rows,
        times=_array_from_rows(rows, "time"),
        fidelity_exact=_array_from_rows(rows, "fidelity_exact"),
        abs_energy_total_error=_array_from_rows(rows, "abs_energy_total_error"),
        site_occupations_abs_error_max=_array_from_rows(rows, "site_occupations_abs_error_max"),
        logical_block_count=_array_from_rows(rows, "logical_block_count"),
        runtime_parameter_count=_array_from_rows(rows, "runtime_parameter_count"),
        rho_miss=_array_from_rows(rows, "rho_miss"),
        motion_kink_score=_array_from_rows(rows, "motion_kink_score"),
        energy_total_controller=_array_from_rows(rows, "energy_total_controller"),
        energy_total_exact=_array_from_rows(rows, "energy_total_exact"),
        action_kinds=_actions_from_rows(rows),
        command_text=command_text,
        exit_code_text=exit_code_text,
        signal_text=signal_text,
        started_at_text=started_at_text,
        finished_at_text=finished_at_text,
    )


"mask_i = 1 iff t_start <= t_i <= t_end"
def _window_mask(times: np.ndarray, *, t_start: float, t_end: float) -> np.ndarray:
    t = np.asarray(times, dtype=float)
    return np.asarray((t >= float(t_start)) & (t <= float(t_end)), dtype=bool)


"count(action == target over mask)"
def _masked_action_count(actions: Sequence[str], mask: np.ndarray, target: str) -> int:
    return int(sum(1 for action, keep in zip(actions, mask, strict=False) if keep and str(action) == str(target)))


"first_time(action == target over mask) = min t"
def _masked_first_action_time(times: np.ndarray, actions: Sequence[str], mask: np.ndarray, target: str) -> float | None:
    for t, action, keep in zip(times, actions, mask, strict=False):
        if keep and str(action) == str(target):
            return float(t)
    return None


"metrics = summarize run inside diagnostic window"
def compute_window_metrics(run: RealtimeRunReport, *, t_start: float, t_end: float) -> WindowMetrics:
    mask = _window_mask(run.times, t_start=t_start, t_end=t_end)
    if not np.any(mask):
        raise ValueError(f"No trajectory samples in window [{t_start}, {t_end}] for {run.input_json}.")
    idx = np.flatnonzero(mask)
    return WindowMetrics(
        t_start=float(t_start),
        t_end=float(t_end),
        num_points=int(idx.size),
        min_fidelity_exact=float(np.nanmin(run.fidelity_exact[mask])),
        max_abs_energy_total_error=float(np.nanmax(run.abs_energy_total_error[mask])),
        max_site_occupations_abs_error=float(np.nanmax(run.site_occupations_abs_error_max[mask])),
        append_count=_masked_action_count(run.action_kinds, mask, "append_candidate"),
        retarget_count=_masked_action_count(run.action_kinds, mask, "oracle_retarget"),
        stay_count=_masked_action_count(run.action_kinds, mask, "stay"),
        first_append_time=_masked_first_action_time(run.times, run.action_kinds, mask, "append_candidate"),
        first_retarget_time=_masked_first_action_time(run.times, run.action_kinds, mask, "oracle_retarget"),
        block_count_start=float(run.logical_block_count[idx[0]]),
        block_count_end=float(run.logical_block_count[idx[-1]]),
        runtime_parameter_count_start=float(run.runtime_parameter_count[idx[0]]),
        runtime_parameter_count_end=float(run.runtime_parameter_count[idx[-1]]),
    )


"text = comma_join(sorted(unique(actions)))"
def _unique_action_summary(actions: Sequence[str]) -> str:
    unique = sorted({str(action) for action in actions})
    return ", ".join(unique) if unique else "none"


"x_fmt = finite(x) ? formatted(x) : 'n/a'"
def _fmt_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    try:
        x = float(value)
    except Exception:
        return "n/a"
    if not math.isfinite(x):
        return "n/a"
    return f"{x:.{digits}g}"


"page = manifest(first) with reproducibility fields"
def _render_manifest_page(pdf: Any, run: RealtimeRunReport, *, window_start: float, window_end: float) -> None:
    settings = run.seed_settings
    summary = run.summary
    extra = {
        "L": settings.get("L"),
        "omega0": settings.get("omega0"),
        "g_ep": settings.get("g_ep"),
        "n_ph_max": settings.get("n_ph_max"),
        "boundary": settings.get("boundary"),
        "ordering": settings.get("ordering"),
        "seed_adapt_pool": settings.get("adapt_pool"),
        "loader_mode": run.payload.get("loader_mode"),
        "generator_family": run.loader_summary.get("generator_family"),
        "fallback_family": run.loader_summary.get("fallback_family"),
        "resolved_family": run.loader_summary.get("resolved_family"),
        "reference_method": run.reference.get("reference_method"),
        "reference_steps_multiplier": run.reference.get("reference_steps_multiplier"),
        "drive_pattern": run.drive_config.get("drive_pattern"),
        "drive_A": run.drive_config.get("drive_A"),
        "drive_omega": run.drive_config.get("drive_omega"),
        "num_times": len(run.trajectory_rows),
        "t_final": float(run.times[-1]),
        "window_start": float(window_start),
        "window_end": float(window_end),
        "run_tag": run.run_tag,
    }
    render_parameter_manifest(
        pdf,
        model="Hubbard-Holstein",
        ansatz="artifact-seeded realtime checkpoint controller",
        drive_enabled=bool(run.drive_config.get("enabled", False)),
        t=float(settings.get("t", 0.0) or 0.0),
        U=float(settings.get("u", settings.get("U", 0.0)) or 0.0),
        dv=float(settings.get("dv", 0.0) or 0.0),
        extra=extra,
    )


"summary_page = objective + verdict + endpoint + provenance"
def _render_summary_page(
    pdf: Any,
    run: RealtimeRunReport,
    *,
    window: WindowMetrics,
    compare_window: WindowMetrics | None,
    objective_text: str | None,
) -> None:
    final_idx = -1
    summary_sections: list[tuple[str, Sequence[tuple[str, Any]]]] = [
        (
            "Objective",
            [
                (
                    "target",
                    objective_text
                    or "Run the heavier HH realtime controller uninterrupted and judge whether the old bad window persists.",
                ),
            ],
        ),
        (
            "Run status",
            [
                ("status", run.summary.get("status", "unknown")),
                ("exit_code", run.exit_code_text or "n/a"),
                ("signal", run.signal_text or "n/a"),
                ("started_at", run.started_at_text or "n/a"),
                ("finished_at", run.finished_at_text or "n/a"),
            ],
        ),
        (
            "Headline results",
            [
                ("window min fidelity", _fmt_float(window.min_fidelity_exact)),
                ("window max |ΔE|", _fmt_float(window.max_abs_energy_total_error)),
                ("window max site error", _fmt_float(window.max_site_occupations_abs_error)),
                ("window append / retarget / stay", f"{window.append_count} / {window.retarget_count} / {window.stay_count}"),
                ("first append in window", _fmt_float(window.first_append_time)),
                ("first retarget in window", _fmt_float(window.first_retarget_time)),
            ],
        ),
        (
            "Final endpoint",
            [
                ("t_final", _fmt_float(run.times[final_idx])),
                ("final fidelity", _fmt_float(run.fidelity_exact[final_idx])),
                ("final |ΔE|", _fmt_float(run.abs_energy_total_error[final_idx])),
                ("final site error", _fmt_float(run.site_occupations_abs_error_max[final_idx])),
                ("final logical blocks", int(round(float(run.logical_block_count[final_idx])))),
                ("final runtime params", int(round(float(run.runtime_parameter_count[final_idx])))),
            ],
        ),
        (
            "Provenance",
            [
                ("seed artifact", str(run.seed_json)),
                ("loader_mode", run.payload.get("loader_mode")),
                ("generator_family", run.loader_summary.get("generator_family")),
                ("fallback_family", run.loader_summary.get("fallback_family")),
                ("actions seen", _unique_action_summary(run.action_kinds)),
            ],
        ),
    ]
    notes: list[str] = []
    if compare_window is not None:
        notes.append(
            "Comparison baseline provided: the old failure window still persists if the current run does not dominate the baseline on window stay-count and error peaks."
        )
    render_executive_summary_page(
        pdf,
        title="HH realtime controller results",
        experiment_statement=str(run.run_tag),
        sections=summary_sections,
        notes=notes,
    )


"full_run_page = fidelity + errors + manifold + action timeline"
def _render_overview_page(pdf: Any, run: RealtimeRunReport, *, t_start: float, t_end: float) -> None:
    plt = get_plt()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(run.times, run.fidelity_exact, color="#1f77b4", linewidth=2.0)
    ax.axvspan(t_start, t_end, color="#f4d03f", alpha=0.18)
    ax.set_title("Fidelity vs time")
    ax.set_xlabel("time")
    ax.set_ylabel("fidelity_exact")
    ax.set_ylim(max(0.0, float(np.nanmin(run.fidelity_exact)) - 0.05), 1.0)
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.plot(run.times, run.abs_energy_total_error, color="#c0392b", linewidth=2.0, label="|ΔE|")
    ax.plot(
        run.times,
        run.site_occupations_abs_error_max,
        color="#27ae60",
        linewidth=2.0,
        label="site max abs err",
    )
    ax.axvspan(t_start, t_end, color="#f4d03f", alpha=0.18)
    ax.set_title("Primary errors vs time")
    ax.set_xlabel("time")
    ax.set_ylabel("error")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1, 0]
    ax.plot(run.times, run.logical_block_count, color="#6c3483", linewidth=2.0, label="logical blocks")
    ax.plot(
        run.times,
        run.runtime_parameter_count,
        color="#d35400",
        linewidth=2.0,
        label="runtime params",
    )
    ax.axvspan(t_start, t_end, color="#f4d03f", alpha=0.18)
    ax.set_title("Manifold size vs time")
    ax.set_xlabel("time")
    ax.set_ylabel("count")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1, 1]
    y_map = {"append_candidate": 1.0, "oracle_retarget": 2.0, "stay": 0.0}
    action_y = np.asarray([y_map.get(action, -1.0) for action in run.action_kinds], dtype=float)
    colors = [
        "#2980b9" if action == "append_candidate" else ("#d35400" if action == "oracle_retarget" else "#7f8c8d")
        for action in run.action_kinds
    ]
    ax.scatter(run.times, action_y, c=colors, s=12, alpha=0.9)
    ax.axvspan(t_start, t_end, color="#f4d03f", alpha=0.18)
    ax.set_title("Action timeline")
    ax.set_xlabel("time")
    ax.set_ylabel("action")
    ax.set_yticks([0.0, 1.0, 2.0], labels=["stay", "append", "retarget"])
    ax.grid(alpha=0.25)

    pdf.savefig(fig)
    plt.close(fig)


"window_page = zoom on failure window + controller diagnostics"
def _render_window_page(pdf: Any, run: RealtimeRunReport, *, t_start: float, t_end: float) -> None:
    plt = get_plt()
    mask = _window_mask(run.times, t_start=t_start, t_end=t_end)
    tw = run.times[mask]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(tw, run.abs_energy_total_error[mask], color="#c0392b", linewidth=2.0, label="|ΔE|")
    ax.plot(tw, run.site_occupations_abs_error_max[mask], color="#27ae60", linewidth=2.0, label="site err")
    ax.set_title(f"Window errors [{t_start}, {t_end}]")
    ax.set_xlabel("time")
    ax.set_ylabel("error")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[0, 1]
    ax.plot(tw, run.logical_block_count[mask], color="#6c3483", linewidth=2.0, label="logical blocks")
    ax.plot(tw, run.runtime_parameter_count[mask], color="#d35400", linewidth=2.0, label="runtime params")
    ax.set_title("Window manifold growth")
    ax.set_xlabel("time")
    ax.set_ylabel("count")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1, 0]
    append_cum = np.cumsum(np.asarray([1 if action == "append_candidate" else 0 for action in run.action_kinds], dtype=int))
    retarget_cum = np.cumsum(np.asarray([1 if action == "oracle_retarget" else 0 for action in run.action_kinds], dtype=int))
    stay_cum = np.cumsum(np.asarray([1 if action == "stay" else 0 for action in run.action_kinds], dtype=int))
    ax.plot(run.times, append_cum, color="#2980b9", linewidth=2.0, label="append cumulative")
    ax.plot(run.times, retarget_cum, color="#d35400", linewidth=2.0, label="retarget cumulative")
    ax.plot(run.times, stay_cum, color="#7f8c8d", linewidth=2.0, label="stay cumulative")
    ax.axvspan(t_start, t_end, color="#f4d03f", alpha=0.18)
    ax.set_title("Cumulative actions")
    ax.set_xlabel("time")
    ax.set_ylabel("count")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1, 1]
    ax.plot(tw, run.rho_miss[mask], color="#16a085", linewidth=2.0, label="rho_miss")
    ax.plot(tw, run.motion_kink_score[mask], color="#8e44ad", linewidth=2.0, label="motion_kink")
    ax.set_title("Controller diagnostics in window")
    ax.set_xlabel("time")
    ax.set_ylabel("score")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)

    pdf.savefig(fig)
    plt.close(fig)


"compare_page = overlay current and baseline inside diagnostic window"
def _render_compare_page(
    pdf: Any,
    run: RealtimeRunReport,
    compare_run: RealtimeRunReport,
    *,
    window: WindowMetrics,
    compare_window: WindowMetrics,
    t_start: float,
    t_end: float,
) -> None:
    plt = get_plt()
    mask_run = _window_mask(run.times, t_start=t_start, t_end=t_end)
    mask_cmp = _window_mask(compare_run.times, t_start=t_start, t_end=t_end)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(compare_run.times[mask_cmp], compare_run.fidelity_exact[mask_cmp], color="#95a5a6", linewidth=2.0, label=f"{compare_run.run_tag}")
    ax.plot(run.times[mask_run], run.fidelity_exact[mask_run], color="#1f77b4", linewidth=2.0, label=f"{run.run_tag}")
    ax.set_title("Window fidelity comparison")
    ax.set_xlabel("time")
    ax.set_ylabel("fidelity_exact")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=8)

    ax = axes[0, 1]
    ax.plot(compare_run.times[mask_cmp], compare_run.abs_energy_total_error[mask_cmp], color="#e67e22", linewidth=2.0, label=f"{compare_run.run_tag} |ΔE|")
    ax.plot(compare_run.times[mask_cmp], compare_run.site_occupations_abs_error_max[mask_cmp], color="#f1c40f", linewidth=2.0, label=f"{compare_run.run_tag} site")
    ax.plot(run.times[mask_run], run.abs_energy_total_error[mask_run], color="#c0392b", linewidth=2.0, label=f"{run.run_tag} |ΔE|")
    ax.plot(run.times[mask_run], run.site_occupations_abs_error_max[mask_run], color="#27ae60", linewidth=2.0, label=f"{run.run_tag} site")
    ax.set_title("Window error comparison")
    ax.set_xlabel("time")
    ax.set_ylabel("error")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=7)

    ax = axes[1, 0]
    ax.plot(compare_run.times[mask_cmp], compare_run.logical_block_count[mask_cmp], color="#7f8c8d", linewidth=2.0, label=f"{compare_run.run_tag} blocks")
    ax.plot(run.times[mask_run], run.logical_block_count[mask_run], color="#6c3483", linewidth=2.0, label=f"{run.run_tag} blocks")
    ax.plot(compare_run.times[mask_cmp], compare_run.runtime_parameter_count[mask_cmp], color="#bdc3c7", linewidth=2.0, linestyle="--", label=f"{compare_run.run_tag} params")
    ax.plot(run.times[mask_run], run.runtime_parameter_count[mask_run], color="#d35400", linewidth=2.0, linestyle="--", label=f"{run.run_tag} params")
    ax.set_title("Window manifold comparison")
    ax.set_xlabel("time")
    ax.set_ylabel("count")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=7)

    ax = axes[1, 1]
    rows = [
        ["metric", run.run_tag, compare_run.run_tag],
        ["min fidelity", _fmt_float(window.min_fidelity_exact), _fmt_float(compare_window.min_fidelity_exact)],
        ["max |ΔE|", _fmt_float(window.max_abs_energy_total_error), _fmt_float(compare_window.max_abs_energy_total_error)],
        ["max site err", _fmt_float(window.max_site_occupations_abs_error), _fmt_float(compare_window.max_site_occupations_abs_error)],
        ["append / retarget / stay", f"{window.append_count} / {window.retarget_count} / {window.stay_count}", f"{compare_window.append_count} / {compare_window.retarget_count} / {compare_window.stay_count}"],
        ["first append", _fmt_float(window.first_append_time), _fmt_float(compare_window.first_append_time)],
        ["first retarget", _fmt_float(window.first_retarget_time), _fmt_float(compare_window.first_retarget_time)],
        [
            "blocks start→end",
            f"{int(round(window.block_count_start))}→{int(round(window.block_count_end))}",
            f"{int(round(compare_window.block_count_start))}→{int(round(compare_window.block_count_end))}",
        ],
        [
            "params start→end",
            f"{int(round(window.runtime_parameter_count_start))}→{int(round(window.runtime_parameter_count_end))}",
            f"{int(round(compare_window.runtime_parameter_count_start))}→{int(round(compare_window.runtime_parameter_count_end))}",
        ],
    ]
    render_compact_table(
        ax,
        title="Window metric table",
        col_labels=rows[0],
        rows=rows[1:],
        fontsize=7,
    )

    pdf.savefig(fig)
    plt.close(fig)


"provenance_page = original run command + report command + artifact paths"
def _render_provenance_page(pdf: Any, run: RealtimeRunReport, *, output_pdf: Path, compare_json: Path | None) -> None:
    header = [
        f"input_json: {run.input_json}",
        f"seed_artifact_json: {run.seed_json}",
        f"output_pdf: {output_pdf.resolve()}",
    ]
    if compare_json is not None:
        header.append(f"compare_json: {compare_json.resolve()}")
    if run.finished_at_text is not None:
        header.append(f"run_finished_at: {run.finished_at_text}")
    if run.exit_code_text is not None:
        header.append(f"run_exit_code: {run.exit_code_text}")
    original_command = run.command_text or "(command.sh missing)"
    combined = (
        "Original run command\n"
        f"{original_command}\n\n"
        "Report compile command\n"
        f"{current_command_string()}"
    )
    render_command_page(
        pdf,
        combined,
        script_name="pipelines/reporting/hh_realtime_controller_report.py",
        extra_header_lines=header,
    )


"write_pdf = manifest + summary + overview + window + optional compare + provenance"
def write_report_pdf(
    input_json: str | Path,
    *,
    output_pdf: str | Path,
    compare_json: str | Path | None = None,
    objective_text: str | None = None,
    window_start: float = _WINDOW_START_DEFAULT,
    window_end: float = _WINDOW_END_DEFAULT,
) -> Path:
    require_matplotlib()
    run = load_realtime_run(input_json)
    compare_run = None if compare_json is None else load_realtime_run(compare_json)
    output_path = Path(output_pdf).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    window = compute_window_metrics(run, t_start=float(window_start), t_end=float(window_end))
    compare_window = (
        None
        if compare_run is None
        else compute_window_metrics(compare_run, t_start=float(window_start), t_end=float(window_end))
    )

    PdfPages = get_PdfPages()
    with PdfPages(str(output_path)) as pdf:
        _render_manifest_page(pdf, run, window_start=float(window_start), window_end=float(window_end))
        _render_summary_page(
            pdf,
            run,
            window=window,
            compare_window=compare_window,
            objective_text=objective_text,
        )
        _render_overview_page(pdf, run, t_start=float(window_start), t_end=float(window_end))
        _render_window_page(pdf, run, t_start=float(window_start), t_end=float(window_end))
        if compare_run is not None and compare_window is not None:
            _render_compare_page(
                pdf,
                run,
                compare_run,
                window=window,
                compare_window=compare_window,
                t_start=float(window_start),
                t_end=float(window_end),
            )
        _render_provenance_page(
            pdf,
            run,
            output_pdf=output_path,
            compare_json=(None if compare_run is None else compare_run.input_json),
        )
    return output_path


"parser = CLI for realtime controller PDF compilation"
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile a PDF report from an HH realtime controller JSON artifact."
    )
    parser.add_argument("--input-json", type=Path, required=True, help="Completed realtime controller JSON artifact.")
    parser.add_argument("--output-pdf", type=Path, required=True, help="Destination PDF path.")
    parser.add_argument(
        "--compare-json",
        type=Path,
        default=None,
        help="Optional baseline realtime controller JSON for side-by-side window comparison.",
    )
    parser.add_argument(
        "--objective-text",
        type=str,
        default=None,
        help="Optional objective sentence shown on the summary page.",
    )
    parser.add_argument("--window-start", type=float, default=_WINDOW_START_DEFAULT)
    parser.add_argument("--window-end", type=float, default=_WINDOW_END_DEFAULT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_path = write_report_pdf(
        args.input_json,
        output_pdf=args.output_pdf,
        compare_json=args.compare_json,
        objective_text=args.objective_text,
        window_start=float(args.window_start),
        window_end=float(args.window_end),
    )
    print(f"report_pdf={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
