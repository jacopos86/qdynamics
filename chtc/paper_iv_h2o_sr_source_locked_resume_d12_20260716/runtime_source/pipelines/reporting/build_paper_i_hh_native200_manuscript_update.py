#!/usr/bin/env python3
"""Build Paper-I HH maxiter=200 native-SPSA manuscript update assets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.reporting.build_paper_i_hh_pass2_costs_plots import (  # noqa: E402
    _compile_payload_parameterization,
)


REGIME_ORDER = (
    "weak-weak",
    "intermediate-weak",
    "strong-weak",
    "weak-strong",
    "intermediate-strong",
    "strong-strong",
)
REGIME_TEX_LABELS = {
    "weak-weak": r"weak--weak: $(U/t,\lambda)=(0.25,0.25)$",
    "intermediate-weak": r"intermediate--weak: $(U/t,\lambda)=(1.25,0.25)$",
    "strong-weak": r"strong--weak: $(U/t,\lambda)=(8,0.25)$",
    "weak-strong": r"weak--strong: $(U/t,\lambda)=(0.25,1.25)$",
    "intermediate-strong": r"intermediate--strong: $(U/t,\lambda)=(1.25,1.25)$",
    "strong-strong": r"strong--strong: $(U/t,\lambda)=(8,1.25)$",
}
NPH_WORK = {
    "weak-weak": 2,
    "intermediate-weak": 2,
    "strong-weak": 2,
    "weak-strong": 4,
    "intermediate-strong": 4,
    "strong-strong": 4,
}
METHOD_ORDER = ("Append-ADAPT", "Geo-ADAPT", "SNAKE")
METHOD_KEY = {
    "Append-ADAPT": "append",
    "Geo-ADAPT": "geo",
    "SNAKE": "snake",
}
METHOD_STYLE = {
    "Append-ADAPT": {"color": "#4C78A8", "marker": "o"},
    "Geo-ADAPT": {"color": "#54A24B", "marker": "^"},
    "SNAKE": {"color": "#E45756", "marker": "s"},
}
PLOT_FONT = Path("/System/Library/Fonts/Supplemental/Arial.ttf")


@dataclass(frozen=True)
class ManuscriptRow:
    regime: str
    method: str
    reported_iteration: int
    active_depth: int
    same_cutoff_abs_delta_e: float
    n1q: int
    n2q: int
    d2q: int
    dcirc: int
    s: float
    s_status: str
    source_json: Path
    source_sha256: str
    compile_source: str
    trajectory: tuple[tuple[int, float], ...]
    source_batch: str = ""
    trajectory_policy: str = ""


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(path_value: str | Path) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else REPO_ROOT / path


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _int_required(value: Any, *, field: str, source: Path) -> int:
    if value in (None, ""):
        raise ValueError(f"missing {field} in {source}")
    return int(round(float(value)))


def _float_required(value: Any, *, field: str, source: Path) -> float:
    out = _float_or_none(value)
    if out is None:
        raise ValueError(f"missing {field} in {source}")
    return float(out)


def _fmt_sci(value: float) -> str:
    return f"{float(value):.2e}".replace("e-0", "e-").replace("e+0", "e+")


def _fmt_salg(value: float) -> str:
    return str(int(round(value)))


def _result_path(row: Mapping[str, str]) -> Path:
    raw = str(row.get("result_json_rel") or "").strip()
    if raw:
        return _resolve(raw)
    out = _resolve(row["record_output_dir"])
    if row["method_label"] == "SNAKE":
        return out / "json" / "result.json"
    return out / "result" / "generic_static_single.json"


def _current_path(row: Mapping[str, str]) -> Path | None:
    raw = str(row.get("current_json_rel") or "").strip()
    return _resolve(raw) if raw else None


def _parse_generic_trajectory(row: Mapping[str, str], exact_energy: float | None) -> tuple[tuple[int, float], ...]:
    current = _current_path(row)
    if current is None or not current.exists():
        return ()
    points: dict[int, float] = {}
    for line in current.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        if not isinstance(item, Mapping):
            continue
        depth = item.get("depth_after") or item.get("depth")
        try:
            depth_int = int(depth)
        except Exception:
            continue
        err = (
            _float_or_none(item.get("abs_delta_e_same_cutoff"))
            or _float_or_none(item.get("same_cutoff_abs_delta_e"))
            or _float_or_none(item.get("abs_delta_e"))
        )
        energy = _float_or_none(item.get("energy_after")) or _float_or_none(item.get("energy"))
        if err is None and energy is not None and exact_energy is not None:
            err = abs(float(energy) - float(exact_energy))
        if err is not None and err > 0.0:
            points[depth_int] = float(err)
    return tuple(sorted(points.items()))


def _parse_progress_trajectory(record_dir: Path, exact_energy: float | None) -> tuple[tuple[int, float], ...]:
    progress = record_dir / "adapt_iteration_progress.jsonl"
    if exact_energy is None or not progress.exists():
        return ()
    points: dict[int, float] = {}
    for line in progress.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        if not isinstance(item, Mapping) or item.get("event") != "iteration_complete":
            continue
        step = item.get("depth_after")
        if step is None:
            iteration = item.get("iteration")
            if iteration is not None:
                step = int(iteration) + 1
        energy = _float_or_none(item.get("energy_after"))
        if step is None or energy is None:
            continue
        points[int(step)] = abs(float(energy) - float(exact_energy))
    return tuple(sorted(points.items()))


def _parse_snake_trajectory(payload: Mapping[str, Any]) -> tuple[tuple[int, float], ...]:
    adapt = payload.get("adapt_vqe") if isinstance(payload.get("adapt_vqe"), Mapping) else {}
    history = adapt.get("history") or adapt.get("history_tail") or adapt.get("adapt_history") or []
    if not isinstance(history, list):
        return ()
    points: dict[int, float] = {}
    for idx, item in enumerate(history):
        if not isinstance(item, Mapping):
            continue
        depth = item.get("depth") or item.get("depth_after") or (idx + 1)
        err = (
            _float_or_none(item.get("delta_abs_current"))
            or _float_or_none(item.get("benchmark_target_abs_delta_current"))
            or _float_or_none(item.get("abs_delta_e_same_cutoff_after"))
            or _float_or_none(item.get("abs_delta_e_after"))
            or _float_or_none(item.get("abs_delta_e"))
        )
        if err is not None and err > 0.0:
            points[int(depth)] = float(err)
    return tuple(sorted(points.items()))


def _merge_trajectory_prefix(
    current: tuple[tuple[int, float], ...],
    previous: tuple[tuple[int, float], ...],
) -> tuple[tuple[int, float], ...]:
    if not previous:
        return current
    merged: dict[int, float] = {int(x): float(y) for x, y in previous}
    merged.update({int(x): float(y) for x, y in current})
    return tuple(sorted(merged.items()))


def _previous_trajectory_map(previous_support_json: Path | None) -> dict[tuple[str, str], tuple[tuple[int, float], ...]]:
    if previous_support_json is None or not previous_support_json.exists():
        return {}
    payload = _read_json(previous_support_json)
    out: dict[tuple[str, str], tuple[tuple[int, float], ...]] = {}
    for row in payload.get("rows", []):
        if not isinstance(row, Mapping):
            continue
        regime = str(row.get("regime") or "")
        method = str(row.get("method") or "")
        points: list[tuple[int, float]] = []
        for point in row.get("trajectory") or []:
            if isinstance(point, Sequence) and len(point) >= 2:
                x = _float_or_none(point[0])
                y = _float_or_none(point[1])
                if x is not None and y is not None and y > 0.0:
                    points.append((int(x), float(y)))
        if regime and method and points:
            out[(regime, method)] = tuple(sorted(points))
    return out


def _extract_result_dict(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    result = payload.get("result")
    return result if isinstance(result, Mapping) else payload


def _load_consolidated_rows(consolidated_json: Path, previous_support_json: Path | None = None) -> list[ManuscriptRow]:
    manifest = _read_json(consolidated_json)
    previous = _previous_trajectory_map(previous_support_json)
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in manifest.get("rows", []):
        if not isinstance(row, Mapping):
            continue
        by_key[(str(row.get("regime")), str(row.get("method")))] = row

    rows: list[ManuscriptRow] = []
    for regime in REGIME_ORDER:
        for method in METHOD_ORDER:
            key = (regime, method)
            if key not in by_key:
                raise KeyError(f"missing consolidated row: {key}")
            raw = by_key[key]
            source_path = _resolve(str(raw["source_json"]))
            source_payload = _read_json(source_path)
            result = _extract_result_dict(source_payload)
            active_depth = _int_required(raw.get("depth"), field="depth", source=source_path)
            reported_iteration = _int_required(
                raw.get("run_decision_step_max") or active_depth,
                field="run_decision_step_max",
                source=source_path,
            )
            final_delta = _float_required(
                raw.get("same_cutoff_abs_delta_e"),
                field="same_cutoff_abs_delta_e",
                source=source_path,
            )
            if method == "SNAKE":
                trajectory = tuple((int(x), float(y)) for x, y in raw.get("trajectory") or [])
                trajectory = _merge_trajectory_prefix(trajectory, previous.get(key, ()))
            else:
                exact_energy = (
                    _float_or_none(result.get("same_cutoff_exact_gs_energy"))
                    or _float_or_none(result.get("exact_energy"))
                    or _float_or_none(result.get("exact_gs_energy"))
                )
                record_dir = _resolve(str(raw["consolidated_record_dir"]))
                trajectory = _parse_progress_trajectory(record_dir, exact_energy)
                if not trajectory:
                    trajectory = tuple((int(x), float(y)) for x, y in raw.get("trajectory") or [])
            if not trajectory:
                trajectory = ((reported_iteration, final_delta),)
            if trajectory[-1][0] < reported_iteration:
                trajectory = tuple(trajectory) + ((reported_iteration, final_delta),)
            rows.append(
                ManuscriptRow(
                    regime=regime,
                    method=method,
                    reported_iteration=reported_iteration,
                    active_depth=active_depth,
                    same_cutoff_abs_delta_e=final_delta,
                    n1q=_int_required(raw.get("N1q"), field="N1q", source=source_path),
                    n2q=_int_required(raw.get("N2q"), field="N2q", source=source_path),
                    d2q=_int_required(raw.get("D2q"), field="D2q", source=source_path),
                    dcirc=_int_required(raw.get("D_circ"), field="D_circ", source=source_path),
                    s=_float_required(raw.get("S"), field="S", source=source_path),
                    s_status=str(raw.get("S_status") or "missing"),
                    source_json=source_path,
                    source_sha256=str(raw.get("source_sha256") or _sha256(source_path)),
                    compile_source=str(raw.get("compile_source") or "consolidated_manifest"),
                    trajectory=trajectory,
                    source_batch=str(raw.get("source_batch") or ""),
                    trajectory_policy=str(raw.get("trajectory_policy") or "progress_jsonl_or_consolidated_manifest"),
                )
            )
    return rows


def _load_selected_rows(records_tsv: Path, report_json: Path) -> list[ManuscriptRow]:
    report = _read_json(report_json)
    terminal = {
        (
            str(row["display_regime"]),
            str(row["method_label"]),
            str(row["engine_key"]),
            int(row["budget"]),
        ): row
        for row in report.get("terminal_records", [])
        if isinstance(row, Mapping)
    }
    with records_tsv.open(newline="", encoding="utf-8") as handle:
        metadata = [{str(k): "" if v is None else str(v) for k, v in row.items()} for row in csv.DictReader(handle, delimiter="\t")]
    meta = {
        (
            row["display_regime"],
            row["method_label"],
            row["engine_key"],
            int(row["budget"]),
        ): row
        for row in metadata
    }

    rows: list[ManuscriptRow] = []
    for regime in REGIME_ORDER:
        for method in METHOD_ORDER:
            key = (regime, method, "native_forced", 200)
            if key not in terminal or key not in meta:
                raise KeyError(f"missing native/maxiter200 row: {key}")
            terminal_row = terminal[key]
            meta_row = meta[key]
            result_path = _result_path(meta_row)
            payload = _read_json(result_path)
            if method == "SNAKE":
                compiled = _compile_payload_parameterization(
                    payload,
                    payload["adapt_vqe"],
                    source_kind="paper_i_hh_native200_manuscript_terminal",
                )
                compile_source = "table_i_qiskit_reconstructed_from_snake_parameterization"
                trajectory = _parse_snake_trajectory(payload)
            else:
                result = payload.get("result") if isinstance(payload.get("result"), Mapping) else payload
                compiled = result
                compile_source = "generic_static_terminal_qiskit_fields"
                exact = _float_or_none(terminal_row.get("same_cutoff_exact_energy"))
                trajectory = _parse_generic_trajectory(meta_row, exact)
            final_delta = _float_required(
                terminal_row.get("same_cutoff_abs_delta_e"),
                field="same_cutoff_abs_delta_e",
                source=result_path,
            )
            depth = _int_required(terminal_row.get("depth"), field="depth", source=result_path)
            if not trajectory:
                trajectory = ((depth, final_delta),)
            rows.append(
                ManuscriptRow(
                    regime=regime,
                    method=method,
                    reported_iteration=depth,
                    active_depth=depth,
                    same_cutoff_abs_delta_e=final_delta,
                    n1q=_int_required(compiled.get("compiled_count_1q_total"), field="compiled_count_1q_total", source=result_path),
                    n2q=_int_required(compiled.get("compiled_count_2q_total"), field="compiled_count_2q_total", source=result_path),
                    d2q=_int_required(compiled.get("compiled_depth_2q_total"), field="compiled_depth_2q_total", source=result_path),
                    dcirc=_int_required(compiled.get("compiled_depth_total"), field="compiled_depth_total", source=result_path),
                    s=_float_required(terminal_row.get("S_alg"), field="S_alg", source=result_path),
                    s_status=str(terminal_row.get("S_alg_status") or "missing"),
                    source_json=result_path,
                    source_sha256=_sha256(result_path),
                    compile_source=compile_source,
                    trajectory=trajectory,
                )
            )
    return rows


def _plot_iteration_figures(rows: Sequence[ManuscriptRow], figures_dir: Path) -> dict[str, str]:
    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "tmp" / "matplotlib_config"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import FixedLocator, FuncFormatter

    font_name = "Arial"
    if PLOT_FONT.exists():
        font_manager.fontManager.addfont(str(PLOT_FONT))
        font_name = font_manager.FontProperties(fname=str(PLOT_FONT)).get_name()
    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.sans-serif": [font_name],
            "mathtext.fontset": "dejavusans",
            "axes.formatter.use_mathtext": False,
            "pdf.use14corefonts": False,
        }
    )

    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for regime in REGIME_ORDER:
        fig, ax = plt.subplots(figsize=(3.35, 2.35))
        y_values: list[float] = []
        for row in rows:
            if row.regime != regime:
                continue
            style = METHOD_STYLE[row.method]
            xs = [x for x, _ in row.trajectory]
            ys = [max(y, 1e-12) for _, y in row.trajectory]
            y_values.extend(ys)
            ax.plot(
                xs,
                ys,
                label=row.method,
                color=style["color"],
                linewidth=1.2,
                alpha=0.95,
            )
            if xs:
                ax.plot(
                    [xs[-1]],
                    [ys[-1]],
                    color=style["color"],
                    marker=style["marker"],
                    markersize=3.0,
                    linestyle="None",
                    alpha=0.98,
                )
        ax.set_yscale("log")
        ax.set_ylim(max(min(y_values or [1e-4]) * 0.55, 1e-12), 2.0)
        ax.set_xlim(0, 30.5)
        ax.yaxis.set_major_locator(FixedLocator([1.0, 1e-1, 1e-2, 1e-3, 1e-4]))
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _pos: "1" if abs(value - 1.0) < 1e-12 else f"1e{int(round(math.log10(value)))}")
        )
        ax.set_xlabel("ADAPT decision step $k$", fontsize=7)
        ax.set_ylabel(r"same-cutoff $|\Delta E|$", fontsize=7)
        ax.tick_params(axis="both", labelsize=6)
        ax.grid(True, which="both", alpha=0.22, linewidth=0.45)
        ax.legend(fontsize=5.7, frameon=False, loc="best", handlelength=1.8)
        ax.set_title(regime, fontsize=7)
        fig.subplots_adjust(left=0.18, bottom=0.17, right=0.98, top=0.90)
        stem = figures_dir / f"paper_i_hh_native200_{regime.replace('-', '_')}_error_vs_iteration_20260619"
        fig.savefig(stem.with_suffix(".png"), dpi=220)
        plt.close(fig)
        paths[regime] = str(stem.with_suffix(".png").relative_to(REPO_ROOT))
    return paths


def _plot_cost_figures(rows: Sequence[ManuscriptRow], output_dir: Path) -> dict[str, str]:
    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "tmp" / "matplotlib_config"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import FuncFormatter, NullFormatter

    font_name = "Arial"
    if PLOT_FONT.exists():
        font_manager.fontManager.addfont(str(PLOT_FONT))
        font_name = font_manager.FontProperties(fname=str(PLOT_FONT)).get_name()
    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.sans-serif": [font_name],
            "mathtext.fontset": "dejavusans",
            "axes.formatter.use_mathtext": False,
            "pdf.use14corefonts": False,
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for xkey, xlabel, name in (
        ("gate_total", "N1q+N2q", "error_vs_total_1q_2q_gates"),
        ("s_alg", "S", "error_vs_s"),
    ):
        fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.2), sharey=False)
        plain_log_formatter = FuncFormatter(
            lambda value, _pos: ""
            if value <= 0.0
            else (f"{value:.0f}" if value >= 1.0 else f"1e{int(round(math.log10(value))) }".replace(" ", ""))
        )
        for ax, regime in zip(axes.flatten(), REGIME_ORDER):
            regime_points: list[tuple[float, float, ManuscriptRow]] = []
            for row in rows:
                if row.regime != regime:
                    continue
                style = METHOD_STYLE[row.method]
                x = (row.n1q + row.n2q) if xkey == "gate_total" else row.s
                regime_points.append((float(x), float(row.same_cutoff_abs_delta_e), row))
                ax.scatter(
                    [x],
                    [row.same_cutoff_abs_delta_e],
                    label=row.method,
                    color=style["color"],
                    marker=style["marker"],
                    s=32,
                    edgecolors="none",
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            if regime_points:
                xs = [point[0] for point in regime_points]
                ys = [point[1] for point in regime_points]
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                if xmin == xmax:
                    xmin *= 0.75
                    xmax *= 1.35
                else:
                    xpad = 10 ** (0.08 * (math.log10(xmax) - math.log10(xmin)))
                    xmin /= xpad
                    xmax *= xpad
                if ymin == ymax:
                    ymin *= 0.75
                    ymax *= 1.35
                else:
                    ypad = 10 ** (0.10 * (math.log10(ymax) - math.log10(ymin)))
                    ymin /= ypad
                    ymax *= ypad
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                for x, y, row in regime_points:
                    xfrac = (math.log10(x) - math.log10(xmin)) / (math.log10(xmax) - math.log10(xmin))
                    yfrac = (math.log10(y) - math.log10(ymin)) / (math.log10(ymax) - math.log10(ymin))
                    dx = -3 if xfrac > 0.82 else 3
                    dy = -3 if yfrac > 0.82 else 2
                    ha = "right" if xfrac > 0.82 else "left"
                    va = "top" if yfrac > 0.82 else "bottom"
                    ax.annotate(
                        row.method.replace("-ADAPT", ""),
                        (x, y),
                        fontsize=5.5,
                        xytext=(dx, dy),
                        textcoords="offset points",
                        ha=ha,
                        va=va,
                    )
            ax.xaxis.set_major_formatter(plain_log_formatter)
            ax.yaxis.set_major_formatter(plain_log_formatter)
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(True, which="both", alpha=0.22, linewidth=0.45)
            ax.set_title(regime, fontsize=8)
            ax.tick_params(axis="both", labelsize=6)
        for ax in axes[-1, :]:
            ax.set_xlabel(xlabel, fontsize=8)
        for ax in axes[:, 0]:
            ax.set_ylabel("same-cutoff |Delta E|", fontsize=8)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8)
        fig.subplots_adjust(left=0.09, bottom=0.07, right=0.98, top=0.92, hspace=0.42, wspace=0.24)
        path = output_dir / f"paper_i_hh_native200_{name}_20260619.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths[name] = str(path.relative_to(REPO_ROOT))
    return paths


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.relative_to(REPO_ROOT) if value.is_absolute() and REPO_ROOT in value.parents else value)
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def build_assets(
    *,
    records_tsv: Path | None,
    report_json: Path | None,
    consolidated_json: Path | None,
    previous_support_json: Path | None,
    output_json: Path,
    figures_dir: Path,
    cost_output_dir: Path,
) -> dict[str, Any]:
    if consolidated_json is not None:
        rows = _load_consolidated_rows(consolidated_json, previous_support_json)
    elif records_tsv is not None and report_json is not None:
        rows = _load_selected_rows(records_tsv, report_json)
    else:
        raise ValueError("provide --consolidated-json or both --records-tsv and --report-json")
    iteration_figures = _plot_iteration_figures(rows, figures_dir)
    cost_figures = _plot_cost_figures(rows, cost_output_dir)
    payload = {
        "schema": "paper_i_hh_native200_depth30_iteration_horizon_v1",
        "records_tsv": str(records_tsv.relative_to(REPO_ROOT)) if records_tsv is not None else None,
        "report_json": str(report_json.relative_to(REPO_ROOT)) if report_json is not None else None,
        "consolidated_json": str(consolidated_json.relative_to(REPO_ROOT)) if consolidated_json is not None else None,
        "previous_support_json": str(previous_support_json.relative_to(REPO_ROOT)) if previous_support_json is not None else None,
        "contract": {
            "spsa_engine": "native_forced",
            "maxiter": 200,
            "decision_step_horizon": 30,
            "methods": list(METHOD_ORDER),
            "metric": "same_cutoff_abs_delta_e = |E_alg(n_ph_work)-E_ED(n_ph_work)|",
            "row_semantics": "iteration-horizon row at ADAPT decision step k=30; active ansatz depth is a reported output",
            "compile_convention": "Table-I Qiskit transpilation convention",
            "estimator_work_column": "S",
        },
        "iteration_figures": iteration_figures,
        "cost_figures": cost_figures,
        "rows": [
            {
                "regime": row.regime,
                "regime_tex_label": REGIME_TEX_LABELS[row.regime],
                "n_ph_work": NPH_WORK[row.regime],
                "method": row.method,
                "method_key": METHOD_KEY[row.method],
                "reported_iteration": row.reported_iteration,
                "active_depth": row.active_depth,
                "same_cutoff_abs_delta_e": row.same_cutoff_abs_delta_e,
                "same_cutoff_abs_delta_e_tex": _fmt_sci(row.same_cutoff_abs_delta_e),
                "N1q": row.n1q,
                "N2q": row.n2q,
                "D2q": row.d2q,
                "D_circ": row.dcirc,
                "S": row.s,
                "S_tex": _fmt_salg(row.s),
                "S_status": row.s_status,
                "compile_source": row.compile_source,
                "source_batch": row.source_batch,
                "trajectory_policy": row.trajectory_policy,
                "source_json": str(row.source_json.relative_to(REPO_ROOT)),
                "source_sha256": row.source_sha256,
                "trajectory": list(row.trajectory),
            }
            for row in rows
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-tsv", type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--consolidated-json", type=Path)
    parser.add_argument("--previous-support-json", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, default=Path("MATH/paper_details/figures"))
    parser.add_argument("--cost-output-dir", type=Path, default=Path("output/pdf"))
    args = parser.parse_args(argv)
    payload = build_assets(
        records_tsv=_resolve(args.records_tsv) if args.records_tsv else None,
        report_json=_resolve(args.report_json) if args.report_json else None,
        consolidated_json=_resolve(args.consolidated_json) if args.consolidated_json else None,
        previous_support_json=_resolve(args.previous_support_json) if args.previous_support_json else None,
        output_json=_resolve(args.output_json),
        figures_dir=_resolve(args.figures_dir),
        cost_output_dir=_resolve(args.cost_output_dir),
    )
    print(json.dumps({"output_json": str(_resolve(args.output_json).relative_to(REPO_ROOT)), "rows": len(payload["rows"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
