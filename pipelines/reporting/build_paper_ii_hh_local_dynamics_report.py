#!/usr/bin/env python3
"""Build a local Paper-II Hubbard--Holstein dynamics review report.

The report consumes local ``run_local_records.py`` output directories.  It does
not launch dynamics jobs and does not edit manuscripts.  It writes a compact
LaTeX-built review PDF plus a machine-readable summary so repeated four-regime
passes can be compared without hand extraction.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.time_dynamics.tables.dynamics_benchmark_contract import json_safe

METHOD_LABELS = {
    "dyn_controller_full": "AP-McLachlan",
    "dyn_qiskit_trotter_qrte": "Qiskit TrotterQRTE",
    "dyn_qiskit_pvqd": "Qiskit PVQD",
    "dyn_qiskit_varqrte": "Qiskit VarQRTE",
}

METHOD_COLORS = {
    "dyn_controller_full": "#1f77b4",
    "dyn_qiskit_trotter_qrte": "#d62728",
    "dyn_qiskit_pvqd": "#2ca02c",
    "dyn_qiskit_varqrte": "#9467bd",
}

METHOD_ORDER = (
    "dyn_controller_full",
    "dyn_qiskit_trotter_qrte",
    "dyn_qiskit_pvqd",
    "dyn_qiskit_varqrte",
)


@dataclass(frozen=True)
class TrajectorySeries:
    times: np.ndarray
    energy: np.ndarray
    exact_energy: np.ndarray | None
    abs_error: np.ndarray | None


@dataclass(frozen=True)
class LoadedRow:
    record_dir: Path
    record: dict[str, Any]
    row: dict[str, Any] | None
    status: str
    reason: str
    algorithm_id: str
    case_id: str
    regime_id: str
    seed_track: str
    raw_payload_json: Path | None
    raw_payload: dict[str, Any] | None
    trajectory: TrajectorySeries | None


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(dict(payload)), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return float(out) if math.isfinite(out) else None


def _array_from_rows(rows: Sequence[Mapping[str, Any]], *keys: str) -> np.ndarray | None:
    values: list[float] = []
    found = False
    for row in rows:
        raw = None
        for key in keys:
            if key in row:
                raw = row.get(key)
                found = True
                break
        val = _finite_float_or_none(raw)
        values.append(float("nan") if val is None else float(val))
    if not found:
        return None
    arr = np.asarray(values, dtype=float)
    return arr if arr.size else None


def _resolve_path(raw: Any, *, base: Path) -> Path | None:
    if raw in {None, ""}:
        return None
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path
    candidate = (ROOT / path).resolve()
    if candidate.exists():
        return candidate
    return (base / path).resolve()


def _status_from_record_dir(record_dir: Path) -> tuple[str, str]:
    status_json = record_dir / "chtc_status.json"
    if not status_json.exists():
        return "running_or_incomplete", "status file missing"
    payload = _as_mapping(_read_json(status_json))
    rc = payload.get("return_code")
    if rc == 0:
        return "completed", "local runner returned 0"
    return "failed", f"local runner return_code={rc}"


def _rows_from_result_dir(result_dir: Path) -> list[dict[str, Any]]:
    candidates = [
        result_dir / "row.json",
        result_dir / "rows.json",
        result_dir / "summary.json",
    ]
    rows: list[dict[str, Any]] = []
    for path in candidates:
        if not path.exists():
            continue
        payload = _read_json(path)
        if isinstance(payload, Mapping) and isinstance(payload.get("rows"), list):
            rows.extend(dict(row) for row in payload["rows"] if isinstance(row, Mapping))
        elif isinstance(payload, list):
            rows.extend(dict(row) for row in payload if isinstance(row, Mapping))
        elif isinstance(payload, Mapping) and payload.get("schema") == "dynamics_benchmark_row_v1":
            rows.append(dict(payload))
    return rows


def _trajectory_from_payload(payload: Mapping[str, Any]) -> TrajectorySeries | None:
    rows = payload.get("trajectory")
    if not isinstance(rows, list) or not rows:
        return None
    t_rows = [dict(row) for row in rows if isinstance(row, Mapping)]
    if not t_rows:
        return None
    times = _array_from_rows(t_rows, "time", "physical_time")
    energy = _array_from_rows(t_rows, "energy_total", "energy_total_controller")
    exact = _array_from_rows(t_rows, "energy_total_exact")
    error = _array_from_rows(t_rows, "abs_energy_total_error")
    if times is None or energy is None:
        return None
    if error is None and exact is not None and exact.shape == energy.shape:
        error = np.abs(energy - exact)
    return TrajectorySeries(
        times=np.asarray(times, dtype=float),
        energy=np.asarray(energy, dtype=float),
        exact_energy=None if exact is None else np.asarray(exact, dtype=float),
        abs_error=None if error is None else np.asarray(error, dtype=float),
    )


def _seed_lock_from_row(row: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    provenance = _as_mapping(row.get("provenance"))
    seed_lock = _as_mapping(provenance.get("seed_lock"))
    if seed_lock:
        return seed_lock
    case_meta = _as_mapping(_as_mapping(row.get("case")).get("metadata"))
    seed_lock = _as_mapping(case_meta.get("seed_lock"))
    if seed_lock:
        return seed_lock
    return {
        "hh_regime_id": record.get("hh_regime_id"),
        "seed_track": record.get("seed_track"),
    }


def _regime_from_sources(record: Mapping[str, Any], row: Mapping[str, Any] | None) -> str:
    if record.get("hh_regime_id"):
        return str(record["hh_regime_id"])
    if row is not None:
        seed_lock = _seed_lock_from_row(row, record)
        if seed_lock.get("hh_regime_id"):
            return str(seed_lock["hh_regime_id"])
    return "unknown_regime"


def _load_record_dir(record_dir: Path) -> LoadedRow:
    record_path = record_dir / "record.json"
    record = _as_mapping(_read_json(record_path)) if record_path.exists() else {}
    result_dir = record_dir / "result"
    rows = _rows_from_result_dir(result_dir)
    selected = rows[0] if rows else None
    status, reason = _status_from_record_dir(record_dir)
    if selected is not None:
        status = str(selected.get("status") or status)
        reason = str(selected.get("reason") or reason)
    algorithm_id = str((selected or {}).get("algorithm_id") or record.get("algorithm_id") or record_dir.name)
    case_id = str(record.get("case_id") or (selected or {}).get("case_id") or record_dir.name)
    regime_id = _regime_from_sources(record, selected)
    seed_track = str(record.get("seed_track") or "")
    if selected is not None and not seed_track:
        seed_track = str(_seed_lock_from_row(selected, record).get("seed_track") or "")
    raw_path = None
    raw_payload = None
    if selected is not None:
        raw_path = _resolve_path(selected.get("artifact_json"), base=result_dir)
    if raw_path is None and (result_dir / "raw_payload.json").exists():
        raw_path = result_dir / "raw_payload.json"
    if raw_path is not None and raw_path.exists():
        candidate = _read_json(raw_path)
        if isinstance(candidate, Mapping):
            raw_payload = dict(candidate)
    trajectory = _trajectory_from_payload(raw_payload) if raw_payload is not None else None
    return LoadedRow(
        record_dir=record_dir,
        record=record,
        row=selected,
        status=status,
        reason=reason,
        algorithm_id=algorithm_id,
        case_id=case_id,
        regime_id=regime_id,
        seed_track=seed_track or "unknown_seed",
        raw_payload_json=raw_path,
        raw_payload=raw_payload,
        trajectory=trajectory,
    )


def load_rows(records_root: Path) -> list[LoadedRow]:
    if not records_root.exists():
        raise FileNotFoundError(f"records root does not exist: {records_root}")
    out: list[LoadedRow] = []
    for child in sorted(records_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "record.json").exists() or (child / "result").exists():
            out.append(_load_record_dir(child))
    return out


def load_rows_from_roots(records_roots: Sequence[Path]) -> tuple[list[LoadedRow], list[LoadedRow]]:
    """Load rows from one or more local-run roots and keep the best row per key."""

    indexed_rows: list[tuple[int, LoadedRow]] = []
    for root_index, records_root in enumerate(records_roots):
        indexed_rows.extend((root_index, row) for row in load_rows(records_root))

    selected: dict[tuple[str, str, str], tuple[int, LoadedRow]] = {}
    omitted: list[LoadedRow] = []

    def rank(item: tuple[int, LoadedRow]) -> tuple[int, int]:
        root_index, row = item
        completed = 1 if row.status == "completed" else 0
        return (completed, root_index)

    for item in indexed_rows:
        root_index, row = item
        key = (row.regime_id, row.seed_track, row.algorithm_id)
        previous = selected.get(key)
        if previous is None:
            selected[key] = item
            continue
        if rank(item) >= rank(previous):
            omitted.append(previous[1])
            selected[key] = item
        else:
            omitted.append(row)

    rows = [item[1] for item in selected.values()]
    rows.sort(
        key=lambda row: (
            row.regime_id,
            row.seed_track,
            METHOD_ORDER.index(row.algorithm_id) if row.algorithm_id in METHOD_ORDER else 999,
            row.algorithm_id,
            str(row.record_dir),
        )
    )
    omitted.sort(key=lambda row: (row.regime_id, row.seed_track, row.algorithm_id, str(row.record_dir)))
    return rows, omitted


def _metric(row: LoadedRow, key: str) -> Any:
    if row.row is None:
        return None
    table_fields = _as_mapping(row.row.get("table_fields"))
    metrics = _as_mapping(row.row.get("metrics"))
    resources = _as_mapping(row.row.get("resources"))
    for source in (table_fields, metrics, resources):
        if key in source:
            return source.get(key)
    return None


def _fmt_sci(value: Any, *, dash: str = "--") -> str:
    val = _finite_float_or_none(value)
    if val is None:
        return dash
    if val == 0.0:
        return "0"
    return f"{val:.3e}"


def _fmt_int(value: Any, *, dash: str = "--") -> str:
    try:
        if value is None:
            return dash
        return str(int(value))
    except (TypeError, ValueError):
        return dash


def _latex_escape(text: Any) -> str:
    raw = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in raw)


def _shorten_for_pdf(value: Any, *, max_chars: int = 92) -> str:
    raw = str(value)
    if len(raw) <= max_chars:
        return raw
    path = Path(raw)
    parts = path.parts
    if len(parts) >= 3:
        suffix = "/".join(parts[-3:])
        out = ".../" + suffix
        if len(out) <= max_chars:
            return out
    return "..." + raw[-max(8, max_chars - 3) :]


def _compact_root_label(path: Path, *, index: int) -> str:
    name = path.name
    prefix = "paper_ii_hh_local_dynamics_runs_20260626_"
    if name.startswith(prefix):
        name = name[len(prefix) :]
    return f"{index}:{_shorten_for_pdf(name, max_chars=44)}"


def _latex_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], *, small: bool = True) -> str:
    colspec = "l" * len(headers)
    prefix = r"\scriptsize" if small else r"\normalsize"
    lines = [
        prefix,
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        " & ".join(_latex_escape(h) for h in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_latex_escape(cell) for cell in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def _plot_regime(regime: str, rows: Sequence[LoadedRow], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.1), sharex=True, constrained_layout=True)
    exact_plotted = False
    method_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if row.trajectory is not None:
            method_counts[row.algorithm_id] += 1
    for algorithm_id in METHOD_ORDER:
        matches = [row for row in rows if row.algorithm_id == algorithm_id and row.trajectory is not None]
        if not matches:
            continue
        for row in matches:
            series = row.trajectory
            assert series is not None
            label = METHOD_LABELS.get(algorithm_id, algorithm_id)
            if method_counts.get(algorithm_id, 0) > 1:
                label = f"{label} ({row.seed_track})"
            color = METHOD_COLORS.get(algorithm_id)
            linestyle = "--" if row.seed_track == "append" else "-"
            if not exact_plotted and series.exact_energy is not None:
                axes[0].plot(series.times, series.exact_energy, color="black", linewidth=2.0, label="ED/reference")
                exact_plotted = True
            axes[0].plot(series.times, series.energy, color=color, linestyle=linestyle, linewidth=1.6, label=label)
            if series.abs_error is not None:
                clipped = np.maximum(series.abs_error, 1.0e-14)
                axes[1].semilogy(series.times, clipped, color=color, linestyle=linestyle, linewidth=1.6, label=label)
    axes[0].set_ylabel("total energy")
    axes[1].set_ylabel(r"$|\Delta E(t)|$")
    axes[1].set_xlabel("time")
    axes[0].set_title(regime.replace("_", "-"))
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _seed_rows(rows: Sequence[LoadedRow]) -> list[list[Any]]:
    seen: set[tuple[str, str, str]] = set()
    out: list[list[Any]] = []
    for loaded in rows:
        if loaded.row is None:
            continue
        seed_lock = _seed_lock_from_row(loaded.row, loaded.record)
        regime = str(seed_lock.get("hh_regime_id") or loaded.regime_id)
        track = str(seed_lock.get("seed_track") or loaded.seed_track)
        seed_hash = str(
            seed_lock.get("static_seed_artifact_sha256")
            or seed_lock.get("normalized_seed_artifact_sha256")
            or seed_lock.get("seed_artifact_sha256")
            or ""
        )
        key = (regime, track, seed_hash)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            [
                regime,
                track,
                _fmt_sci(seed_lock.get("static_abs_delta_e")),
                _fmt_int(seed_lock.get("static_parameter_count")),
                seed_hash[:12] if seed_hash else "--",
                _shorten_for_pdf(
                    seed_lock.get("selected_static_seed_source")
                    or seed_lock.get("source_artifact_json")
                    or "--"
                ),
            ]
        )
    return sorted(out, key=lambda item: (str(item[0]), str(item[1])))


def _cost_rows(rows: Sequence[LoadedRow]) -> list[list[Any]]:
    out: list[list[Any]] = []
    ordered = sorted(
        rows,
        key=lambda row: (
            row.regime_id,
            METHOD_ORDER.index(row.algorithm_id) if row.algorithm_id in METHOD_ORDER else 999,
            row.algorithm_id,
        ),
    )
    for loaded in ordered:
        out.append(
            [
                loaded.regime_id,
                loaded.seed_track,
                METHOD_LABELS.get(loaded.algorithm_id, loaded.algorithm_id),
                loaded.status,
                _fmt_sci(_metric(loaded, "mean_abs_energy_total_error")),
                _fmt_sci(_metric(loaded, "max_abs_energy_total_error")),
                _fmt_sci(_metric(loaded, "final_abs_energy_total_error")),
                _fmt_sci(_metric(loaded, "epsilon_obs_2")),
                _fmt_int(_metric(loaded, "compiled_count_2q_total")),
                _fmt_int(_metric(loaded, "compiled_depth_2q_total")),
                _fmt_int(_metric(loaded, "compiled_depth_total")),
            ]
        )
    return out


def _row_summary(loaded: LoadedRow) -> dict[str, Any]:
    return {
        "record_dir": str(loaded.record_dir),
        "record_id": loaded.record.get("record_id"),
        "case_id": loaded.case_id,
        "regime_id": loaded.regime_id,
        "seed_track": loaded.seed_track,
        "algorithm_id": loaded.algorithm_id,
        "status": loaded.status,
        "reason": loaded.reason,
        "raw_payload_json": None if loaded.raw_payload_json is None else str(loaded.raw_payload_json),
        "trajectory_points": None if loaded.trajectory is None else int(loaded.trajectory.times.size),
        "mean_abs_energy_total_error": _metric(loaded, "mean_abs_energy_total_error"),
        "max_abs_energy_total_error": _metric(loaded, "max_abs_energy_total_error"),
        "final_abs_energy_total_error": _metric(loaded, "final_abs_energy_total_error"),
        "epsilon_obs_2": _metric(loaded, "epsilon_obs_2"),
        "compiled_count_2q_total": _metric(loaded, "compiled_count_2q_total"),
        "compiled_depth_2q_total": _metric(loaded, "compiled_depth_2q_total"),
        "compiled_depth_total": _metric(loaded, "compiled_depth_total"),
    }


def _status_counts(rows: Iterable[LoadedRow]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.status)] += 1
    return dict(sorted(counts.items()))


def _build_tex(
    *,
    records_roots: Sequence[Path],
    output_dir: Path,
    rows: Sequence[LoadedRow],
    omitted_rows: Sequence[LoadedRow],
    plot_paths: Mapping[str, Path],
    summary_json: Path,
) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()
    complete = [row for row in rows if row.status == "completed"]
    manifest_rows = [
        ("records roots", "; ".join(_compact_root_label(path, index=index + 1) for index, path in enumerate(records_roots))),
        ("output dir", _shorten_for_pdf(output_dir.name, max_chars=56)),
        ("generated UTC", generated_at),
        ("record count", str(len(rows))),
        ("omitted duplicate rows", str(len(omitted_rows))),
        ("completed count", str(len(complete))),
        ("status counts", json.dumps(_status_counts(rows), sort_keys=True)),
        ("summary JSON", summary_json.name),
        ("exact-reference policy", "diagnostic/reporting only"),
    ]
    body: list[str] = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.65in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\usepackage{longtable}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{5pt}",
        r"\begin{document}",
        r"\section*{Paper-II HH Local Dynamics Review}",
        _latex_table(["field", "value"], manifest_rows, small=True),
        r"\section*{Seed Provenance}",
        _latex_table(
            ["regime", "track", r"static |dE|", "params", "seed hash", "source"],
            _seed_rows(rows) or [["--", "--", "--", "--", "--", "no completed seed rows found"]],
            small=True,
        ),
        r"\section*{Dynamics Cost/Error Rows}",
        _latex_table(
            [
                "regime",
                "track",
                "method",
                "status",
                "mean |dE|",
                "max |dE|",
                "final |dE|",
                "eps obs",
                "2q count",
                "2q depth",
                "depth",
            ],
            _cost_rows(rows) or [["--"] * 11],
            small=True,
        ),
    ]
    if plot_paths:
        body.append(r"\clearpage")
        body.append(r"\section*{Trajectory Panels}")
        for regime, path in sorted(plot_paths.items()):
            rel = path.relative_to(output_dir)
            body.extend(
                [
                    rf"\subsection*{{{_latex_escape(regime)}}}",
                    r"\begin{figure}[H]",
                    r"\centering",
                    rf"\includegraphics[width=0.96\linewidth]{{{_latex_escape(str(rel))}}}",
                    r"\end{figure}",
                ]
            )
    body.append(r"\section*{Run Notes}")
    note_rows = []
    for row in rows:
        if row.status == "completed":
            continue
        note_rows.append([row.regime_id, METHOD_LABELS.get(row.algorithm_id, row.algorithm_id), row.status, row.reason])
    body.append(_latex_table(["regime", "method", "status", "reason"], note_rows or [["--", "--", "--", "no incomplete rows"]]))
    body.extend([r"\end{document}", ""])
    return "\n".join(body)


def _compile_latex(tex_path: Path) -> Path:
    if shutil.which("pdflatex"):
        cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
        subprocess.run(cmd, cwd=tex_path.parent, check=True)
        subprocess.run(cmd, cwd=tex_path.parent, check=True)
        return tex_path.with_suffix(".pdf")
    if shutil.which("tectonic"):
        subprocess.run(["tectonic", tex_path.name], cwd=tex_path.parent, check=True)
        return tex_path.with_suffix(".pdf")
    raise RuntimeError("No LaTeX engine available; expected pdflatex or tectonic.")


def build_report(*, records_roots: Sequence[Path], output_dir: Path, compile_pdf: bool) -> dict[str, Any]:
    if not records_roots:
        raise ValueError("At least one records root is required.")
    rows, omitted_rows = load_rows_from_roots(records_roots)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    by_regime: dict[str, list[LoadedRow]] = defaultdict(list)
    for row in rows:
        by_regime[row.regime_id].append(row)
    plot_paths: dict[str, Path] = {}
    for regime, regime_rows in sorted(by_regime.items()):
        if any(row.trajectory is not None for row in regime_rows):
            path = plot_dir / f"{regime}_energy_error.png"
            _plot_regime(regime, regime_rows, path)
            plot_paths[regime] = path
    summary = {
        "schema": "paper_ii_hh_local_dynamics_review_v1",
        "records_roots": [str(path) for path in records_roots],
        "output_dir": str(output_dir),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status_counts": _status_counts(rows),
        "row_count": len(rows),
        "omitted_duplicate_row_count": len(omitted_rows),
        "completed_count": sum(1 for row in rows if row.status == "completed"),
        "plot_paths": {key: str(value) for key, value in sorted(plot_paths.items())},
        "rows": [_row_summary(row) for row in rows],
        "omitted_duplicate_rows": [_row_summary(row) for row in omitted_rows],
    }
    summary_json = output_dir / "paper_ii_hh_local_dynamics_review_summary.json"
    _write_json(summary_json, summary)
    tex_path = output_dir / "paper_ii_hh_local_dynamics_review.tex"
    tex_path.write_text(
        _build_tex(
            records_roots=records_roots,
            output_dir=output_dir,
            rows=rows,
            omitted_rows=omitted_rows,
            plot_paths=plot_paths,
            summary_json=summary_json,
        ),
        encoding="utf-8",
    )
    pdf_path = None
    compile_status = "skipped"
    if compile_pdf:
        try:
            pdf_path = _compile_latex(tex_path)
            compile_status = "compiled"
        except Exception as exc:
            compile_status = f"failed: {exc}"
    summary["tex_path"] = str(tex_path)
    summary["pdf_path"] = None if pdf_path is None else str(pdf_path)
    summary["compile_status"] = compile_status
    _write_json(summary_json, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-root", type=Path, required=True, action="append")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-compile", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records_roots = [path.expanduser().resolve() for path in args.records_root]
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else records_roots[0] / "paper_ii_hh_local_dynamics_review"
    )
    summary = build_report(records_roots=records_roots, output_dir=output_dir, compile_pdf=not args.no_compile)
    print(f"summary_json={summary.get('output_dir')}/paper_ii_hh_local_dynamics_review_summary.json")
    print(f"tex_path={summary.get('tex_path')}")
    print(f"pdf_path={summary.get('pdf_path')}")
    print(f"compile_status={summary.get('compile_status')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
