"""Render the first two required ADAPT scaffold plot views."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from docs.reports.pdf_utils import current_command_string, get_plt, require_matplotlib
from plots.adapt_scaffold.artifact_reader import AdaptPlotArtifact


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _jsonable_cell(value: Any) -> Any:
    if isinstance(value, tuple):
        return json.dumps(list(value))
    if isinstance(value, list):
        return json.dumps(value)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _ensure_parent(path)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable_cell(value) for key, value in row.items()})


def build_energy_trace_rows(artifact: AdaptPlotArtifact) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in artifact.history_steps:
        abs_delta_after = (
            abs(float(step.energy_after_opt) - float(artifact.exact_gs_energy))
            if artifact.exact_gs_energy is not None
            else None
        )
        rows.append(
            {
                "step_index": int(step.step_index),
                "selected_labels": list(step.selected_labels),
                "batch_size": int(step.batch_size),
                "selection_mode": step.selection_mode,
                "energy_before_opt": float(step.energy_before_opt),
                "energy_after_opt": float(step.energy_after_opt),
                "delta_energy": float(step.delta_energy),
                "delta_abs_drop_from_prev": (
                    None
                    if step.delta_abs_drop_from_prev is None
                    else float(step.delta_abs_drop_from_prev)
                ),
                "exact_gs_energy": artifact.exact_gs_energy,
                "abs_delta_after": abs_delta_after,
            }
        )
    return rows


def write_energy_vs_iteration_outputs(
    artifact: AdaptPlotArtifact,
    *,
    output_png: Path,
    output_csv: Path,
) -> dict[str, Any]:
    rows = build_energy_trace_rows(artifact)
    _write_csv(output_csv, rows)

    require_matplotlib()
    plt = get_plt()
    steps = np.asarray([row["step_index"] for row in rows], dtype=float)
    energy_before = np.asarray([row["energy_before_opt"] for row in rows], dtype=float)
    energy_after = np.asarray([row["energy_after_opt"] for row in rows], dtype=float)
    if not np.all(np.isfinite(energy_before)) or not np.all(np.isfinite(energy_after)):
        raise ValueError("Energy trace contains non-finite values.")

    _ensure_parent(output_png)
    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    ax.plot(
        steps,
        energy_before,
        linestyle="--",
        marker="o",
        color="0.65",
        linewidth=1.3,
        label="Before optimize",
    )
    ax.plot(
        steps,
        energy_after,
        linestyle="-",
        marker="o",
        color="tab:blue",
        linewidth=2.0,
        label="After optimize",
    )
    for step, before, after, row in zip(steps, energy_before, energy_after, rows):
        color = "tab:green" if after <= before else "tab:red"
        ax.vlines(step, before, after, colors=color, linewidth=2.0, alpha=0.9)
        if int(row["batch_size"]) > 1:
            ax.annotate(
                f"+{int(row['batch_size'])}",
                xy=(step, after),
                xytext=(0, 8),
                textcoords="offset points",
                fontsize=8,
                ha="center",
                color="0.25",
            )
    if artifact.exact_gs_energy is not None:
        ax.axhline(
            float(artifact.exact_gs_energy),
            linestyle=":",
            linewidth=1.5,
            color="tab:orange",
            label="Exact GS",
        )
    ax.set_title("ADAPT energy vs generator iteration")
    ax.set_xlabel("ADAPT selection step")
    ax.set_ylabel("Energy")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.text(
        0.01,
        0.01,
        current_command_string(),
        ha="left",
        va="bottom",
        fontsize=7,
        color="0.35",
        family="monospace",
    )
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "status": "written",
        "path_png": str(output_png),
        "path_csv": str(output_csv),
        "row_count": int(len(rows)),
    }


def _step_usefulness(
    *,
    energy_before_opt: float,
    energy_after_opt: float,
    delta_abs_drop_from_prev: float | None,
    exact_gs_energy: float | None,
) -> tuple[str, float]:
    if delta_abs_drop_from_prev is not None and np.isfinite(delta_abs_drop_from_prev):
        return "delta_abs_drop_from_prev", float(delta_abs_drop_from_prev)
    if exact_gs_energy is not None:
        return (
            "exact_abs_error_improvement",
            float(abs(energy_before_opt - exact_gs_energy) - abs(energy_after_opt - exact_gs_energy)),
        )
    return "raw_energy_drop", float(energy_before_opt - energy_after_opt)


def build_generator_usefulness_rows(artifact: AdaptPlotArtifact) -> list[dict[str, Any]]:
    step_by_index = {step.step_index: step for step in artifact.history_steps}
    survivor_map = {
        int(record.admission_order_index): record
        for record in artifact.final_scaffold_records
        if record.admission_order_index is not None
    }
    rows: list[dict[str, Any]] = []
    for admission in artifact.admission_records:
        step = step_by_index.get(admission.step_index)
        if step is None:
            raise ValueError(f"Admission record references missing history step {admission.step_index}.")
        usefulness_metric, step_usefulness = _step_usefulness(
            energy_before_opt=float(step.energy_before_opt),
            energy_after_opt=float(step.energy_after_opt),
            delta_abs_drop_from_prev=step.delta_abs_drop_from_prev,
            exact_gs_energy=artifact.exact_gs_energy,
        )
        batch_size = max(int(step.batch_size), 1)
        final_record = survivor_map.get(admission.admission_order_index)
        rows.append(
            {
                "rank": None,
                "admission_order_index": int(admission.admission_order_index),
                "step_index": int(admission.step_index),
                "record_index": int(admission.record_index),
                "generator_label": str(admission.generator_label),
                "generator_id": admission.generator_id,
                "template_id": admission.template_id,
                "family_id": admission.family_id,
                "selection_mode": step.selection_mode,
                "batch_size": int(batch_size),
                "usefulness_metric": usefulness_metric,
                "step_usefulness": float(step_usefulness),
                "allocated_usefulness": float(step_usefulness / batch_size),
                "step_delta_energy": float(step.delta_energy),
                "allocated_delta_energy": float(step.delta_energy / batch_size),
                "survives_in_final_scaffold": (
                    (
                        None
                        if (not artifact.final_matching_complete and final_record is None)
                        else (final_record is not None)
                    )
                    if artifact.final_scaffold_records
                    else None
                ),
                "final_scaffold_order_index": (
                    None if final_record is None else int(final_record.final_order_index)
                ),
                "runtime_split_mode": admission.runtime_split_mode,
                "support_sites": list(admission.support_sites),
                "support_qubits": list(admission.support_qubits),
                "num_polynomial_terms": admission.num_polynomial_terms,
                "has_boson_support": admission.has_boson_support,
                "has_fermion_support": admission.has_fermion_support,
            }
        )

    rows.sort(
        key=lambda row: (
            -float(row["allocated_usefulness"]),
            int(row["step_index"]),
            int(row["record_index"]),
            str(row["generator_label"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = int(rank)
    return rows


def write_generator_usefulness_outputs(
    artifact: AdaptPlotArtifact,
    *,
    output_png: Path,
    output_csv: Path,
    top_k: int = 30,
) -> dict[str, Any]:
    rows = build_generator_usefulness_rows(artifact)
    _write_csv(output_csv, rows)
    shown = rows[: max(int(top_k), 1)]

    require_matplotlib()
    plt = get_plt()
    _ensure_parent(output_png)

    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    for row in reversed(shown):
        labels.append(f"s{row['step_index']}r{row['record_index']} | {str(row['generator_label'])[:56]}")
        values.append(float(row["allocated_usefulness"]))
        survivor = row["survives_in_final_scaffold"]
        positive = float(row["allocated_usefulness"]) >= 0.0
        if survivor is True:
            colors.append("tab:blue" if positive else "tab:red")
        elif survivor is False:
            colors.append("#89a7d8" if positive else "#e6a19a")
        else:
            colors.append("#6f8fb8" if positive else "#d78780")

    fig_height = max(4.0, 0.4 * len(shown) + 1.5)
    fig, ax = plt.subplots(figsize=(12.0, fig_height))
    ax.barh(np.arange(len(shown)), values, color=colors, edgecolor="black", linewidth=0.4)
    ax.axvline(0.0, color="0.35", linewidth=1.0)
    ax.set_yticks(np.arange(len(shown)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Allocated usefulness")
    ax.set_title("Generator usefulness ranking")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "status": "written",
        "path_png": str(output_png),
        "path_csv": str(output_csv),
        "row_count": int(len(rows)),
        "top_k": int(len(shown)),
        "ranking_metric": "delta_abs_drop_from_prev -> exact_abs_error_improvement -> raw_energy_drop",
    }


__all__ = [
    "build_energy_trace_rows",
    "build_generator_usefulness_rows",
    "write_energy_vs_iteration_outputs",
    "write_generator_usefulness_outputs",
]
