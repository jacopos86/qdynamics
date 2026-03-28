#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from docs.reports.pdf_utils import (
    current_command_string,
    get_PdfPages,
    get_plt,
    render_command_page,
    render_compact_table,
    render_parameter_manifest,
    render_text_page,
    require_matplotlib,
)
from pipelines.exact_bench.nisq_dynamics_pareto import (
    build_hh_dynamics_hamiltonian,
    evolve_exact,
    evolve_suzuki2,
    prune_hamiltonian_terms,
    reorder_terms_by_qubit_locality,
    reorder_terms_by_weight,
)
from src.quantum.drives_time_potential import build_gaussian_sinusoid_density_drive
from src.quantum.hartree_fock_reference_state import hubbard_holstein_reference_state
from src.quantum.pauli_actions import compile_pauli_action_exyz


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CLAUDE_DRIVEN_T1 = Path("artifacts/json/nisq_dynamics_pareto_L2_driven_t1p0_final_20260325.json")
_DEFAULT_CLAUDE_DRIVEN_T2 = Path("artifacts/json/nisq_dynamics_pareto_L2_driven_t2p0_final_20260325.json")
_DEFAULT_OURS_PROGRESS = Path("artifacts/agent_runs/20260326_hh_l2_driven_realtime_pareto_sweep/progress.json")
_DEFAULT_OUTPUT_PDF = Path("artifacts/reports/hh_driven_dynamics_comparison_20260326.pdf")


"payload = json(path)"
def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object at {path}, got {type(payload).__name__}.")
    return payload


"resolved = repo_root / relative_path"
def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (_REPO_ROOT / candidate).resolve()


"series = [row[section][key]]_row"
def _trajectory_array(payload: Mapping[str, Any], section: str, key: str) -> np.ndarray:
    trajectory = payload.get("trajectory", [])
    if not isinstance(trajectory, list):
        raise TypeError("trajectory must be a list.")
    values: list[float] = []
    for row in trajectory:
        if not isinstance(row, Mapping):
            raise TypeError("trajectory rows must be mappings.")
        block = row.get(section, {})
        if not isinstance(block, Mapping):
            raise TypeError(f"trajectory.{section} rows must be mappings.")
        values.append(float(block.get(key, float("nan"))))
    return np.asarray(values, dtype=float)


"times = [row['time']]_row"
def _time_array(payload: Mapping[str, Any]) -> np.ndarray:
    trajectory = payload.get("trajectory", [])
    if not isinstance(trajectory, list):
        raise TypeError("trajectory must be a list.")
    return np.asarray([float(row.get("time", float("nan"))) for row in trajectory], dtype=float)


"result* = argmax(metric(result))"
def _best_result_under_hw_limit(payload: Mapping[str, Any], *, hw_limit: int = 100) -> dict[str, Any]:
    rows = [row for row in payload.get("results", []) if isinstance(row, Mapping) and not bool(row.get("skipped", False))]
    eligible: list[dict[str, Any]] = []
    for row in rows:
        transpile = row.get("transpile", {})
        if not isinstance(transpile, Mapping):
            continue
        fake = transpile.get("fake_marrakesh", {})
        if not isinstance(fake, Mapping):
            continue
        hw_2q = fake.get("compiled_count_2q")
        if hw_2q is None:
            continue
        if int(hw_2q) <= int(hw_limit):
            eligible.append(dict(row))
    if not eligible:
        raise ValueError("No eligible Claude results under the requested HW 2Q limit.")
    return max(eligible, key=lambda row: float(row.get("fidelity_min", float("-inf"))))


"result = select(ordering, steps, prune)"
def _find_result(
    payload: Mapping[str, Any],
    *,
    ordering: str,
    trotter_steps: int,
    prune_threshold: float,
) -> dict[str, Any]:
    for row in payload.get("results", []):
        if not isinstance(row, Mapping):
            continue
        if bool(row.get("skipped", False)):
            continue
        if str(row.get("ordering")) != str(ordering):
            continue
        if int(row.get("trotter_steps", -1)) != int(trotter_steps):
            continue
        if abs(float(row.get("prune_threshold", float("nan"))) - float(prune_threshold)) > 1.0e-12:
            continue
        return dict(row)
    raise KeyError(
        f"Could not find Claude result ordering={ordering!r}, trotter_steps={trotter_steps}, prune={prune_threshold}."
    )


"hw_2q = transpile.fake_marrakesh.compiled_count_2q"
def _hw_2q(row: Mapping[str, Any]) -> int | None:
    transpile = row.get("transpile", {})
    if not isinstance(transpile, Mapping):
        return None
    fake = transpile.get("fake_marrakesh", {})
    if not isinstance(fake, Mapping):
        return None
    value = fake.get("compiled_count_2q")
    return None if value is None else int(value)


"depth = transpile.fake_marrakesh.depth"
def _hw_depth(row: Mapping[str, Any]) -> int | None:
    transpile = row.get("transpile", {})
    if not isinstance(transpile, Mapping):
        return None
    fake = transpile.get("fake_marrakesh", {})
    if not isinstance(fake, Mapping):
        return None
    value = fake.get("depth")
    return None if value is None else int(value)


@dataclass(frozen=True)
class ClaudeCondition:
    label: str
    payload: dict[str, Any]
    best_under_100: dict[str, Any]
    best_weight_sorted_3step: dict[str, Any]
    trace_variants: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class OursRun:
    label: str
    payload: dict[str, Any]
    times: np.ndarray
    fidelity: np.ndarray
    abs_energy_error: np.ndarray
    energy_ansatz: np.ndarray
    energy_reference: np.ndarray
    condition_number: np.ndarray
    summary: dict[str, Any]
    loader: dict[str, Any]


"condition = load_claude_condition(path)"
def _load_claude_condition(path: Path, label: str) -> ClaudeCondition:
    payload = _read_json(path)
    best_under_100 = _best_result_under_hw_limit(payload, hw_limit=100)
    trace_variants = {
        "weight_sorted_2step_full": _find_result(payload, ordering="weight_sorted", trotter_steps=2, prune_threshold=0.0),
        "weight_sorted_3step_full": _find_result(payload, ordering="weight_sorted", trotter_steps=3, prune_threshold=0.0),
        "weight_sorted_2step_pruned": _find_result(payload, ordering="weight_sorted", trotter_steps=2, prune_threshold=0.3),
        "native_2step_full": _find_result(payload, ordering="native", trotter_steps=2, prune_threshold=0.0),
    }
    return ClaudeCondition(
        label=str(label),
        payload=payload,
        best_under_100=best_under_100,
        best_weight_sorted_3step=trace_variants["weight_sorted_3step_full"],
        trace_variants=trace_variants,
    )


"run = load_our_run(path)"
def _load_our_run(path: Path, label: str) -> OursRun:
    payload = _read_json(path)
    return OursRun(
        label=str(label),
        payload=payload,
        times=_time_array(payload),
        fidelity=_trajectory_array(payload, "audit", "fidelity_exact_audit"),
        abs_energy_error=_trajectory_array(payload, "audit", "abs_energy_total_error_exact_audit"),
        energy_ansatz=_trajectory_array(payload, "audit", "energy_ansatz_exact_audit"),
        energy_reference=_trajectory_array(payload, "audit", "energy_reference_exact_audit"),
        condition_number=_trajectory_array(payload, "geometry", "condition_number"),
        summary=dict(payload.get("summary", {})),
        loader=dict(payload.get("loader", {})),
    )


"provider = drive.coeff_map_exyz or None"
def _claude_drive_provider(payload: Mapping[str, Any], nq: int) -> Any | None:
    settings = payload.get("settings", {})
    if not isinstance(settings, Mapping):
        return None
    if not bool(settings.get("drive_enabled", False)):
        return None
    drive_obj = build_gaussian_sinusoid_density_drive(
        n_sites=int(settings.get("L", 2)),
        nq_total=int(nq),
        indexing="blocked",
        A=float(settings.get("drive_amplitude", 0.3)),
        omega=float(settings.get("drive_omega", 2.0)),
        tbar=float(settings.get("drive_tbar", 3.0)),
        pattern_mode="staggered",
    )
    return drive_obj.coeff_map_exyz


"labels, coeffs = prune_then_reorder(result_variant)"
def _ordered_labels_and_coeffs_for_claude_variant(
    payload: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    base_labels: Sequence[str],
    base_coeffs: Mapping[str, complex],
) -> tuple[list[str], dict[str, complex]]:
    prune_threshold = float(row.get("prune_threshold", 0.0))
    ordering = str(row.get("ordering", "native"))
    if prune_threshold > 0.0:
        labels_pruned, coeffs_pruned = prune_hamiltonian_terms(list(base_labels), dict(base_coeffs), prune_threshold)
    else:
        labels_pruned = list(base_labels)
        coeffs_pruned = dict(base_coeffs)
    if ordering == "weight_sorted":
        labels_ordered = reorder_terms_by_weight(labels_pruned)
    elif ordering == "qubit_local":
        labels_ordered = reorder_terms_by_qubit_locality(labels_pruned)
    else:
        labels_ordered = list(labels_pruned)
    return labels_ordered, coeffs_pruned


"times, exact_E, trotter_E = reconstruct_claude_energy_traces(condition, variant_keys=None)"
def _reconstruct_claude_energy_traces(
    condition: ClaudeCondition,
    variant_keys: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    payload = condition.payload
    settings = payload.get("settings", {})
    if not isinstance(settings, Mapping):
        raise TypeError("Claude settings block must be a mapping.")

    ham = build_hh_dynamics_hamiltonian(
        L=int(settings.get("L", 2)),
        t_hop=float(settings.get("t_hop", 1.0)),
        U=float(settings.get("U", 4.0)),
        omega0=float(settings.get("omega0", 1.0)),
        g=float(settings.get("g", 0.5)),
        n_ph_max=int(settings.get("n_ph_max", 1)),
        boundary=str(settings.get("boundary", "open")),
    )
    nq = int(ham["nq"])
    base_labels = list(ham["ordered_labels"])
    base_coeffs = dict(ham["coeff_map"])
    hmat = ham["hmat"]
    evals = ham["evals"]
    evecs = ham["evecs"]
    psi0 = hubbard_holstein_reference_state(
        dims=int(settings.get("L", 2)),
        n_ph_max=int(settings.get("n_ph_max", 1)),
        boson_encoding="binary",
        indexing="blocked",
    )
    drive_provider = _claude_drive_provider(payload, nq)
    times = np.linspace(0.0, float(settings.get("t_final", 0.0)), int(settings.get("num_times", 0)))

    exact_energies: list[float] = []
    for t_val in times:
        psi_exact = evolve_exact(
            psi0,
            hmat,
            float(t_val),
            evals,
            evecs,
            drive_provider=drive_provider,
            reference_steps=max(2000, 200 * int(float(t_val) + 1.0)),
        )
        exact_energies.append(float(np.real(np.vdot(psi_exact, hmat @ psi_exact))))

    selected_keys = list(condition.trace_variants.keys()) if variant_keys is None else [key for key in variant_keys if key in condition.trace_variants]

    variant_energies: dict[str, np.ndarray] = {}
    for key in selected_keys:
        row = condition.trace_variants[key]
        labels_ordered, coeffs_pruned = _ordered_labels_and_coeffs_for_claude_variant(
            payload,
            row,
            base_labels=base_labels,
            base_coeffs=base_coeffs,
        )
        compiled = {label: compile_pauli_action_exyz(label, nq) for label in labels_ordered}
        series: list[float] = []
        for t_val in times:
            psi_trot = evolve_suzuki2(
                psi0,
                labels_ordered,
                coeffs_pruned,
                compiled,
                float(t_val),
                int(row.get("trotter_steps", 1)),
                drive_provider=drive_provider,
            )
            series.append(float(np.real(np.vdot(psi_trot, hmat @ psi_trot))))
        variant_energies[key] = np.asarray(series, dtype=float)

    return times, np.asarray(exact_energies, dtype=float), variant_energies


"lines = executive_summary(claude, ours, progress)"
def _summary_lines(
    claude_t1: ClaudeCondition,
    claude_t2: ClaudeCondition,
    our_runs: Sequence[OursRun],
    progress: Mapping[str, Any],
) -> list[str]:
    frontier = progress.get("frontier_names", [])
    frontier_str = ", ".join(str(x) for x in frontier) if isinstance(frontier, list) and frontier else "(none)"
    locked = next(run for run in our_runs if run.label == "driven_locked7_short")
    pareto = next(run for run in our_runs if run.label == "driven_pareto_short")
    lines = [
        "Driven-only comparison summary",
        "",
        "Claude report (complete driven Pareto sweep):",
        (
            f"- Driven t=1.0 best <=100 HW 2Q: {claude_t1.best_under_100.get('variant_label')} | "
            f"HW 2Q={_hw_2q(claude_t1.best_under_100)} | depth={_hw_depth(claude_t1.best_under_100)} | "
            f"min fidelity={float(claude_t1.best_under_100.get('fidelity_min', float('nan'))):.4f} | "
            f"final |ΔE|={float(claude_t1.best_under_100.get('energy_error_final', float('nan'))):.4f}"
        ),
        (
            f"- Driven t=2.0 best <=100 HW 2Q: {claude_t2.best_under_100.get('variant_label')} | "
            f"HW 2Q={_hw_2q(claude_t2.best_under_100)} | depth={_hw_depth(claude_t2.best_under_100)} | "
            f"min fidelity={float(claude_t2.best_under_100.get('fidelity_min', float('nan'))):.4f} | "
            f"final |ΔE|={float(claude_t2.best_under_100.get('energy_error_final', float('nan'))):.4f}"
        ),
        (
            f"- Driven t=2.0 recovery point: {claude_t2.best_weight_sorted_3step.get('variant_label')} | "
            f"HW 2Q={_hw_2q(claude_t2.best_weight_sorted_3step)} | depth={_hw_depth(claude_t2.best_weight_sorted_3step)} | "
            f"min fidelity={float(claude_t2.best_weight_sorted_3step.get('fidelity_min', float('nan'))):.4f} | "
            f"final |ΔE|={float(claude_t2.best_weight_sorted_3step.get('energy_error_final', float('nan'))):.4f}"
        ),
        "",
        "Our report (current driven fixed-manifold McLachlan sweep, partial):",
        (
            f"- Progress: completed={int(progress.get('completed_runs', 0))} / failed={int(progress.get('failed_runs', 0))} / frontier={frontier_str}"
        ),
        (
            f"- {locked.label}: min fidelity={float(locked.summary.get('min_fidelity_exact_audit', float('nan'))):.6f} | "
            f"max |ΔE|={float(locked.summary.get('max_abs_energy_total_error_exact_audit', float('nan'))):.6f} | "
            f"runtime params={int(locked.summary.get('runtime_parameter_count', -1))} | "
            f"logical blocks={int(locked.summary.get('logical_block_count', -1))} | "
            f"max cond={float(locked.summary.get('max_condition_number', float('nan'))):.3e}"
        ),
        (
            f"- {pareto.label}: min fidelity={float(pareto.summary.get('min_fidelity_exact_audit', float('nan'))):.6f} | "
            f"max |ΔE|={float(pareto.summary.get('max_abs_energy_total_error_exact_audit', float('nan'))):.6f} | "
            f"runtime params={int(pareto.summary.get('runtime_parameter_count', -1))} | "
            f"logical blocks={int(pareto.summary.get('logical_block_count', -1))} | "
            f"max cond={float(pareto.summary.get('max_condition_number', float('nan'))):.3e}"
        ),
        "",
        "Read carefully:",
        "- Claude metrics are hardware-oriented (transpiled 2Q gates + depth on FakeMarrakesh).",
        "- Our current metrics are driven trajectory quality + geometry complexity; hardware transpilation is not yet attached.",
        "- Static Claude panels were intentionally omitted here; this report is driven-only per user direction.",
    ]
    return lines


"page = driven_comparison_table(claude, ours)"
def _render_comparison_tables(
    pdf: Any,
    *,
    claude_t1: ClaudeCondition,
    claude_t2: ClaudeCondition,
    our_runs: Sequence[OursRun],
) -> None:
    plt = get_plt()
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.5))

    claude_rows = [
        [
            "driven t=1",
            str(claude_t1.best_under_100.get("variant_label", "")),
            str(_hw_2q(claude_t1.best_under_100)),
            str(_hw_depth(claude_t1.best_under_100)),
            f"{float(claude_t1.best_under_100.get('fidelity_min', float('nan'))):.4f}",
            f"{float(claude_t1.best_under_100.get('energy_error_final', float('nan'))):.4f}",
        ],
        [
            "driven t=2",
            str(claude_t2.best_under_100.get("variant_label", "")),
            str(_hw_2q(claude_t2.best_under_100)),
            str(_hw_depth(claude_t2.best_under_100)),
            f"{float(claude_t2.best_under_100.get('fidelity_min', float('nan'))):.4f}",
            f"{float(claude_t2.best_under_100.get('energy_error_final', float('nan'))):.4f}",
        ],
        [
            "driven t=2 recovery",
            str(claude_t2.best_weight_sorted_3step.get("variant_label", "")),
            str(_hw_2q(claude_t2.best_weight_sorted_3step)),
            str(_hw_depth(claude_t2.best_weight_sorted_3step)),
            f"{float(claude_t2.best_weight_sorted_3step.get('fidelity_min', float('nan'))):.4f}",
            f"{float(claude_t2.best_weight_sorted_3step.get('energy_error_final', float('nan'))):.4f}",
        ],
    ]
    render_compact_table(
        axes[0],
        title="Claude driven Pareto highlights",
        col_labels=["Case", "Variant", "HW 2Q", "Depth", "Min fidelity", "Final |ΔE|"],
        rows=claude_rows,
        fontsize=7,
    )

    our_rows = []
    for run in our_runs:
        our_rows.append([
            run.label,
            str(int(run.summary.get("runtime_parameter_count", -1))),
            str(int(run.summary.get("logical_block_count", -1))),
            f"{float(run.summary.get('min_fidelity_exact_audit', float('nan'))):.6f}",
            f"{float(run.summary.get('max_abs_energy_total_error_exact_audit', float('nan'))):.6f}",
            f"{float(run.summary.get('max_condition_number', float('nan'))):.3e}",
        ])
    render_compact_table(
        axes[1],
        title="Our driven fixed-manifold McLachlan highlights",
        col_labels=["Run", "Runtime params", "Blocks", "Min fidelity", "Max |ΔE|", "Max cond"],
        rows=our_rows,
        fontsize=7,
    )

    fig.suptitle("Driven-only comparison tables", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


"fig = plot_claude_pareto(conditions)"
def _plot_claude_driven_pareto(claude_conditions: Sequence[ClaudeCondition], *, output_path: Path) -> None:
    plt = get_plt()
    fig, axes = plt.subplots(1, len(claude_conditions), figsize=(12, 4.8), sharey=True)
    if len(claude_conditions) == 1:
        axes = [axes]
    colors = {"native": "tab:red", "weight_sorted": "tab:green", "qubit_local": "tab:blue"}
    markers = {0.0: "o", 0.1: "s", 0.3: "^", 0.5: "D"}

    for ax, condition in zip(axes, claude_conditions):
        payload = condition.payload
        rows = [row for row in payload.get("results", []) if isinstance(row, Mapping) and not bool(row.get("skipped", False))]
        for row in rows:
            x = _hw_2q(row)
            if x is None:
                continue
            y = float(row.get("fidelity_min", float("nan")))
            ordering = str(row.get("ordering", "native"))
            prune = float(row.get("prune_threshold", 0.0))
            ax.scatter(
                x,
                y,
                color=colors.get(ordering, "black"),
                marker=markers.get(prune, "o"),
                s=42,
                alpha=0.8,
                edgecolors="black",
                linewidths=0.25,
            )
        pareto = [row for row in payload.get("pareto_front", []) if isinstance(row, Mapping)]
        pareto_hw = [(x, float(row.get("fidelity_min", float("nan")))) for row in pareto if (x := _hw_2q(row)) is not None]
        pareto_hw.sort(key=lambda item: item[0])
        if pareto_hw:
            ax.plot([p[0] for p in pareto_hw], [p[1] for p in pareto_hw], color="black", linewidth=1.4, alpha=0.7)
            ax.scatter([p[0] for p in pareto_hw], [p[1] for p in pareto_hw], color="gold", marker="*", s=95, edgecolors="black", linewidths=0.35)
        ax.axvline(100, color="red", linestyle="--", alpha=0.5)
        ax.axhline(0.9, color="orange", linestyle=":", alpha=0.6)
        ax.set_title(condition.label)
        ax.set_xlabel("HW 2Q gates (FakeMarrakesh)")
        ax.set_xlim(0, 430)
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Min fidelity vs exact")
    fig.suptitle("Claude driven Pareto front (driven-only panels)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


"fig = plot_claude_metric(metric)"
def _plot_claude_driven_traces(
    claude_conditions: Sequence[ClaudeCondition],
    *,
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
    logy: bool = False,
) -> None:
    plt = get_plt()
    fig, axes = plt.subplots(1, len(claude_conditions), figsize=(12, 4.8), sharey=False)
    if len(claude_conditions) == 1:
        axes = [axes]

    styles = {
        "weight_sorted_2step_full": ("tab:green", "-", "2-step weight-sorted full"),
        "weight_sorted_3step_full": ("tab:blue", "-", "3-step weight-sorted full"),
        "weight_sorted_2step_pruned": ("tab:purple", "--", "2-step weight-sorted prune=0.3"),
        "native_2step_full": ("tab:red", "--", "2-step native full"),
    }

    for ax, condition in zip(axes, claude_conditions):
        settings = condition.payload.get("settings", {})
        t_final = float(settings.get("t_final", 0.0))
        num_times = int(settings.get("num_times", 0))
        times = np.linspace(0.0, t_final, num_times)
        for key, row in condition.trace_variants.items():
            series = np.asarray(row.get(metric_key, []), dtype=float)
            if len(series) == 0:
                continue
            color, linestyle, label = styles[key]
            ax.plot(times, series, color=color, linestyle=linestyle, linewidth=1.6, label=label)
        if metric_key == "fidelity_trajectory":
            ax.axhline(0.98, color="green", linestyle=":", alpha=0.5)
            ax.axhline(0.9, color="orange", linestyle=":", alpha=0.5)
            ax.set_ylim(0.0, 1.02)
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(1.0e-8, 5.0)
        ax.set_title(condition.label)
        ax.set_xlabel("Time")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


"fig = plot_our_traces(runs)"
def _plot_our_driven_traces(runs: Sequence[OursRun], *, output_path: Path) -> None:
    plt = get_plt()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=False)
    colors = {
        "driven_locked7_short": "tab:blue",
        "driven_pareto_short": "tab:green",
    }

    ax = axes[0, 0]
    for run in runs:
        ax.plot(run.times, run.fidelity, linewidth=1.8, color=colors.get(run.label, "black"), label=run.label)
    ax.axhline(0.99, color="green", linestyle=":", alpha=0.5)
    ax.set_title("Our driven sweep: fidelity vs time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Fidelity vs exact")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[0, 1]
    for run in runs:
        ax.plot(run.times, run.abs_energy_error, linewidth=1.8, color=colors.get(run.label, "black"), label=run.label)
    ax.set_yscale("log")
    ax.set_title("Our driven sweep: |ΔE| vs time")
    ax.set_xlabel("Time")
    ax.set_ylabel("|ΔE| vs exact")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1, 0]
    for run in runs:
        color = colors.get(run.label, "black")
        ax.plot(run.times, run.energy_ansatz, linewidth=1.8, color=color, label=f"{run.label} ansatz")
        ax.plot(run.times, run.energy_reference, linewidth=1.2, color=color, linestyle="--", label=f"{run.label} exact")
    ax.set_title("Our driven sweep: total energy vs time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Total energy")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")

    ax = axes[1, 1]
    for run in runs:
        ax.plot(run.times, run.condition_number, linewidth=1.8, color=colors.get(run.label, "black"), label=run.label)
    ax.set_yscale("log")
    ax.set_title("Our driven sweep: condition number vs time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Condition number")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Our driven fixed-manifold McLachlan runs (completed so far)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


"fig = plot_claude_total_energy(condition)"
def _plot_claude_driven_total_energy(claude_conditions: Sequence[ClaudeCondition], *, output_path: Path) -> None:
    plt = get_plt()
    fig, axes = plt.subplots(1, len(claude_conditions), figsize=(12, 4.8), sharey=False)
    if len(claude_conditions) == 1:
        axes = [axes]

    styles = {
        "weight_sorted_2step_full": ("tab:green", "-", "2-step weight-sorted full"),
        "weight_sorted_3step_full": ("tab:blue", "-", "3-step weight-sorted full"),
        "weight_sorted_2step_pruned": ("tab:purple", "--", "2-step weight-sorted prune=0.3"),
        "native_2step_full": ("tab:red", "--", "2-step native full"),
    }

    for ax, condition in zip(axes, claude_conditions):
        times, exact_energies, variant_energies = _reconstruct_claude_energy_traces(condition)
        ax.plot(times, exact_energies, color="black", linestyle="-", linewidth=2.0, label="exact")
        for key, series in variant_energies.items():
            color, linestyle, label = styles[key]
            ax.plot(times, series, color=color, linestyle=linestyle, linewidth=1.6, label=label)
        ax.set_title(condition.label)
        ax.set_xlabel("Time")
        ax.set_ylabel("Total energy")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle("Claude driven trajectories: total energy vs time with exact overlay", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


"pdf += image_page(title, image)"
def _render_image_page(pdf: Any, *, image_path: Path, title: str, subtitle: str = "") -> None:
    plt = get_plt()
    img = plt.imread(str(image_path))
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_subplot(111)
    ax.imshow(img)
    ax.axis("off")
    fig.suptitle(title, fontsize=14, y=0.98)
    if subtitle:
        fig.text(0.5, 0.94, subtitle, ha="center", va="center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


"report = build_report(inputs)"
def build_report(
    *,
    claude_driven_t1_json: Path,
    claude_driven_t2_json: Path,
    our_progress_json: Path,
    output_pdf: Path,
) -> Path:
    require_matplotlib()
    plt = get_plt()
    PdfPages = get_PdfPages()

    claude_t1 = _load_claude_condition(claude_driven_t1_json, label="Claude driven t=1.0")
    claude_t2 = _load_claude_condition(claude_driven_t2_json, label="Claude driven t=2.0")
    progress = _read_json(our_progress_json)
    our_runs = [
        _load_our_run(_resolve_path(run["output_json"]), label=str(run["name"]))
        for run in progress.get("runs", [])
        if isinstance(run, Mapping) and str(run.get("status", "")) == "completed"
    ]
    our_runs.sort(key=lambda run: run.label)

    if not our_runs:
        raise ValueError("No completed driven sweep runs found in progress.json.")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    asset_dir = output_pdf.parent
    claude_pareto_png = asset_dir / "hh_driven_compare_claude_pareto.png"
    claude_fidelity_png = asset_dir / "hh_driven_compare_claude_fidelity.png"
    claude_deltae_png = asset_dir / "hh_driven_compare_claude_deltae.png"
    claude_total_energy_png = asset_dir / "hh_driven_compare_claude_total_energy.png"
    our_traces_png = asset_dir / "hh_driven_compare_our_traces.png"
    our_existing_summary_png = _resolve_path("artifacts/reports/20260326_hh_l2_driven_realtime_pareto_sweep.png")

    _plot_claude_driven_pareto([claude_t1, claude_t2], output_path=claude_pareto_png)
    _plot_claude_driven_traces(
        [claude_t1, claude_t2],
        metric_key="fidelity_trajectory",
        ylabel="Fidelity vs exact",
        title="Claude driven trajectories: fidelity vs time",
        output_path=claude_fidelity_png,
        logy=False,
    )
    _plot_claude_driven_traces(
        [claude_t1, claude_t2],
        metric_key="energy_error_trajectory",
        ylabel="|ΔE| vs exact",
        title="Claude driven trajectories: |ΔE| vs time",
        output_path=claude_deltae_png,
        logy=True,
    )
    _plot_claude_driven_total_energy([claude_t1, claude_t2], output_path=claude_total_energy_png)
    _plot_our_driven_traces(our_runs, output_path=our_traces_png)

    manifest_extra = {
        "L": 2,
        "omega0": 1.0,
        "g_ep": 0.5,
        "n_ph_max": 1,
        "Claude drive": "A=0.3, omega=2.0, tbar=3.0",
        "Our drive": "A=0.6, omega=1.0, tbar=1.0, pattern=staggered",
        "Compared methods": "Claude Suzuki-2 Trotter Pareto vs our fixed-manifold McLachlan",
        "Our sweep status": f"{int(progress.get('completed_runs', 0))} completed, {int(progress.get('failed_runs', 0))} failed",
    }

    with PdfPages(str(output_pdf)) as pdf:
        render_parameter_manifest(
            pdf,
            model="Hubbard-Holstein",
            ansatz="Claude: Suzuki-2 Trotter sweep; Ours: fixed-manifold McLachlan (locked7 + pareto)",
            drive_enabled=True,
            t=1.0,
            U=4.0,
            dv=0.0,
            extra=manifest_extra,
            command=current_command_string(),
        )
        render_text_page(pdf, _summary_lines(claude_t1, claude_t2, our_runs, progress), fontsize=9, line_spacing=0.03)
        _render_comparison_tables(pdf, claude_t1=claude_t1, claude_t2=claude_t2, our_runs=our_runs)
        _render_image_page(
            pdf,
            image_path=claude_pareto_png,
            title="Claude driven Pareto front",
            subtitle="Driven-only view: HW 2Q gate cost vs min fidelity on FakeMarrakesh.",
        )
        _render_image_page(
            pdf,
            image_path=claude_fidelity_png,
            title="Claude driven fidelity traces",
            subtitle="Weight-sorted 2-step is the best <=100-HW-2Q driven point at t=1; 3-step recovers t=2.",
        )
        _render_image_page(
            pdf,
            image_path=claude_deltae_png,
            title="Claude driven |ΔE| traces",
            subtitle="Driven energy-error growth is what pushes the t=2 frontier from 2-step to 3-step.",
        )
        _render_image_page(
            pdf,
            image_path=claude_total_energy_png,
            title="Claude driven total energy traces",
            subtitle="Exact overlay added; plotted with the same energy convention used in Claude's sweep JSON generation.",
        )
        _render_image_page(
            pdf,
            image_path=our_traces_png,
            title="Our driven fixed-manifold McLachlan traces",
            subtitle="Includes total energy vs time with exact overlay; both completed runs stay above 0.9919 min fidelity through t=10.",
        )
        if our_existing_summary_png.exists():
            _render_image_page(
                pdf,
                image_path=our_existing_summary_png,
                title="Our current driven sweep frontier",
                subtitle="Only two completed points so far; locked7 is leanest, pareto is slightly higher fidelity.",
            )
        render_command_page(
            pdf,
            current_command_string(),
            script_name="pipelines/hardcoded/hh_driven_dynamics_comparison_report.py",
            extra_header_lines=[
                f"Claude driven t=1 JSON: {claude_driven_t1_json}",
                f"Claude driven t=2 JSON: {claude_driven_t2_json}",
                f"Our progress JSON: {our_progress_json}",
            ],
        )

    plt.close("all")
    return output_pdf


"args = parse_cli(argv)"
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a driven-only PDF comparison of Claude's Pareto report and our current sweep.")
    parser.add_argument("--claude-driven-t1-json", type=Path, default=_DEFAULT_CLAUDE_DRIVEN_T1)
    parser.add_argument("--claude-driven-t2-json", type=Path, default=_DEFAULT_CLAUDE_DRIVEN_T2)
    parser.add_argument("--our-progress-json", type=Path, default=_DEFAULT_OURS_PROGRESS)
    parser.add_argument("--output-pdf", type=Path, default=_DEFAULT_OUTPUT_PDF)
    return parser.parse_args(argv)


"exit_code = main(argv)"
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_pdf = build_report(
        claude_driven_t1_json=_resolve_path(args.claude_driven_t1_json),
        claude_driven_t2_json=_resolve_path(args.claude_driven_t2_json),
        our_progress_json=_resolve_path(args.our_progress_json),
        output_pdf=_resolve_path(args.output_pdf),
    )
    print(f"Wrote PDF: {output_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
