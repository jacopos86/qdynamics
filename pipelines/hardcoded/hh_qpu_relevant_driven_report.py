#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from pipelines.hardcoded.hh_driven_dynamics_comparison_report import (
    _hw_2q,
    _hw_depth,
    _load_claude_condition,
    _load_our_run,
    _read_json,
    _reconstruct_claude_energy_traces,
    _resolve_path,
)


_DEFAULT_CLAUDE_DRIVEN_T1 = Path("artifacts/json/nisq_dynamics_pareto_L2_driven_t1p0_final_20260325.json")
_DEFAULT_CLAUDE_DRIVEN_T2 = Path("artifacts/json/nisq_dynamics_pareto_L2_driven_t2p0_final_20260325.json")
_DEFAULT_OURS_PROGRESS = Path("artifacts/agent_runs/20260326_hh_l2_driven_realtime_pareto_sweep/progress.json")
_DEFAULT_COMPILE_LOCKED = Path("artifacts/json/compile_scout_locked7_marrakesh.json")
_DEFAULT_COMPILE_PARETO = Path("artifacts/json/compile_scout_pareto_marrakesh.json")
_DEFAULT_OUTPUT = Path("artifacts/reports/hh_qpu_relevant_driven_dynamics_20260326.pdf")


@dataclass(frozen=True)
class ShortlistCandidate:
    label: str
    family: str
    method: str
    horizon: str
    hw_2q: int
    depth: int
    fidelity_min: float
    delta_max: float
    note: str


"backend = compile_scout.selected_backend"
def _selected_backend(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    block = payload.get("selected_backend", {})
    if not isinstance(block, Mapping):
        raise TypeError("selected_backend must be a mapping.")
    return block


"candidate* = shortlist(claude, ours, compile_scouts)"
def _build_shortlist(
    *,
    claude_t1: Any,
    claude_t2: Any,
    our_runs_by_name: Mapping[str, Any],
    compile_locked: Mapping[str, Any],
    compile_pareto: Mapping[str, Any],
) -> list[ShortlistCandidate]:
    c_t1_2 = claude_t1.trace_variants["weight_sorted_2step_full"]
    c_t1_3 = claude_t1.trace_variants["weight_sorted_3step_full"]
    c_t2_3 = claude_t2.trace_variants["weight_sorted_3step_full"]

    run_locked = our_runs_by_name["driven_locked7_short"]
    run_pareto = our_runs_by_name["driven_pareto_short"]
    backend_locked = _selected_backend(compile_locked)
    backend_pareto = _selected_backend(compile_pareto)

    return [
        ShortlistCandidate(
            label="Claude Trotter t=1 2-step",
            family="claude",
            method="Suzuki-2 Trotter",
            horizon="driven, t=1.0",
            hw_2q=int(_hw_2q(c_t1_2) or -1),
            depth=int(_hw_depth(c_t1_2) or -1),
            fidelity_min=float(c_t1_2.get("fidelity_min", float("nan"))),
            delta_max=float(c_t1_2.get("energy_error_max", float("nan"))),
            note="best <=100 HW 2Q Claude point",
        ),
        ShortlistCandidate(
            label="Claude Trotter t=1 3-step",
            family="claude",
            method="Suzuki-2 Trotter",
            horizon="driven, t=1.0",
            hw_2q=int(_hw_2q(c_t1_3) or -1),
            depth=int(_hw_depth(c_t1_3) or -1),
            fidelity_min=float(c_t1_3.get("fidelity_min", float("nan"))),
            delta_max=float(c_t1_3.get("energy_error_max", float("nan"))),
            note="higher-accuracy Claude option",
        ),
        ShortlistCandidate(
            label="Claude Trotter t=2 3-step",
            family="claude",
            method="Suzuki-2 Trotter",
            horizon="driven, t=2.0",
            hw_2q=int(_hw_2q(c_t2_3) or -1),
            depth=int(_hw_depth(c_t2_3) or -1),
            fidelity_min=float(c_t2_3.get("fidelity_min", float("nan"))),
            delta_max=float(c_t2_3.get("energy_error_max", float("nan"))),
            note="longer-horizon Claude point",
        ),
        ShortlistCandidate(
            label="Our McLachlan locked7",
            family="ours",
            method="fixed-manifold McLachlan",
            horizon="driven, t=10 short run",
            hw_2q=int(backend_locked.get("compiled_count_2q", -1)),
            depth=int(backend_locked.get("compiled_depth", -1)),
            fidelity_min=float(run_locked.summary.get("min_fidelity_exact_audit", float("nan"))),
            delta_max=float(run_locked.summary.get("max_abs_energy_total_error_exact_audit", float("nan"))),
            note=f"per-shot prep; oracle evals={int(run_locked.summary.get('oracle_evaluations_total', -1))}",
        ),
        ShortlistCandidate(
            label="Our McLachlan pareto",
            family="ours",
            method="fixed-manifold McLachlan",
            horizon="driven, t=10 short run",
            hw_2q=int(backend_pareto.get("compiled_count_2q", -1)),
            depth=int(backend_pareto.get("compiled_depth", -1)),
            fidelity_min=float(run_pareto.summary.get("min_fidelity_exact_audit", float("nan"))),
            delta_max=float(run_pareto.summary.get("max_abs_energy_total_error_exact_audit", float("nan"))),
            note=f"per-shot prep; oracle evals={int(run_pareto.summary.get('oracle_evaluations_total', -1))}",
        ),
    ]


"rows = shortlist_table(candidates)"
def _shortlist_rows(candidates: Sequence[ShortlistCandidate]) -> list[list[str]]:
    rows: list[list[str]] = []
    for c in candidates:
        rows.append([
            c.label,
            c.method,
            c.horizon,
            str(c.hw_2q),
            str(c.depth),
            f"{c.fidelity_min:.4f}",
            f"{c.delta_max:.4f}",
            c.note,
        ])
    return rows


"lines = shortlist_summary(candidates)"
def _summary_lines(candidates: Sequence[ShortlistCandidate]) -> list[str]:
    best_low_cost = min(candidates, key=lambda c: (c.hw_2q, -c.fidelity_min, c.delta_max))
    best_accuracy = max(candidates, key=lambda c: (c.fidelity_min, -c.hw_2q))
    lines = [
        "QPU-relevant driven time-dynamics shortlist",
        "",
        "Filter used for this PDF:",
        "- driven-only",
        "- fidelity roughly 0.9+ or otherwise clearly near the useful frontier",
        "- |ΔE| roughly 1e-1 or 1e-2 scale",
        "- 2Q cost near 100 when possible; slightly above retained only if accuracy gain is material",
        "",
        f"Lowest-cost shortlisted point: {best_low_cost.label} | 2Q={best_low_cost.hw_2q} | depth={best_low_cost.depth} | min fidelity={best_low_cost.fidelity_min:.4f} | max |ΔE|={best_low_cost.delta_max:.4f}",
        f"Highest-fidelity shortlisted point: {best_accuracy.label} | 2Q={best_accuracy.hw_2q} | depth={best_accuracy.depth} | min fidelity={best_accuracy.fidelity_min:.4f} | max |ΔE|={best_accuracy.delta_max:.4f}",
        "",
        "How to read this report:",
        "- Claude entries are direct hardware-costed driven Trotter candidates on FakeMarrakesh.",
        "- Our McLachlan entries use compiled state-prep cost from the imported ansatz family plus exact-audit trajectory quality from the driven sweep.",
        "- For our entries, low gate cost does not mean low wall-clock workload: oracle evaluation count is also shown because measurement/controller overhead matters.",
    ]
    return lines


"fig = shortlist_scatter(candidates)"
def _render_shortlist_scatter(pdf: Any, candidates: Sequence[ShortlistCandidate]) -> None:
    plt = get_plt()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    colors = {"claude": "tab:green", "ours": "tab:blue"}

    ax = axes[0]
    ax.axvspan(0, 110, color="#dff0d8", alpha=0.4)
    ax.axhspan(0.9, 1.02, color="#d9edf7", alpha=0.35)
    ax.axvline(100, color="red", linestyle="--", alpha=0.6)
    ax.axhline(0.9, color="orange", linestyle=":", alpha=0.7)
    for c in candidates:
        ax.scatter(c.hw_2q, c.fidelity_min, s=90, color=colors.get(c.family, "black"), edgecolors="black", linewidths=0.4)
        ax.annotate(c.label, (c.hw_2q, c.fidelity_min), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_title("Shortlist: 2Q cost vs min fidelity")
    ax.set_xlabel("HW 2Q gates")
    ax.set_ylabel("Min fidelity")
    ax.set_xlim(0, 160)
    ax.set_ylim(0.88, 1.005)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.axvspan(0, 110, color="#dff0d8", alpha=0.4)
    ax.axhline(1e-1, color="orange", linestyle=":", alpha=0.7)
    ax.axhline(1e-2, color="green", linestyle=":", alpha=0.7)
    for c in candidates:
        ax.scatter(c.hw_2q, c.delta_max, s=90, color=colors.get(c.family, "black"), edgecolors="black", linewidths=0.4)
        ax.annotate(c.label, (c.hw_2q, c.delta_max), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_title("Shortlist: 2Q cost vs max |ΔE|")
    ax.set_xlabel("HW 2Q gates")
    ax.set_ylabel("Max |ΔE|")
    ax.set_xlim(0, 160)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1.0)
    ax.grid(alpha=0.3)

    fig.suptitle("QPU-relevant driven shortlist only", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


"fig = claude_shortlist_traces(conditions)"
def _render_claude_pages(pdf: Any, claude_t1: Any, claude_t2: Any) -> None:
    plt = get_plt()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5))

    t1_times, t1_exact_e, t1_variant_e = _reconstruct_claude_energy_traces(claude_t1)
    t2_times, t2_exact_e, t2_variant_e = _reconstruct_claude_energy_traces(claude_t2)

    t1_2 = claude_t1.trace_variants["weight_sorted_2step_full"]
    t1_3 = claude_t1.trace_variants["weight_sorted_3step_full"]
    t2_3 = claude_t2.trace_variants["weight_sorted_3step_full"]

    ax = axes[0, 0]
    ax.plot(t1_times, np.asarray(t1_2["fidelity_trajectory"], dtype=float), color="tab:green", linewidth=1.8, label="2-step full")
    ax.plot(t1_times, np.asarray(t1_3["fidelity_trajectory"], dtype=float), color="tab:blue", linewidth=1.8, label="3-step full")
    ax.axhline(0.9, color="orange", linestyle=":", alpha=0.6)
    ax.axhline(0.98, color="green", linestyle=":", alpha=0.5)
    ax.set_title("Claude driven t=1: fidelity")
    ax.set_xlabel("Time")
    ax.set_ylabel("Fidelity")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[0, 1]
    ax.plot(t1_times, t1_exact_e, color="black", linewidth=2.0, label="exact")
    ax.plot(t1_times, t1_variant_e["weight_sorted_2step_full"], color="tab:green", linewidth=1.8, label="2-step full")
    ax.plot(t1_times, t1_variant_e["weight_sorted_3step_full"], color="tab:blue", linewidth=1.8, label="3-step full")
    ax.set_title("Claude driven t=1: total energy")
    ax.set_xlabel("Time")
    ax.set_ylabel("Total energy")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    ax = axes[1, 0]
    ax.plot(t2_times, np.asarray(t2_3["fidelity_trajectory"], dtype=float), color="tab:blue", linewidth=1.8, label="3-step full")
    ax.axhline(0.9, color="orange", linestyle=":", alpha=0.6)
    ax.set_title("Claude driven t=2: fidelity")
    ax.set_xlabel("Time")
    ax.set_ylabel("Fidelity")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1, 1]
    ax.plot(t2_times, t2_exact_e, color="black", linewidth=2.0, label="exact")
    ax.plot(t2_times, t2_variant_e["weight_sorted_3step_full"], color="tab:blue", linewidth=1.8, label="3-step full")
    ax.set_title("Claude driven t=2: total energy")
    ax.set_xlabel("Time")
    ax.set_ylabel("Total energy")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    fig.suptitle("Claude shortlist trajectories (driven only)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


"fig = our_shortlist_traces(runs, compile_scouts)"
def _render_our_pages(pdf: Any, *, run_locked: Any, run_pareto: Any, compile_locked: Mapping[str, Any], compile_pareto: Mapping[str, Any]) -> None:
    plt = get_plt()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5))

    ax = axes[0, 0]
    ax.plot(run_locked.times, run_locked.fidelity, color="tab:blue", linewidth=1.8, label="locked7")
    ax.plot(run_pareto.times, run_pareto.fidelity, color="tab:green", linewidth=1.8, label="pareto")
    ax.axhline(0.99, color="green", linestyle=":", alpha=0.5)
    ax.axhline(0.9, color="orange", linestyle=":", alpha=0.6)
    ax.set_title("Our driven McLachlan: fidelity")
    ax.set_xlabel("Time")
    ax.set_ylabel("Fidelity vs exact")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[0, 1]
    ax.plot(run_locked.times, run_locked.energy_reference, color="black", linewidth=2.0, label="exact")
    ax.plot(run_locked.times, run_locked.energy_ansatz, color="tab:blue", linewidth=1.6, label="locked7")
    ax.plot(run_pareto.times, run_pareto.energy_ansatz, color="tab:green", linewidth=1.6, label="pareto")
    ax.set_title("Our driven McLachlan: total energy")
    ax.set_xlabel("Time")
    ax.set_ylabel("Total energy")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    ax = axes[1, 0]
    labels = ["locked7", "pareto"]
    x = np.arange(len(labels))
    width = 0.35
    vals_2q = [int(_selected_backend(compile_locked).get("compiled_count_2q", 0)), int(_selected_backend(compile_pareto).get("compiled_count_2q", 0))]
    vals_depth = [int(_selected_backend(compile_locked).get("compiled_depth", 0)), int(_selected_backend(compile_pareto).get("compiled_depth", 0))]
    ax.bar(x - width / 2, vals_2q, width=width, color="tab:blue", alpha=0.8, label="HW 2Q")
    ax.bar(x + width / 2, vals_depth, width=width, color="tab:green", alpha=0.8, label="Depth")
    ax.axhline(100, color="red", linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Our per-shot hardware cost")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1, 1]
    oracle_evals = [int(run_locked.summary.get("oracle_evaluations_total", 0)), int(run_pareto.summary.get("oracle_evaluations_total", 0))]
    ax.bar(labels, oracle_evals, color=["tab:blue", "tab:green"], alpha=0.85)
    ax.set_yscale("log")
    ax.set_title("Our controller workload")
    ax.set_ylabel("Oracle evaluations total")
    ax.grid(alpha=0.3, axis="y")
    for idx, value in enumerate(oracle_evals):
        ax.text(idx, value * 1.1, str(value), ha="center", va="bottom", fontsize=8)

    fig.suptitle("Our shortlisted driven McLachlan runs", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


"pdf_path = build_qpu_relevant_report(...)"
def build_report(
    *,
    claude_driven_t1_json: Path,
    claude_driven_t2_json: Path,
    our_progress_json: Path,
    compile_locked_json: Path,
    compile_pareto_json: Path,
    output_pdf: Path,
) -> Path:
    require_matplotlib()
    PdfPages = get_PdfPages()

    claude_t1 = _load_claude_condition(claude_driven_t1_json, label="Claude driven t=1.0")
    claude_t2 = _load_claude_condition(claude_driven_t2_json, label="Claude driven t=2.0")
    progress = _read_json(our_progress_json)
    our_runs = [
        _load_our_run(_resolve_path(run["output_json"]), label=str(run["name"]))
        for run in progress.get("runs", [])
        if isinstance(run, Mapping) and str(run.get("status", "")) == "completed"
    ]
    our_runs_by_name = {run.label: run for run in our_runs}
    compile_locked = _read_json(compile_locked_json)
    compile_pareto = _read_json(compile_pareto_json)

    candidates = _build_shortlist(
        claude_t1=claude_t1,
        claude_t2=claude_t2,
        our_runs_by_name=our_runs_by_name,
        compile_locked=compile_locked,
        compile_pareto=compile_pareto,
    )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(output_pdf)) as pdf:
        render_parameter_manifest(
            pdf,
            model="Hubbard-Holstein",
            ansatz="Driven shortlist: Claude Suzuki-2 Trotter + our fixed-manifold McLachlan",
            drive_enabled=True,
            t=1.0,
            U=4.0,
            dv=0.0,
            extra={
                "L": 2,
                "omega0": 1.0,
                "g_ep": 0.5,
                "Target region": "fidelity ~0.9+, |ΔE| ~1e-1 to 1e-2, ~100 2Q when possible",
                "Claude drive": "A=0.3, omega=2.0, tbar=3.0",
                "Our drive": "A=0.6, omega=1.0, tbar=1.0, staggered",
            },
            command=current_command_string(),
        )
        render_text_page(pdf, _summary_lines(candidates), fontsize=9, line_spacing=0.03)

        plt = get_plt()
        fig, ax = plt.subplots(figsize=(11.0, 4.8))
        render_compact_table(
            ax,
            title="Shortlisted QPU-relevant driven candidates",
            col_labels=["Candidate", "Method", "Horizon", "HW 2Q", "Depth", "Min fidelity", "Max |ΔE|", "Note"],
            rows=_shortlist_rows(candidates),
            fontsize=6,
        )
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        _render_shortlist_scatter(pdf, candidates)
        _render_claude_pages(pdf, claude_t1, claude_t2)
        _render_our_pages(
            pdf,
            run_locked=our_runs_by_name["driven_locked7_short"],
            run_pareto=our_runs_by_name["driven_pareto_short"],
            compile_locked=compile_locked,
            compile_pareto=compile_pareto,
        )
        render_command_page(
            pdf,
            current_command_string(),
            script_name="pipelines/hardcoded/hh_qpu_relevant_driven_report.py",
            extra_header_lines=[
                f"Claude driven t=1 JSON: {claude_driven_t1_json}",
                f"Claude driven t=2 JSON: {claude_driven_t2_json}",
                f"Our progress JSON: {our_progress_json}",
                f"Compile scout locked7 JSON: {compile_locked_json}",
                f"Compile scout pareto JSON: {compile_pareto_json}",
            ],
        )
    return output_pdf


"args = parse_cli(argv)"
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a concise driven-only QPU-relevant time-dynamics PDF.")
    parser.add_argument("--claude-driven-t1-json", type=Path, default=_DEFAULT_CLAUDE_DRIVEN_T1)
    parser.add_argument("--claude-driven-t2-json", type=Path, default=_DEFAULT_CLAUDE_DRIVEN_T2)
    parser.add_argument("--our-progress-json", type=Path, default=_DEFAULT_OURS_PROGRESS)
    parser.add_argument("--compile-locked-json", type=Path, default=_DEFAULT_COMPILE_LOCKED)
    parser.add_argument("--compile-pareto-json", type=Path, default=_DEFAULT_COMPILE_PARETO)
    parser.add_argument("--output-pdf", type=Path, default=_DEFAULT_OUTPUT)
    return parser.parse_args(argv)


"exit_code = main(argv)"
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_pdf = build_report(
        claude_driven_t1_json=_resolve_path(args.claude_driven_t1_json),
        claude_driven_t2_json=_resolve_path(args.claude_driven_t2_json),
        our_progress_json=_resolve_path(args.our_progress_json),
        compile_locked_json=_resolve_path(args.compile_locked_json),
        compile_pareto_json=_resolve_path(args.compile_pareto_json),
        output_pdf=_resolve_path(args.output_pdf),
    )
    print(f"Wrote PDF: {output_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
