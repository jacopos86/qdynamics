#!/usr/bin/env python3
"""Render a warm-start-focused PDF from L=3 HH accessibility sweep JSON.

Input contract:
- artifacts/tmp/l3_hh_accessibility_fixes_runs.json
- row selected by --run-id (e.g., fix1_warm_start_B or fix1_warm_start_C)

Report intent:
- Explicitly surface HVA warm-start stage details that precede ADAPT iteration 0.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reports.pdf_utils import (
    current_command_string,
    get_PdfPages,
    get_plt,
    render_command_page,
    render_compact_table,
    render_parameter_manifest,
    render_text_page,
    require_matplotlib,
)


DEFAULT_INPUT = REPO_ROOT / "artifacts" / "tmp" / "l3_hh_accessibility_fixes_runs.json"


def _fmt_float(v: Any, fmt: str = ".6e", fallback: str = "N/A") -> str:
    try:
        if v is None:
            return fallback
        x = float(v)
        if not math.isfinite(x):
            return fallback
        return format(x, fmt)
    except Exception:
        return fallback


def _fmt_int(v: Any, fallback: str = "N/A") -> str:
    try:
        if v is None:
            return fallback
        return str(int(v))
    except Exception:
        return fallback


def _load_row(path: Path, run_id: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise KeyError("Expected top-level key 'rows' as list")
    for row in rows:
        if str(row.get("run_id", "")) == str(run_id):
            return row
    raise KeyError(f"run_id not found: {run_id}")


def _render_warmstart_page(pdf: Any, row: dict[str, Any]) -> None:
    ws = row.get("warm_start", {})
    result = row.get("result", {})
    exact = row.get("E_exact_sector")
    if not isinstance(ws, dict) or not ws:
        raise KeyError("Selected row has no warm_start block")
    if not isinstance(result, dict) or not result:
        raise KeyError("Selected row has no result block")

    e_warm = ws.get("E_warm")
    e_best = result.get("E_best")
    d_warm = None
    d_final = None
    rel_warm = None
    rel_final = None
    if exact is not None and e_warm is not None:
        d_warm = abs(float(e_warm) - float(exact))
        rel_warm = d_warm / max(abs(float(exact)), 1e-14)
    if exact is not None and e_best is not None:
        d_final = abs(float(e_best) - float(exact))
        rel_final = d_final / max(abs(float(exact)), 1e-14)
    improve_abs = None
    improve_rel = None
    if d_warm is not None and d_final is not None:
        improve_abs = d_warm - d_final
        improve_rel = improve_abs / max(d_warm, 1e-14)

    lines = [
        "Warm-Start (HVA) Stage and ADAPT Outcome",
        "",
        f"run_id: {row.get('run_id', 'N/A')}",
        f"status: {row.get('status', 'N/A')}",
        "",
        "HVA warm-start pre-optimization (before ADAPT step 0):",
        f"  E_exact_sector:             {_fmt_float(exact, '.12f')}",
        f"  E_warm:                     {_fmt_float(e_warm, '.12f')}",
        f"  warm_delta_E_abs:           {_fmt_float(d_warm, '.6e')}",
        f"  warm_relative_error_abs:    {_fmt_float(rel_warm, '.6e')}",
        f"  reps:                       {_fmt_int(ws.get('reps'))}",
        f"  restarts:                   {_fmt_int(ws.get('restarts'))}",
        f"  maxiter:                    {_fmt_int(ws.get('maxiter'))}",
        f"  warm_num_parameters:        {_fmt_int(ws.get('warm_num_parameters'))}",
        f"  warm_nfev:                  {_fmt_int(ws.get('warm_nfev'))}",
        f"  warm_nit:                   {_fmt_int(ws.get('warm_nit'))}",
        f"  warm_runtime_s:             {_fmt_float(ws.get('warm_runtime_s'), '.2f')}",
        "",
        "ADAPT stage (starting from warm state):",
        f"  E_best:                     {_fmt_float(e_best, '.12f')}",
        f"  E_last:                     {_fmt_float(result.get('E_last'), '.12f')}",
        f"  final_delta_E_abs:          {_fmt_float(d_final, '.6e')}",
        f"  final_relative_error_abs:   {_fmt_float(rel_final, '.6e')}",
        f"  adapt_depth_reached:        {_fmt_int(result.get('adapt_depth_reached'))}",
        f"  adapt_stop_reason:          {result.get('adapt_stop_reason', 'N/A')}",
        f"  num_parameters:             {_fmt_int(result.get('num_parameters'))}",
        f"  nfev_total:                 {_fmt_int(result.get('nfev_total'))}",
        f"  nit_total:                  {_fmt_int(result.get('nit_total'))}",
        f"  runtime_s:                  {_fmt_float(result.get('runtime_s'), '.2f')}",
        "",
        "Warm-start -> final improvement:",
        f"  improvement_abs:            {_fmt_float(improve_abs, '.6e')}",
        f"  improvement_rel:            {_fmt_float(improve_rel, '.6e')}",
        "",
        "Interpretation:",
        "  ADAPT iteration 0 energy is post-warm-start, not the bare reference-state energy.",
    ]
    render_text_page(pdf, lines, fontsize=10, line_spacing=0.03, max_line_width=120)


def _render_stage_context_page(pdf: Any, row: dict[str, Any]) -> None:
    cfg = row.get("config", {})
    rung = row.get("rung", {})
    grad0 = row.get("step0_top_gradients", [])
    fam_lines = []
    if isinstance(grad0, list):
        for g in grad0[:10]:
            fam_lines.append(
                f"  idx={g.get('idx', 'N/A')} "
                f"family={g.get('family', 'N/A')} "
                f"|grad|={_fmt_float(g.get('abs_gradient'), '.3e')} "
                f"label={g.get('label', 'N/A')}"
            )

    lines = [
        "Run Context",
        "",
        "Configuration:",
        f"  L={cfg.get('L', 'N/A')} sector={cfg.get('sector', 'N/A')}",
        f"  t={cfg.get('t', 'N/A')} U={cfg.get('U', 'N/A')} dv={cfg.get('dv', 'N/A')}",
        f"  omega0={cfg.get('omega0', 'N/A')} g_ep={cfg.get('g_ep', 'N/A')} n_ph_max={cfg.get('n_ph_max', 'N/A')}",
        f"  boson_encoding={cfg.get('boson_encoding', 'N/A')} ordering={cfg.get('ordering', 'N/A')} boundary={cfg.get('boundary', 'N/A')}",
        f"  pool_name={cfg.get('pool_name', 'N/A')} pool_size={cfg.get('pool_size', 'N/A')}",
        f"  paop_r={cfg.get('paop_r', 'N/A')} split={cfg.get('paop_split_paulis', 'N/A')} prune_eps={cfg.get('paop_prune_eps', 'N/A')} norm={cfg.get('paop_normalization', 'N/A')}",
        "",
        "ADAPT rung:",
        f"  adapt_max_depth={rung.get('adapt_max_depth', 'N/A')} adapt_maxiter={rung.get('adapt_maxiter', 'N/A')}",
        f"  eps_grad={rung.get('eps_grad', 'N/A')} eps_energy={rung.get('eps_energy', 'N/A')}",
        "",
        "Step-0 gradient snapshot (top entries):",
    ] + fam_lines

    render_text_page(pdf, lines, fontsize=10, line_spacing=0.03, max_line_width=120)


def _float_list(vals: Any) -> list[float]:
    out: list[float] = []
    if not isinstance(vals, list):
        return out
    for v in vals:
        try:
            x = float(v)
        except Exception:
            continue
        if math.isfinite(x):
            out.append(x)
    return out


def _render_trace_plot_page(pdf: Any, row: dict[str, Any]) -> None:
    plt = get_plt()
    result = row.get("result", {})
    exact = row.get("E_exact_sector")
    ws = row.get("warm_start", {})

    hist = _float_list(result.get("history"))
    trace = result.get("selected_trace")
    if not isinstance(trace, list):
        trace = []

    grad_vals: list[float] = []
    step_delta: list[float] = []
    for t in trace:
        if not isinstance(t, dict):
            continue
        try:
            g = float(t.get("max_abs_grad_among_candidates"))
            if math.isfinite(g):
                grad_vals.append(g)
        except Exception:
            pass
        try:
            d = float(t.get("delta_E_step"))
            if math.isfinite(d):
                step_delta.append(d)
        except Exception:
            pass

    fig, axs = plt.subplots(2, 2, figsize=(12.8, 8.2))
    ax00, ax01, ax10, ax11 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

    if hist:
        x = list(range(len(hist)))
        ax00.plot(x, hist, color="#0b5d1e", lw=1.8, label="E(history)")
        if exact is not None:
            ex = float(exact)
            ax00.axhline(ex, color="#7f3c8d", ls="--", lw=1.2, label="E_exact")
        if ws.get("E_warm") is not None:
            ax00.scatter([0], [float(ws["E_warm"])], color="#d1495b", s=28, zorder=3, label="E_warm")
        ax00.set_title("Energy vs ADAPT index")
        ax00.set_xlabel("ADAPT index (0 is post-warm-start)")
        ax00.set_ylabel("Energy")
        ax00.grid(alpha=0.3)
        ax00.legend(fontsize=8)
    else:
        ax00.text(0.5, 0.5, "No energy history", ha="center", va="center")
        ax00.axis("off")

    if hist and exact is not None:
        err = [abs(v - float(exact)) for v in hist]
        x = list(range(len(err)))
        ax01.plot(x, err, color="#1f77b4", lw=1.8)
        if max(err) / max(min(err), 1e-15) >= 100.0 and min(err) > 0.0:
            ax01.set_yscale("log")
        ax01.set_title(r"$|E - E_{exact}|$ vs ADAPT index")
        ax01.set_xlabel("ADAPT index")
        ax01.set_ylabel("absolute error")
        ax01.grid(alpha=0.3)
    else:
        ax01.text(0.5, 0.5, "No exact-aligned error trace", ha="center", va="center")
        ax01.axis("off")

    if grad_vals:
        x = list(range(1, len(grad_vals) + 1))
        ax10.plot(x, grad_vals, color="#e66100", lw=1.8)
        if max(grad_vals) / max(min([g for g in grad_vals if g > 0.0] + [1e-15]), 1e-15) >= 100.0:
            ax10.set_yscale("log")
        ax10.set_title("Selected-step max|gradient|")
        ax10.set_xlabel("Depth")
        ax10.set_ylabel("max |gradient|")
        ax10.grid(alpha=0.3)
    else:
        ax10.text(0.5, 0.5, "No gradient trace", ha="center", va="center")
        ax10.axis("off")

    if step_delta:
        x = list(range(1, len(step_delta) + 1))
        ax11.bar(x, step_delta, color="#0072b2", alpha=0.8)
        ax11.axhline(0.0, color="black", lw=0.8)
        ax11.set_title(r"Per-step $\Delta E$ (E_after - E_before)")
        ax11.set_xlabel("Depth")
        ax11.set_ylabel("delta_E_step")
        ax11.grid(alpha=0.25, axis="y")
    else:
        ax11.text(0.5, 0.5, "No per-step delta trace", ha="center", va="center")
        ax11.axis("off")

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _render_operator_mix_page(pdf: Any, row: dict[str, Any]) -> None:
    plt = get_plt()
    result = row.get("result", {})
    fam_counts = result.get("selected_family_counts", {})
    trace = result.get("selected_trace")
    if not isinstance(trace, list):
        trace = []

    step0 = row.get("step0_top_gradients", [])
    if not isinstance(step0, list):
        step0 = []

    fig, axs = plt.subplots(1, 2, figsize=(12.8, 5.8))
    ax0, ax1 = axs[0], axs[1]

    if isinstance(fam_counts, dict) and fam_counts:
        fam = list(fam_counts.keys())
        vals = [int(fam_counts[k]) for k in fam]
        ax0.bar(fam, vals, color=["#3b8bba", "#e66100", "#4daf4a", "#984ea3"][: len(fam)])
        ax0.set_title("Selected operator family counts")
        ax0.set_ylabel("count")
        ax0.tick_params(axis="x", rotation=20)
        ax0.grid(alpha=0.25, axis="y")
    else:
        ax0.text(0.5, 0.5, "No family-count data", ha="center", va="center")
        ax0.axis("off")

    top = []
    for g in step0[:10]:
        if not isinstance(g, dict):
            continue
        lab = str(g.get("label", "unknown"))
        val = g.get("abs_gradient", None)
        try:
            valf = float(val)
        except Exception:
            continue
        if not math.isfinite(valf):
            continue
        top.append((lab, valf))

    if top:
        labels = [t[0].split(":")[-1] for t in top]
        vals = [t[1] for t in top]
        y = list(range(len(vals)))
        ax1.barh(y, vals, color="#d95f02")
        ax1.set_yticks(y)
        ax1.set_yticklabels(labels, fontsize=7)
        ax1.invert_yaxis()
        ax1.set_title("Step-0 top |gradient| operators")
        ax1.set_xlabel("abs_gradient")
        ax1.grid(alpha=0.25, axis="x")
    else:
        ax1.text(0.5, 0.5, "No step-0 gradient data", ha="center", va="center")
        ax1.axis("off")

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _render_sector_runtime_page(pdf: Any, row: dict[str, Any]) -> None:
    plt = get_plt()
    result = row.get("result", {})
    ws = row.get("warm_start", {})
    sd = result.get("sector_diag", {})

    fig, axs = plt.subplots(1, 2, figsize=(12.8, 5.8))
    ax0, ax1 = axs[0], axs[1]

    n_up_t = sd.get("N_up_target")
    n_up_e = sd.get("N_up_expect")
    n_dn_t = sd.get("N_dn_target")
    n_dn_e = sd.get("N_dn_expect")

    vals = []
    labels = []
    for name, v in [
        ("N_up_target", n_up_t),
        ("N_up_expect", n_up_e),
        ("N_dn_target", n_dn_t),
        ("N_dn_expect", n_dn_e),
    ]:
        try:
            vf = float(v)
        except Exception:
            continue
        if math.isfinite(vf):
            labels.append(name)
            vals.append(vf)

    if vals:
        ax0.bar(labels, vals, color=["#1b9e77", "#66a61e", "#7570b3", "#e7298a"][: len(vals)])
        ax0.set_title("Sector diagnostics: target vs expectation")
        ax0.tick_params(axis="x", rotation=20)
        ax0.grid(alpha=0.25, axis="y")
    else:
        ax0.text(0.5, 0.5, "No sector diagnostics", ha="center", va="center")
        ax0.axis("off")

    runt_labels = []
    runt_vals = []
    for name, v in [
        ("warm_runtime_s", ws.get("warm_runtime_s")),
        ("adapt_runtime_s", result.get("runtime_s")),
    ]:
        try:
            vf = float(v)
        except Exception:
            continue
        if math.isfinite(vf):
            runt_labels.append(name)
            runt_vals.append(vf)

    if runt_vals:
        ax1.bar(runt_labels, runt_vals, color=["#3182bd", "#e6550d"][: len(runt_vals)])
        ax1.set_title("Runtime split")
        ax1.set_ylabel("seconds")
        ax1.grid(alpha=0.25, axis="y")
    else:
        ax1.text(0.5, 0.5, "No runtime data", ha="center", va="center")
        ax1.axis("off")

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _render_selected_trace_table(pdf: Any, row: dict[str, Any]) -> None:
    trace = row.get("result", {}).get("selected_trace", [])
    if not isinstance(trace, list) or not trace:
        render_text_page(pdf, ["Selected-trace table unavailable: no selected_trace."])
        return

    headers = ["depth", "family", "max|g|", "delta_E_step", "label"]
    rows: list[list[str]] = []
    for t in trace:
        if not isinstance(t, dict):
            continue
        rows.append(
            [
                str(t.get("depth", "")),
                str(t.get("selected_family", "")),
                _fmt_float(t.get("max_abs_grad_among_candidates"), ".3e"),
                _fmt_float(t.get("delta_E_step"), ".3e"),
                str(t.get("selected_label", "")),
            ]
        )

    if not rows:
        render_text_page(pdf, ["Selected-trace table unavailable: invalid rows."])
        return

    plt = get_plt()
    chunk_size = 18
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i : i + chunk_size]
        fig, ax = plt.subplots(figsize=(15.5, 8.2))
        title = f"Selected operator trace (rows {i+1}-{i+len(chunk)})"
        render_compact_table(ax, title=title, col_labels=headers, rows=chunk, fontsize=6)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def _render_report(row: dict[str, Any], input_json: Path, output_pdf: Path) -> None:
    require_matplotlib()
    PdfPages = get_PdfPages()

    cfg = row.get("config", {})
    ws = row.get("warm_start", {})
    exact = row.get("E_exact_sector")

    extra_manifest = {
        "run_id": row.get("run_id"),
        "fix_id": row.get("fix_id"),
        "rung_id": row.get("rung_id"),
        "status": row.get("status"),
        "sector": cfg.get("sector"),
        "omega0": cfg.get("omega0"),
        "g_ep": cfg.get("g_ep"),
        "n_ph_max": cfg.get("n_ph_max"),
        "boson_encoding": cfg.get("boson_encoding"),
        "ordering": cfg.get("ordering"),
        "boundary": cfg.get("boundary"),
        "pool_name": cfg.get("pool_name"),
        "pool_size": cfg.get("pool_size"),
        "warm_reps": ws.get("reps"),
        "warm_restarts": ws.get("restarts"),
        "warm_maxiter": ws.get("maxiter"),
        "warm_nfev": ws.get("warm_nfev"),
        "warm_runtime_s": ws.get("warm_runtime_s"),
        "exact_filtered_energy": exact,
    }

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(output_pdf)) as pdf:
        render_parameter_manifest(
            pdf,
            model="Hubbard-Holstein",
            ansatz="Warm-start HH-HVA -> ADAPT(PAOP)",
            drive_enabled=False,
            t=float(cfg.get("t", 0.0)),
            U=float(cfg.get("U", 0.0)),
            dv=float(cfg.get("dv", 0.0)),
            extra=extra_manifest,
            command=current_command_string(),
        )
        _render_warmstart_page(pdf, row)
        _render_stage_context_page(pdf, row)
        _render_trace_plot_page(pdf, row)
        _render_operator_mix_page(pdf, row)
        _render_sector_runtime_page(pdf, row)
        _render_selected_trace_table(pdf, row)

        utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        render_command_page(
            pdf,
            current_command_string(),
            script_name="test/render_l3_hh_accessibility_warmstart_pdf.py",
            extra_header_lines=[
                f"Source JSON: {input_json}",
                f"Generated UTC: {utc_now}",
            ],
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render warm-start summary PDF for L=3 HH accessibility run.")
    p.add_argument(
        "--input-json",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Path to accessibility sweep JSON payload.",
    )
    p.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Row run_id (e.g., fix1_warm_start_B).",
    )
    p.add_argument(
        "--output-pdf",
        type=str,
        default="",
        help="Output PDF path. Default: same dir as input-json with '<run-id>_summary.pdf'.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_json = Path(args.input_json).expanduser().resolve()
    row = _load_row(input_json, args.run_id)
    output_pdf = (
        Path(args.output_pdf).expanduser().resolve()
        if str(args.output_pdf).strip()
        else input_json.parent / f"{args.run_id}_summary.pdf"
    )
    _render_report(row=row, input_json=input_json, output_pdf=output_pdf)
    print(f"WROTE {output_pdf}")


if __name__ == "__main__":
    main()
