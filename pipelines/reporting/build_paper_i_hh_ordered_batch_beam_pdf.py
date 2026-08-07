#!/usr/bin/env python3
"""Build the Paper-I HH ordered batch-beam diagnostic PDF."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.reporting.build_paper_i_hh_child_fairness_pdf import (  # noqa: E402
    _compile_local_snake_terminal_cost,
)


BATCH_ID = "paper_i_hh_fullmeta_singleton_symmetry_ordered_batch_beam_allregime_powell_20260703_v1"
DEFAULT_RECORDS_TSV = (
    REPO_ROOT
    / "chtc"
    / "phase3_optuna"
    / "input"
    / BATCH_ID
    / "paper_i_hh_spsa_budget_ladder_records.tsv"
)
DEFAULT_MANIFEST_JSON = (
    REPO_ROOT
    / "chtc"
    / "phase3_optuna"
    / "input"
    / BATCH_ID
    / "paper_i_hh_spsa_budget_ladder_manifest.json"
)
DEFAULT_RESULT_ROOT = (
    REPO_ROOT
    / "output"
    / "chtc_retrievals"
    / "paper_i_hh_ordered_batch_beam_20260703_current_fetch"
    / "raw_outputs"
    / BATCH_ID
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "pdf" / "paper_i_hh_ordered_batch_beam_matrix_20260703"
DEFAULT_STEM = "paper_i_hh_ordered_batch_beam_matrix_20260703"

REGIME_ORDER = (
    "weak-weak",
    "intermediate-weak",
    "strong-weak",
    "weak-strong",
    "intermediate-strong",
    "strong-strong",
)
MODE_ORDER = ("greedy_reduced_plane", "combinatorial_reduced_plane")
MODE_LABEL = {
    "greedy_reduced_plane": "greedy",
    "combinatorial_reduced_plane": "combinatorial",
}
LAMBDA_ORDER = ("0", "0.01", "0.025", "0.1")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _fmt_float(value: float | None, *, sig: int = 3) -> str:
    if value is None:
        return "--"
    if value == 0:
        return "0"
    return f"{value:.{sig}e}"


def _fmt_int(value: int | None) -> str:
    return "--" if value is None else str(value)


def _tex_escape(text: Any) -> str:
    out = str(text)
    return (
        out.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def _cost_int(cost: Mapping[str, Any], *keys: str) -> int | None:
    status = str(cost.get("status") or "").lower()
    source = str(cost.get("compiled_resource_source_kind") or cost.get("source_kind") or "").lower()
    if status not in {"done", "ok"}:
        return None
    if "qiskit" not in source:
        return None
    for key in keys:
        parsed = _int_or_none(cost.get(key))
        if parsed is not None:
            return parsed
    return None


def _history_points(adapt_vqe: Mapping[str, Any]) -> list[tuple[int, float]]:
    history = adapt_vqe.get("history")
    points: list[tuple[int, float]] = []
    if isinstance(history, Sequence) and not isinstance(history, (str, bytes)):
        for index, item in enumerate(history):
            if not isinstance(item, Mapping):
                continue
            x = _int_or_none(
                item.get("iteration")
                if item.get("iteration") is not None
                else item.get("k")
                if item.get("k") is not None
                else item.get("k_iter")
            )
            if x is None:
                x = index + 1
            y = (
                _float_or_none(item.get("delta_abs_current"))
                or _float_or_none(item.get("abs_delta_e"))
                or _float_or_none(item.get("benchmark_target_abs_delta_e_current"))
            )
            if y is not None and y > 0:
                points.append((x, y))
    final_error = _float_or_none(adapt_vqe.get("abs_delta_e"))
    final_iter = len(history) if isinstance(history, Sequence) and not isinstance(history, (str, bytes)) else None
    if final_error is not None and final_error > 0 and final_iter:
        if points and points[-1][0] == final_iter:
            points[-1] = (final_iter, final_error)
        else:
            points.append((final_iter, final_error))
    return points


@dataclass
class LoadedRow:
    record_id: str
    proc: int
    regime: str
    mode: str
    lambda_beam: str
    status: str
    k_iter: int | None = None
    d_ans: int | None = None
    abs_delta_e: float | None = None
    fidelity: float | None = None
    n2q: int | None = None
    d2q: int | None = None
    dc: int | None = None
    s_alg: float | None = None
    max_batch_size: int | None = None
    batching_fired: bool = False
    cost_status: str | None = None
    cost_source: str | None = None
    source_json: str | None = None
    source_dir: str | None = None
    history_points: list[tuple[int, float]] | None = None


def _load_records(records_tsv: Path) -> list[dict[str, str]]:
    with records_tsv.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _load_row(row: Mapping[str, str], proc: int, result_root: Path) -> LoadedRow:
    record_id = str(row["record_id"])
    record_dir = result_root / record_id
    result_path = record_dir / "json" / "result.json"
    base = {
        "record_id": record_id,
        "proc": proc,
        "regime": str(row.get("display_regime") or ""),
        "mode": str(row.get("phase2_batch_selection_mode") or ""),
        "lambda_beam": str(row.get("adapt_beam_lambda") or ""),
    }
    if not result_path.exists():
        return LoadedRow(**base, status="running_or_pending")
    try:
        payload = _read_json(result_path)
        adapt_vqe = payload.get("adapt_vqe") if isinstance(payload, Mapping) else None
        if not isinstance(adapt_vqe, Mapping):
            return LoadedRow(**base, status="failed:missing_adapt_vqe", source_json=_rel(result_path))
    except Exception as exc:
        return LoadedRow(**base, status=f"failed:{type(exc).__name__}", source_json=_rel(result_path))
    cost = _compile_local_snake_terminal_cost(record_dir, result_path)
    history = adapt_vqe.get("history")
    max_batch_size = None
    batching_fired = False
    if isinstance(history, Sequence) and not isinstance(history, (str, bytes)):
        sizes = [
            _int_or_none(item.get("batch_size"))
            for item in history
            if isinstance(item, Mapping) and _int_or_none(item.get("batch_size")) is not None
        ]
        if sizes:
            max_batch_size = max(sizes)
            batching_fired = any(size > 1 for size in sizes)
    return LoadedRow(
        **base,
        status="done",
        k_iter=len(history) if isinstance(history, Sequence) and not isinstance(history, (str, bytes)) else None,
        d_ans=_int_or_none(adapt_vqe.get("ansatz_depth")),
        abs_delta_e=_float_or_none(adapt_vqe.get("abs_delta_e")),
        fidelity=_float_or_none(adapt_vqe.get("exact_state_fidelity")),
        n2q=_cost_int(cost, "compiled_count_2q_total", "N2q"),
        d2q=_cost_int(cost, "compiled_depth_2q_total", "D2q"),
        dc=_cost_int(cost, "compiled_depth_total", "Dc"),
        s_alg=_float_or_none(cost.get("S_alg")),
        max_batch_size=max_batch_size,
        batching_fired=batching_fired,
        cost_status=str(cost.get("status") or ""),
        cost_source=str(cost.get("compiled_resource_source_kind") or cost.get("source_kind") or ""),
        source_json=_rel(result_path),
        source_dir=_rel(record_dir),
        history_points=_history_points(adapt_vqe),
    )


def _plot_regime_mode(rows: Sequence[LoadedRow], regime: str, mode: str, figures_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    plotted = False
    for lam in LAMBDA_ORDER:
        match = next((row for row in rows if row.regime == regime and row.mode == mode and row.lambda_beam == lam), None)
        if match is None or not match.history_points:
            continue
        xs = [x for x, _ in match.history_points]
        ys = [y for _, y in match.history_points]
        ax.plot(xs, ys, marker="o", markersize=2.5, linewidth=1.2, label=f"lambda={lam}")
        plotted = True
    ax.set_title(f"{regime}: {MODE_LABEL.get(mode, mode)}")
    ax.set_xlabel("ADAPT controller iteration k")
    ax.set_ylabel("abs(Delta E)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.set_xlim(left=1, right=30)
    if plotted:
        ax.legend(fontsize=7, frameon=False)
    else:
        ax.text(0.5, 0.5, "pending / not fetched", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    path = figures_dir / f"{regime.replace('-', '_')}__{MODE_LABEL.get(mode, mode)}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_sidecars(
    rows: Sequence[LoadedRow],
    output_dir: Path,
    stem: str,
    manifest: Mapping[str, Any],
    *,
    batch_id: str,
    cluster_id: str,
) -> tuple[Path, Path]:
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    payload = {
        "schema": "paper_i_hh_ordered_batch_beam_present_results_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "batch_id": batch_id,
        "cluster_id": cluster_id,
        "manifest": manifest,
        "rows": [asdict(row) for row in rows],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = [
        "proc",
        "record_id",
        "regime",
        "mode",
        "lambda_beam",
        "status",
        "k_iter",
        "d_ans",
        "abs_delta_e",
        "fidelity",
        "n2q",
        "d2q",
        "dc",
        "s_alg",
        "max_batch_size",
        "batching_fired",
        "cost_status",
        "cost_source",
        "source_json",
        "source_dir",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            data = asdict(row)
            data.pop("history_points", None)
            writer.writerow({field: data.get(field) for field in fields})
    return json_path, csv_path


def _table_rows(rows: Sequence[LoadedRow], regime: str) -> str:
    out: list[str] = []
    for mode in MODE_ORDER:
        for lam in LAMBDA_ORDER:
            row = next((r for r in rows if r.regime == regime and r.mode == mode and r.lambda_beam == lam), None)
            if row is None:
                continue
            out.append(
                " & ".join(
                    [
                        _tex_escape(MODE_LABEL.get(row.mode, row.mode)),
                        _tex_escape(row.lambda_beam),
                        _tex_escape(row.status),
                        _fmt_int(row.k_iter),
                        _fmt_int(row.d_ans),
                        _fmt_float(row.abs_delta_e),
                        _fmt_float(row.fidelity, sig=4),
                        _fmt_int(row.n2q),
                        _fmt_int(row.d2q),
                        _fmt_int(row.dc),
                        _fmt_float(row.s_alg, sig=3),
                        _fmt_int(row.max_batch_size),
                    ]
                )
                + r" \\"
            )
    return "\n".join(out)


def _write_tex(
    rows: Sequence[LoadedRow],
    *,
    output_dir: Path,
    stem: str,
    manifest: Mapping[str, Any],
    figures: Mapping[tuple[str, str], Path],
    json_path: Path,
    csv_path: Path,
    records_tsv: Path,
    result_root: Path,
    batch_id: str,
    cluster_id: str,
    maxiter: int | None,
    max_depth: int | None,
) -> Path:
    done = sum(1 for row in rows if row.status == "done")
    pending = sum(1 for row in rows if row.status != "done")
    tex_path = output_dir / f"{stem}.tex"
    lines: list[str] = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.45in,landscape]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{xcolor}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{4pt}",
        r"\begin{document}",
        "% Machine-readable sidecars:",
        f"% JSON: {_rel(json_path)}",
        f"% CSV: {_rel(csv_path)}",
        r"\section*{Paper-I HH Ordered Batch-Beam Matrix: Present Results}",
        r"\small",
        r"\begin{tabular}{p{0.16\textwidth}p{0.80\textwidth}}",
        r"\toprule",
        f"Batch & \\path{{{batch_id}}} \\\\",
        f"Cluster & {_tex_escape(cluster_id)} \\\\",
        r"Run class & diagnostic / candidate matrix evidence, not promoted \\",
        r"Optimizer & POWELL \\",
        r"Rows & 6 regimes $\times$ 2 batch modes $\times$ 4 lambda values = 48 \\",
        f"Budget & maxiter {_fmt_int(maxiter)}; depth cap {_fmt_int(max_depth)} \\\\",
        r"Pool/child policy & full\_meta unfiltered with HVA included; A1 Phase-III singleton hard\_guard \\",
        r"Batch modes & greedy\_reduced\_plane; combinatorial\_reduced\_plane \\",
        r"lambda\_beam & 0, 0.01, 0.025, 0.1 \\",
        f"Fetched rows in this PDF & {done} done; {pending} running/pending/not fetched \\\\",
        f"Generated UTC & {_tex_escape(datetime.now(timezone.utc).isoformat())} \\\\",
        f"Records TSV & \\path{{{_rel(records_tsv)}}} \\\\",
        f"Fetched root & \\path{{{_rel(result_root)}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\vfill",
        r"Cost columns are terminal Qiskit-compiled SNAKE circuit costs: $N_{2q}$, $D_{2q}$, and $D_c$. "
        r"$S_{\rm alg}$ is the algorithmic work sidecar value emitted by the same terminal-cost reconstruction. "
        r"Iteration plots show post-step ADAPT controller iterations from fetched result histories.",
    ]
    for regime in REGIME_ORDER:
        lines.extend(
            [
                r"\clearpage",
                f"\\section*{{{_tex_escape(regime)}}}",
                r"\begin{minipage}{0.49\textwidth}",
                f"\\includegraphics[width=\\linewidth]{{figures/{figures[(regime, MODE_ORDER[0])].name}}}",
                r"\end{minipage}\hfill",
                r"\begin{minipage}{0.49\textwidth}",
                f"\\includegraphics[width=\\linewidth]{{figures/{figures[(regime, MODE_ORDER[1])].name}}}",
                r"\end{minipage}",
                r"\vspace{0.4em}",
                r"\scriptsize",
                r"\begin{tabular}{lllr r r r r r r r r}",
                r"\toprule",
                r"Mode & $\lambda_{\rm beam}$ & Status & $k_{\rm iter}$ & $d_{\rm ans}$ & absDeltaE & F & $N_{2q}$ & $D_{2q}$ & $D_c$ & $S_{\rm alg}$ & maxB \\",
                r"\midrule",
                _table_rows(rows, regime),
                r"\bottomrule",
                r"\end{tabular}",
            ]
        )
    lines.append(r"\end{document}")
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tex_path


def build_report(
    *,
    records_tsv: Path,
    manifest_json: Path,
    result_root: Path,
    output_dir: Path,
    stem: str,
    cluster_id: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    records = _load_records(records_tsv)
    manifest = _read_json(manifest_json) if manifest_json.exists() else {}
    batch_id = str(manifest.get("batch_id") or (records[0].get("batch_id") if records else BATCH_ID))
    maxiter = _int_or_none(
        manifest.get("ordered_batch_beam_diagnostic", {}).get("maxiter")
        if isinstance(manifest.get("ordered_batch_beam_diagnostic"), Mapping)
        else None
    )
    if maxiter is None and isinstance(manifest.get("source_contract"), Mapping):
        maxiter = _int_or_none(manifest["source_contract"].get("maxiter"))
    max_depth = _int_or_none(
        manifest.get("ordered_batch_beam_diagnostic", {}).get("max_depth")
        if isinstance(manifest.get("ordered_batch_beam_diagnostic"), Mapping)
        else None
    )
    if max_depth is None and isinstance(manifest.get("source_contract"), Mapping):
        max_depth = _int_or_none(manifest["source_contract"].get("max_depth"))
    rows = [_load_row(row, proc=index, result_root=result_root) for index, row in enumerate(records)]
    figures = {
        (regime, mode): _plot_regime_mode(rows, regime, mode, figures_dir)
        for regime in REGIME_ORDER
        for mode in MODE_ORDER
    }
    json_path, csv_path = _write_sidecars(
        rows,
        output_dir,
        stem,
        manifest if isinstance(manifest, Mapping) else {},
        batch_id=batch_id,
        cluster_id=cluster_id,
    )
    tex_path = _write_tex(
        rows,
        output_dir=output_dir,
        stem=stem,
        manifest=manifest if isinstance(manifest, Mapping) else {},
        figures=figures,
        json_path=json_path,
        csv_path=csv_path,
        records_tsv=records_tsv,
        result_root=result_root,
        batch_id=batch_id,
        cluster_id=cluster_id,
        maxiter=maxiter,
        max_depth=max_depth,
    )
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=output_dir,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {
        "pdf": str(output_dir / f"{stem}.pdf"),
        "tex": str(tex_path),
        "json": str(json_path),
        "csv": str(csv_path),
        "done": str(sum(1 for row in rows if row.status == "done")),
        "pending": str(sum(1 for row in rows if row.status != "done")),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-tsv", type=Path, default=DEFAULT_RECORDS_TSV)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument("--cluster-id", default="8637793")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_report(
        records_tsv=args.records_tsv,
        manifest_json=args.manifest_json,
        result_root=args.result_root,
        output_dir=args.output_dir,
        stem=str(args.stem),
        cluster_id=str(args.cluster_id),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
